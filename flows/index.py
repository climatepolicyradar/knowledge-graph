import asyncio
import base64
import json
import os
import tempfile
from pathlib import Path
from typing import Generator, Optional, Union

from cpr_sdk.models.search import Concept as VespaConcept
from cpr_sdk.models.search import Passage as VespaPassage
from cpr_sdk.s3 import _get_s3_keys_with_prefix, _s3_object_read_text
from cpr_sdk.search_adaptors import VespaSearchAdapter
from cpr_sdk.ssm import get_aws_ssm_param
from prefect import flow
from prefect.concurrency.asyncio import concurrency
from prefect.logging import get_logger, get_run_logger

from src.concept import Concept
from src.labelled_passage import LabelledPassage
from src.span import Span


def get_vespa_search_adapter_from_aws_secrets(
    cert_dir: str,
    vespa_instance_url_param_name: str = "VESPA_INSTANCE_URL",
    vespa_public_cert_param_name: str = "VESPA_PUBLIC_CERT_READ",
    vespa_private_key_param_name: str = "VESPA_PRIVATE_KEY_READ",
) -> VespaSearchAdapter:
    """
    Get a VespaSearchAdapter instance by retrieving secrets from AWS Secrets Manager.

    We then save the secrets to local files in the cert_dir directory and instantiate
    the VespaSearchAdapter.
    """
    vespa_instance_url = get_aws_ssm_param(vespa_instance_url_param_name)
    vespa_public_cert_encoded = get_aws_ssm_param(vespa_public_cert_param_name)
    vespa_private_key_encoded = get_aws_ssm_param(vespa_private_key_param_name)

    vespa_public_cert = base64.b64decode(vespa_public_cert_encoded).decode("utf-8")
    vespa_private_key = base64.b64decode(vespa_private_key_encoded).decode("utf-8")

    with open(f"{cert_dir}/cert.pem", "w") as f:
        f.write(vespa_public_cert)
    with open(f"{cert_dir}/key.pem", "w") as f:
        f.write(vespa_private_key)

    return VespaSearchAdapter(instance_url=vespa_instance_url, cert_directory=cert_dir)


def s3_obj_generator(s3_path: str) -> Generator[tuple[str, list[dict]], None, None]:
    """
    A generator that yields objects from an s3 path.

    params:
    - s3_path: The path in s3 to yield objects from.
    """
    object_keys = _get_s3_keys_with_prefix(s3_prefix=s3_path)
    # We retrieve the bucket from the path using the second element in the path (parts[1]).
    # Path("s3://bucket/prefix/file.json").parts -> ('s3:', 'bucket', 'prefix', 'file.json')
    bucket = Path(s3_path).parts[1]
    for key in object_keys:
        obj = _s3_object_read_text(s3_path=(os.path.join("s3://", bucket, key)))
        yield key, json.loads(obj)


def labelled_passages_generator(
    generator_func: Generator,
) -> Generator[tuple[str, list[LabelledPassage]], None, None]:
    """
    A wrapper function for the s3 object generator.

    Converts each yielded object from the generator to a Pydantic LabelledPassage object.
    """
    for s3_key, obj in generator_func:
        yield s3_key, [LabelledPassage(**labelled_passage) for labelled_passage in obj]


def get_document_passages_from_vespa(
    document_import_id: str, vespa_search_adapter: VespaSearchAdapter
) -> list[tuple[str, VespaPassage]]:
    """
    Retrieve all the passages for a document in vespa.

    params:
    - document_import_id: The document import id for a unique family document.
    """
    logger = get_logger()
    logger.info(
        "Getting document passages from vespa.",
        extra={"props": {"document_import_id": document_import_id}},
    )

    vespa_query_response = vespa_search_adapter.client.query(
        yql=(
            # trunk-ignore(bandit/B608)
            "select * from document_passage where family_document_ref contains "
            f'"id:doc_search:family_document::{document_import_id}"'
        )
    )
    logger.info(
        "Vespa search response for document.",
        extra={
            "props": {
                "document_import_id": document_import_id,
                "total_hits": len(vespa_query_response.hits),
            }
        },
    )

    return [
        (passage["id"], VespaPassage.model_validate(passage["fields"]))
        for passage in vespa_query_response.hits
    ]


def get_text_block_id_from_concept(concept: VespaConcept) -> str:
    """Identify the text block id that a concept relates to."""
    return concept.id


def get_model_from_span(span: Span) -> str:
    """
    Get the model used to label the span.

    Labellers are stored in a list, these can contain many labellers as seen in the
    example below, referring to human and machine annotators.

    [
        "alice",
        "bob",
        "68edec6f-fe74-413d-9cf1-39b1c3dad2c0",
        'KeywordClassifier("extreme weather")',
    ]

    In the context of inference the labellers array should only hold the model used to
    label the span as seen in the example below.

    [
        'KeywordClassifier("extreme weather")',
    ]
    """
    if len(span.labellers) != 1:
        raise ValueError(
            f"Span has more than one labeller. Expected 1, got {len(span.labellers)}."
        )
    return span.labellers[0]


def get_passage_for_concept(
    concept: VespaConcept, document_passages: list[tuple[str, VespaPassage]]
) -> Union[tuple[str, VespaPassage], tuple[None, None]]:
    """
    Return the passage and passage id that a concept relates to.

    Concepts relate to a specific passage or text block within a document
    and therefore, we must find the relevant text block to update when running
    partial updates.

    The concept id is assumed to be in the format:
    - ${family_import_id}.${text_block_id}.
    And thus, can be used to identify the relevant passage.
    """
    concept_text_block_id = get_text_block_id_from_concept(concept)

    passage_for_concept = next(
        (
            passage
            for passage in document_passages
            if passage[1].text_block_id == concept_text_block_id
        ),
        None,
    )

    if passage_for_concept:
        return passage_for_concept

    return None, None


def get_updated_passage_concepts_dict(
    passage: VespaPassage, concept: VespaConcept
) -> list[dict]:
    """Update a passage's concepts with the new concept."""
    passage.concepts = list(passage.concepts) if passage.concepts else []
    passage.concepts.append(concept)

    return [concept.model_dump(mode="json") for concept in passage.concepts]


def get_parent_concepts_from_concept(
    concept: Concept,
) -> tuple[list[dict], str]:
    """
    Extract parent concepts from a Concept object.

    Currently we pull the name from the Classifier used to label the passage, this
    doesn't hold the concept id. This is a temporary solution that is not desirable as
    the relationship between concepts can change frequently and thus shouldn't be
    coupled with inference.
    """
    parent_concepts = [
        {"id": subconcept, "name": ""} for subconcept in concept.subconcept_of
    ]
    parent_concept_ids_flat = (
        ",".join([parent_concept["id"] for parent_concept in parent_concepts]) + ","
    )

    return parent_concepts, parent_concept_ids_flat


def convert_labelled_passages_to_concepts(
    labelled_passages: list[LabelledPassage],
) -> list[VespaConcept]:
    """
    Convert a labelled passage to a list of VespaConcept objects.

    The labelled passage contains a list of spans relating to concepts that we must
    convert to vespa Concept objects.
    """
    concepts = []
    for labelled_passage in labelled_passages:
        # The concept used to label the passage holds some information on the parent
        # concepts and thus this is being used as a temporary solution for providing
        # the relationship between concepts.
        concept = Concept.model_validate(labelled_passage.metadata["concept"])
        parent_concepts, parent_concept_ids_flat = get_parent_concepts_from_concept(
            concept=concept
        )

        concepts.extend(
            [
                VespaConcept(
                    id=labelled_passage.id,
                    name=concept.preferred_label,
                    parent_concepts=parent_concepts,
                    parent_concept_ids_flat=parent_concept_ids_flat,
                    model=get_model_from_span(span),
                    end=span.end_index,
                    start=span.start_index,
                    timestamp=labelled_passage.metadata["inference_timestamp"],
                )
                for span in labelled_passage.spans
            ]
        )

    return concepts


@flow
async def run_partial_updates_of_concepts_for_document_passages(
    document_import_id: str,
    document_concepts: list[VespaConcept],
    vespa_search_adapter: VespaSearchAdapter,
) -> None:
    """
    Run partial update for vespa Concepts on a text block in the document_passage index.

    Assumptions:
    - The id field of the Concept object holds the context of the text block that it
    relates to. E.g. the concept id 1.10 would relate to the text block id 10.
    """
    logger = get_run_logger()

    async with concurrency("concept_partial_updates", occupy=10):
        document_passages = get_document_passages_from_vespa(
            document_import_id=document_import_id,
            vespa_search_adapter=vespa_search_adapter,
        )
        if not document_passages:
            logger.error(
                "No hits for document import id in vespa. "
                "Either the document doesn't exist or there are no passages related to "
                "the document.",
                extra={"props": {"document_import_id": document_import_id}},
            )
            raise ValueError(
                f"No passages found for document in vespa - {document_import_id}"
            )

        for concept in document_concepts:
            passage_id, passage_for_concept = get_passage_for_concept(
                concept, document_passages
            )

            if passage_id and passage_for_concept:
                # Extract data ID (last element after "::"), e.g.,
                # "CCLW.executive.10014.4470.623" from passage_id like
                # "id:doc_search:document_passage::CCLW.executive.10014.4470.623".
                vespa_search_adapter.client.update_data(
                    schema="document_passage",
                    namespace="doc_search",
                    data_id=passage_id.split("::")[-1],
                    fields={
                        "concepts": get_updated_passage_concepts_dict(
                            passage_for_concept, concept
                        )
                    },
                )
                logger.info(
                    "Updated concept for passage.",
                    extra={
                        "props": {"passage_id": passage_id, "concept_id": concept.id}
                    },
                )
            else:
                logger.error(
                    "No passages found for concept.",
                    extra={
                        "props": {
                            "concept_id": concept.id,
                            "document_import_id": document_import_id,
                        }
                    },
                )


@flow
async def index_labelled_passages_from_s3_to_vespa(
    s3_path: str,
    vespa_search_adapter: Optional[VespaSearchAdapter] = None,
) -> None:
    """
    Asynchronously index concepts from S3 files into Vespa.

    This function retrieves concept documents from files stored in an S3 path and
    indexes them in a Vespa instance. The name of each file in the specified S3 path is
    expected to represent the document's import ID.

    Assumptions:
    - The S3 file names represent document import IDs.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        if not vespa_search_adapter:
            vespa_search_adapter = get_vespa_search_adapter_from_aws_secrets(
                cert_dir=temp_dir,
                vespa_private_key_param_name="VESPA_PRIVATE_KEY_FULL_ACCESS",
                vespa_public_cert_param_name="VESPA_PUBLIC_CERT_FULL_ACCESS",
            )

        s3_objects = s3_obj_generator(s3_path=s3_path)
        document_labelled_passages = labelled_passages_generator(
            generator_func=s3_objects
        )
        document_concepts = [
            (
                s3_path,
                convert_labelled_passages_to_concepts(
                    labelled_passages=labelled_passages
                ),
            )
            for s3_path, labelled_passages in document_labelled_passages
        ]

        indexing_tasks = [
            run_partial_updates_of_concepts_for_document_passages(
                document_import_id=Path(s3_key).stem,
                document_concepts=concepts,
                vespa_search_adapter=vespa_search_adapter,
            )
            for s3_key, concepts in document_concepts
        ]

        await asyncio.gather(*indexing_tasks)
