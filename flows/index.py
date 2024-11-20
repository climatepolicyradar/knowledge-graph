import asyncio
import base64
import contextlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional, Union

import boto3
from cpr_sdk.models.search import Concept as VespaConcept
from cpr_sdk.models.search import Passage as VespaPassage
from cpr_sdk.s3 import _get_s3_keys_with_prefix, _s3_object_read_text
from cpr_sdk.search_adaptors import VespaSearchAdapter
from cpr_sdk.ssm import get_aws_ssm_param
from prefect import flow
from prefect.concurrency.asyncio import concurrency
from prefect.logging import get_logger, get_run_logger

from flows.inference import DOCUMENT_TARGET_PREFIX_DEFAULT
from scripts.cloud import ClassifierSpec, get_prefect_job_variable
from src.concept import Concept
from src.labelled_passage import LabelledPassage
from src.span import Span


@dataclass()
class Config:
    """Configuration used across flow runs."""

    cache_bucket: Optional[str] = None
    document_source_prefix: str = DOCUMENT_TARGET_PREFIX_DEFAULT
    bucket_region: str = "eu-west-1"
    # An instance of VespaSearchAdapter.
    #
    # E.g.
    #
    # VespaSearchAdapter(
    #   instance_url="https://vespa-instance-url.com",
    #   cert_directory="certs/"
    # )
    vespa_search_adapter: Optional[VespaSearchAdapter] = None

    @classmethod
    async def create(cls, temp_dir: Optional[str] = None) -> "Config":
        """Create a new Config instance with initialized values."""
        config = cls()

        if not config.cache_bucket:
            config.cache_bucket = await get_prefect_job_variable(
                "pipeline_cache_bucket_name"
            )

        if not config.vespa_search_adapter:
            config.vespa_search_adapter = get_vespa_search_adapter_from_aws_secrets(
                cert_dir=temp_dir,
                vespa_private_key_param_name="VESPA_PRIVATE_KEY_FULL_ACCESS",
                vespa_public_cert_param_name="VESPA_PUBLIC_CERT_FULL_ACCESS",
            )

        return config


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
    cert_dir_path = Path(cert_dir)
    if not cert_dir_path.exists():
        raise FileNotFoundError(f"Certificate directory does not exist: {cert_dir}")

    vespa_instance_url = get_aws_ssm_param(vespa_instance_url_param_name)
    vespa_public_cert_encoded = get_aws_ssm_param(vespa_public_cert_param_name)
    vespa_private_key_encoded = get_aws_ssm_param(vespa_private_key_param_name)

    vespa_public_cert = base64.b64decode(vespa_public_cert_encoded).decode("utf-8")
    vespa_private_key = base64.b64decode(vespa_private_key_encoded).decode("utf-8")

    cert_path = cert_dir_path / "cert.pem"
    key_path = cert_dir_path / "key.pem"

    with open(cert_path, "w", encoding="utf-8") as f:
        f.write(vespa_public_cert)

    with open(key_path, "w", encoding="utf-8") as f:
        f.write(vespa_private_key)

    return VespaSearchAdapter(
        instance_url=vespa_instance_url,
        cert_directory=str(cert_dir_path),
    )


def s3_obj_generator_from_s3_prefixes(
    s3_prefixes: List[str],
) -> Generator[tuple[str, List[dict]], None, None]:
    for s3_prefix in s3_prefixes:
        object_keys = _get_s3_keys_with_prefix(s3_prefix=s3_prefix)
        bucket = Path(s3_prefix).parts[1]
        for key in object_keys:
            obj = _s3_object_read_text(s3_path=(os.path.join("s3://", bucket, key)))
            yield key, json.loads(obj)


def s3_obj_generator_from_s3_paths(
    s3_paths: List[str],
) -> Generator[tuple[str, List[dict]], None, None]:
    """
    A generator that yields objects from a list of s3 paths.

    We extract the key from the s3 path by removing the first two elements in the path.
    E.g. "s3://bucket/prefix/file.json" -> "prefix/file.json"

    params:
    - s3_paths: A list of s3 paths to yield objects from.
    """
    for s3_path in s3_paths:
        yield (
            "/".join(Path(s3_path).parts[2:]),
            json.loads(_s3_object_read_text(s3_path=s3_path)),
        )


def labelled_passages_generator(
    generator_func: Generator,
) -> Generator[tuple[str, List[LabelledPassage]], None, None]:
    """
    A wrapper function for the s3 object generator.

    Converts each yielded object from the generator to a Pydantic LabelledPassage object.
    """
    for s3_key, obj in generator_func:
        yield s3_key, [LabelledPassage(**labelled_passage) for labelled_passage in obj]


def get_document_passages_from_vespa(
    document_import_id: str, vespa_search_adapter: VespaSearchAdapter
) -> List[tuple[str, VespaPassage]]:
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


def get_text_block_id_from_labelled_passage(labelled_passage: LabelledPassage) -> str:
    """Identify the text block id that a labelled passage relates to."""
    return labelled_passage.id


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


def get_passage_for_text_block(
    text_block_id: str, document_passages: List[tuple[str, VespaPassage]]
) -> Union[tuple[str, str, VespaPassage], tuple[None, None, None]]:
    """
    Return the data id, passage and passage id that a text block relates to.

    Concepts relate to a specific passage or text block within a document
    and therefore, we must find the relevant text block to update when running
    partial updates.

    Extract data ID (last element after "::"), e.g., "CCLW.executive.10014.4470.623"
    from passage_id like "id:doc_search:document_passage::CCLW.executive.10014.4470.623".
    """
    passage = next(
        (
            passage
            for passage in document_passages
            if passage[1].text_block_id == text_block_id
        ),
        None,
    )

    if passage:
        data_id = passage[0].split("::")[-1]
        passage_id = passage[0]
        passage_content = passage[1]
        return data_id, passage_id, passage_content

    return None, None, None


def get_parent_concepts_from_concept(
    concept: Concept,
) -> tuple[List[dict], str]:
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
    labelled_passages: List[LabelledPassage],
) -> List[tuple[str, VespaConcept]]:
    """
    Convert a labelled passage to a list of VespaConcept objects and their text block id.

    The labelled passage contains a list of spans relating to concepts that we must
    convert to vespa Concept objects.
    """
    concepts = []
    for labelled_passage in labelled_passages:
        # The concept used to label the passage holds some information on the parent
        # concepts and thus this is being used as a temporary solution for providing
        # the relationship between concepts. This has the downside that it ties a
        # labelled passage to a particular concept when in fact the Spans that a
        # labelled passage has can be labelled by multiple concepts.
        concept = Concept.model_validate(labelled_passage.metadata["concept"])
        parent_concepts, parent_concept_ids_flat = get_parent_concepts_from_concept(
            concept=concept
        )
        text_block_id = get_text_block_id_from_labelled_passage(labelled_passage)

        for span in labelled_passage.spans:
            if span.concept_id is None:
                raise ValueError("Concept ID is None.")
            concepts.append(
                (
                    text_block_id,
                    VespaConcept(
                        id=span.concept_id,
                        name=concept.preferred_label,
                        parent_concepts=parent_concepts,
                        parent_concept_ids_flat=parent_concept_ids_flat,
                        model=get_model_from_span(span),
                        end=span.end_index,
                        start=span.start_index,
                        timestamp=labelled_passage.metadata["inference_timestamp"],
                    ),
                )
            )

    return concepts


def get_updated_passage_concepts(
    passage: VespaPassage, concepts: List[VespaConcept]
) -> List[dict]:
    """
    Update a passage's concepts with the new concept.

    During the update we remove all the old concepts related to a model. This is as it
    was decided that holding out dated concepts/spans on the passage in vespa for a
    model is not useful.

    It is also, not possible to duplicate a Concept object in the concepts array as we
    are removing all instances where the model is the same.
    """
    if not passage.concepts:
        return [concept.model_dump(mode="json") for concept in concepts]

    new_concept_models = {concept.model for concept in concepts}

    existing_concepts_to_keep = [
        concept
        for concept in passage.concepts
        if concept.model not in new_concept_models
    ]

    updated_concepts = existing_concepts_to_keep + concepts

    return [concept_.model_dump(mode="json") for concept_ in updated_concepts]


def group_concepts_on_text_block(
    document_concepts: List[tuple[str, VespaConcept]],
) -> dict[str, List[VespaConcept]]:
    """
    Group concepts on text block id.

    Concepts relate to a specific passage or text block within a document and therefore,
    we must group the concept updates to all of them at once.
    """
    concepts_grouped = {}
    for text_block_id, concept in document_concepts:
        if text_block_id in concepts_grouped:
            concepts_grouped[text_block_id].append(concept)
        else:
            concepts_grouped[text_block_id] = [concept]
    return concepts_grouped


@flow
async def run_partial_updates_of_concepts_for_document_passages(
    document_import_id: str,
    document_concepts: List[tuple[str, VespaConcept]],
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

        grouped_concepts = group_concepts_on_text_block(document_concepts)

        for text_block_id, concepts in grouped_concepts.items():
            data_id, passage_id, passage_for_text_block = get_passage_for_text_block(
                text_block_id, document_passages
            )

            if data_id and passage_id and passage_for_text_block:
                vespa_search_adapter.client.update_data(
                    schema="document_passage",
                    namespace="doc_search",
                    data_id=data_id,
                    fields={
                        "concepts": get_updated_passage_concepts(
                            passage_for_text_block, concepts
                        )
                    },
                )
                logger.info(
                    "Updated concepts for passage.",
                    extra={"props": {"passage_id": passage_id}},
                )
            else:
                logger.error(
                    "No passages found for text block.",
                    extra={
                        "props": {
                            "text_block_id": text_block_id,
                        }
                    },
                )


def get_bucket_paginator(config: Config):
    """Return a S3 paginator for the pipeline cache bucket, with a prefix."""
    s3 = boto3.client("s3", region_name=config.bucket_region)
    paginator = s3.get_paginator("list_objects_v2")
    return paginator.paginate(
        Bucket=config.cache_bucket,
        Prefix=config.document_source_prefix,
    )


def list_bucket_doc_ids(config: Config) -> List[str]:
    """Scan configured bucket and return all IDs."""
    page_iterator = get_bucket_paginator(config)
    doc_ids = []

    for p in page_iterator:
        if "Contents" in p:
            for o in p["Contents"]:
                # Get just the stem, which we expect to remove `.json`
                doc_id = Path(o["Key"]).stem
                doc_ids.append(doc_id)

    return doc_ids


def determine_document_ids(
    requested_document_ids: Optional[List[str]],
    current_bucket_ids: List[str],
) -> List[str]:
    """
    Confirm chosen document ids or default to all if not specified.

    Compares the requested_document_ids to what actually exists in the bucket.
    If a document id has been requested but does not exist this will
    raise a `ValueError`.
    """
    missing_from_bucket = list(set(requested_document_ids) - set(current_bucket_ids))
    if len(missing_from_bucket) > 0:
        raise ValueError(
            f"Requested `document_ids` not found in bucket: {missing_from_bucket}"
        )

    return requested_document_ids


def s3_paths_or_s3_prefixes(
    classifier_specs: Optional[List[ClassifierSpec]],
    document_ids: Optional[List[str]],
    config: Config,
) -> tuple[Optional[List[str]], Optional[List[str]]]:
    """
    Return the paths or prefix for the documents and classifiers.

    - s3_prefix: The S3 prefix (directory) to yield objects from. Example:
      "s3://cpr-sandbox-data-pipeline-cache/labelled_passages"
    - s3_paths: A list of S3 object keys to yield objects from. Example:
      [
        "s3://cpr-sandbox-data-pipeline-cache/labelled_passages/Q787/v4/CCLW.executive.1813.2418.json",
        "s3://cpr-sandbox-data-pipeline-cache/labelled_passages/Q787/v4/CCLW.legislative.10695.6015.json",
      ]
    """
    match (classifier_specs, document_ids):
        case (None, None):
            # Run on all documents, regardless of classifier
            s3_prefix = "s3://" + os.path.join(
                config.cache_bucket,
                config.document_source_prefix,
            )
            return None, [s3_prefix]

        case (list(), None):
            # Run on all documents, for the specified classifier
            s3_prefixes = [
                "s3://"
                + os.path.join(
                    config.cache_bucket,
                    config.document_source_prefix,
                    classifier_spec.name,
                    classifier_spec.alias,
                )
                for classifier_spec in classifier_specs
            ]
            return None, s3_prefixes

        case (list(), list()):
            # Run on specified documents, for the specified classifier
            document_paths = [
                "s3://"
                + os.path.join(
                    config.cache_bucket,
                    config.document_source_prefix,
                    classifier_spec.name,
                    classifier_spec.alias,
                    f"{doc_id}.json",
                )
                for classifier_spec in classifier_specs
                for doc_id in document_ids
            ]
            return document_paths, None

        case (None, list()):
            raise ValueError(
                "if document IDs are specified, a classifier "
                "specifcation must also be specified, since they're "
                "namespaced by classifiers (e.g. "
                "`s3://cpr-sandbox-data-pipeline-cache/labelled_passages/Q787/"
                "v4/CCLW.legislative.10695.6015.json`)"
            )


def s3_obj_generator(
    s3_prefixes: Optional[List[str]],
    s3_paths: Optional[List[str]],
):
    match (s3_prefixes, s3_paths):
        case (list(), list()):
            raise ValueError(
                "Either s3_prefixes or s3_paths must be provided, not both."
            )
        case (list(), None):
            return s3_obj_generator_from_s3_prefixes(s3_prefixes=s3_prefixes)
        case (None, list()):
            return s3_obj_generator_from_s3_paths(s3_paths=s3_paths)
        case (None, None):
            raise ValueError("Either s3_prefix or s3_paths must be provided.")
        case (_, _):
            raise ValueError(
                f"Unexpected types: `s3_prefixes={type(s3_prefixes)}`, "
                f"`s3_paths={type(s3_paths)}`"
            )


async def index_by_s3(
    vespa_search_adapter: VespaSearchAdapter,
    s3_prefixes: Optional[List[str]] = None,
    s3_paths: Optional[List[str]] = None,
) -> None:
    """
    Asynchronously index concepts from S3 files into Vespa.

    This function retrieves concept documents from files stored in an S3 path and
    indexes them in a Vespa instance. The name of each file in the specified S3 path is
    expected to represent the document's import ID.

    When `s3_prefix` is provided, the function will index all files within that S3
    prefix (directory). When `s3_paths` is provided, the function will index only the
    files specified in the list of S3 object keys. If both are provided `s3_paths` will
    be used.

    Assumptions:
    - The S3 file names represent document import IDs.

    params:
    - s3_prefix: The S3 prefix (directory) to yield objects from.
        E.g. "s3://bucket/prefix/"
    - s3_paths: A list of S3 object keys to yield objects from.
        E.g. {"s3://bucket/prefix/file1.json", "s3://bucket/prefix/file2.json"}
    - vespa_search_adapter: An instance of VespaSearchAdapter.
        E.g. VespaSearchAdapter(
            instance_url="https://vespa-instance-url.com",
            cert_directory="certs/"
        )
    """
    s3_objects = s3_obj_generator(s3_prefixes, s3_paths)

    document_labelled_passages = labelled_passages_generator(generator_func=s3_objects)
    document_concepts = [
        (
            s3_path,
            convert_labelled_passages_to_concepts(labelled_passages=labelled_passages),
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


@flow
async def index_labelled_passages_from_s3_to_vespa(
    classifier_specs: Optional[List[ClassifierSpec]] = None,
    document_ids: Optional[List[str]] = None,
    config: Optional[Config] = None,
) -> None:
    """
    Asynchronously index concepts from S3 into Vespa.

    This function retrieves concept documents from files stored in an
    S3 path and indexes them in a Vespa instance. The name of each
    file in the specified S3 path is expected to represent the
    document's import ID.
    """
    logger = get_run_logger()

    # We want the directory used for the `VespaSearchAdapter` to be
    # automatically cleaned up.
    #
    # To do this, we rely on the `tempfile.TemporaryDirectory`'s behaviour,
    # or, a `contextlib.nullcontext` no-op, if a temporary directory
    # wasn't needed.
    if not config:
        cm = tempfile.TemporaryDirectory()

        config = await Config.create(temp_dir=cm.name)
    else:
        cm = contextlib.nullcontext()

    with cm:
        logger.info(f"Running with config: {config}")

        s3_paths, s3_prefixes = s3_paths_or_s3_prefixes(
            classifier_specs,
            document_ids,
            config,
        )

        logger.info(
            "S3 prefix and paths",
            extra={"props": {"s3_prefix": s3_prefixes, "s3_paths": s3_paths}},
        )

        await index_by_s3(
            config.vespa_search_adapter,
            s3_prefixes,
            s3_paths,
        )
