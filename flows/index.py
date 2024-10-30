import asyncio
import json
import os
import tempfile
from pathlib import Path
from typing import Generator, Optional, Union

from cpr_sdk.models.search import Concept as VespaConcept
from cpr_sdk.models.search import Passage as VespaPassage
from cpr_sdk.s3 import _get_s3_keys_with_prefix, _s3_object_read_text
from cpr_sdk.search_adaptors import VespaSearchAdapter
from prefect import flow
from prefect.logging import get_logger, get_run_logger

from flows.inference import get_aws_ssm_param


def get_vespa_search_adapter_from_aws_secrets(
    cert_dir: Optional[str] = None,
) -> VespaSearchAdapter:
    """
    Get a VespaSearchAdapter instance by retrieving secrets from AWS Secrets Manager.

    We then save the secrets to local files in the cert_dir directory and instantiate
    the VespaSearchAdapter.
    """
    vespa_instance_url = get_aws_ssm_param("VESPA_INSTANCE_URL")
    vespa_public_cert = get_aws_ssm_param("VESPA_PUBLIC_CERT")
    vespa_private_key = get_aws_ssm_param("VESPA_PRIVATE_KEY")

    if cert_dir:
        os.makedirs(cert_dir, exist_ok=True)
    else:
        cert_dir = tempfile.mkdtemp()

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
    bucket = Path(s3_path).parts[1]
    for key in object_keys:
        obj = _s3_object_read_text(s3_path="s3://" + bucket + "/" + key)
        yield key, json.loads(obj)


def document_concepts_generator(
    generator_func: Generator,
) -> Generator[tuple[str, list[VespaConcept]], None, None]:
    """
    A wrapper function for the s3 object generator.

    Converts each yielded object from the generator to a Pydantic VespaConcept object.
    """
    for s3_key, obj in generator_func:
        yield s3_key, [VespaConcept(**concept) for concept in obj]


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


# TODO: Add test
def get_passage_for_concept(
    concept: VespaConcept, document_passages: list[tuple[str, VespaPassage]]
) -> Union[tuple[str, VespaPassage], tuple[None, None]]:
    """
    Return the passage and passage id that a concept relates to.

    Concepts relate to a specific passage or text block within a document
    and therefore, we must find the relevant text block to update when running partial updates.

    The concept id is assumed to be in the format:
    - ${family_import_id}.${text_block_id}.
    And thus, can be used to identify the relevant passage.
    """
    document_passages_id_map = {
        passage[1].text_block_id: passage for passage in document_passages
    }

    passage_for_concept = document_passages_id_map.get(concept.id.split(".")[-1])

    if passage_for_concept:
        return passage_for_concept

    return None, None


@flow
async def run_partial_updates_of_concepts_for_document_passages_with_semaphore(
    document_import_id: str,
    document_concepts: list[VespaConcept],
    vespa_search_adapter: VespaSearchAdapter,
    semaphore: asyncio.Semaphore,
) -> None:
    """
    Run partial update for vespa Concepts on a text block in the document_passage index in vespa.

    Assumptions:
    - The id of a passage in vespa is suffixed with ${family_import_id}.${text_block_id}.
    E.g. id:doc_search:document_passage::CCLW.executive.1.1.100 would be the id relating
    to the text block id 100 for the document import id CCLW.executive.1.1.
    - The id field of the Concept object holds the context of the text block that it
    relates to. E.g. the concept id 1.10 would relate to the text block id 10.
    """
    logger = get_run_logger()

    async with semaphore:
        document_passages = get_document_passages_from_vespa(
            document_import_id=document_import_id,
            vespa_search_adapter=vespa_search_adapter,
        )
        if document_passages == []:
            logger.error(
                "No hits for document import id in vespa. "
                "Either the document doesn't exist or there are no passages related to the document."
            )
            return

        for concept in document_concepts:
            passage_id, passage_for_concept = get_passage_for_concept(
                concept, document_passages
            )

            if passage_id and passage_for_concept:
                if passage_for_concept.concepts:
                    passage_for_concept.concepts = passage_for_concept.concepts + [
                        concept
                    ]
                    new_concepts = [
                        json.loads(concept.model_dump_json())
                        for concept in passage_for_concept.concepts
                    ]
                else:
                    new_concepts = [json.loads(concept.model_dump_json())]

                vespa_search_adapter.client.update_data(
                    schema="document_passage",
                    namespace="doc_search",
                    data_id=passage_id.split("::")[-1],
                    fields={"concepts": new_concepts},
                )
                logger.info(
                    "Updated concept for passage.",
                    extra={"props": {"concept": concept.model_dump()}},
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
async def index_concepts_from_s3_to_vespa(
    s3_path: str,
    vespa_search_adapter: Optional[VespaSearchAdapter] = None,
    concurrency_limit: int = 10,
) -> None:
    """
    Index vespa Concept objects from s3 into vespa.

    Assumptions:
    - The name of the files in the s3 path are the document import ids.
    """
    if not vespa_search_adapter:
        vespa_search_adapter = get_vespa_search_adapter_from_aws_secrets()
    s3_obj_gen = s3_obj_generator(s3_path=s3_path)
    document_concepts_gen = document_concepts_generator(generator_func=s3_obj_gen)

    semaphore = asyncio.Semaphore(concurrency_limit)

    sub_flows = [
        run_partial_updates_of_concepts_for_document_passages_with_semaphore(
            document_import_id=Path(s3_key).stem,
            document_concepts=document_concepts,
            vespa_search_adapter=vespa_search_adapter,
            semaphore=semaphore,
        )
        for s3_key, document_concepts in document_concepts_gen
    ]

    await asyncio.gather(*sub_flows)
