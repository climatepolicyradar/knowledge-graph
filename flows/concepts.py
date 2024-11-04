import asyncio
import json
import os
from datetime import datetime
from io import BytesIO
from typing import Optional

import boto3
from cpr_sdk.models.search import Concept as VespaConcept
from prefect import flow, get_run_logger, task

from flows.inference import (
    Config,
    determine_document_ids,
    list_bucket_doc_ids,
    load_s3_object,
)
from src.concept import Concept
from src.labelled_passage import LabelledPassage


@flow(log_prints=True)
def get_concept_class_fields():
    logger = get_run_logger()
    concept_fields = Concept.__fields__.keys()
    logger.info(f"Concept class fields: {concept_fields}")


def load_labelled_passage(
    config: Config, document_id: str
) -> tuple[LabelledPassage, datetime]:
    """Download and opens a parser output based on a document ID."""
    s3_object, timestamp = load_s3_object(config, document_id)
    document = LabelledPassage.model_validate_json(s3_object)
    return document, timestamp


def get_parent_concepts_from_labelled_passage(
    concept: Concept,
) -> tuple[list[dict], str]:
    """
    Extract parent concepts from a Concept object.

    Currently we pull the name from the Classifier used to label the passage, this
    doesn't hold the concept id. This is a temporary solution that is not desirable as
    the relationship between concepts can change frequently and thus shouldn't be
    coupled with inference.
    """
    parent_concepts = [{"id": concept.subconcept_of, "name": ""}]
    parent_concept_ids_flat = (
        ",".join([parent_concept["id"] for parent_concept in parent_concepts]) + ","
    )

    return parent_concepts, parent_concept_ids_flat


def store_concepts(
    config: Config,
    concepts: list[VespaConcept],
    document_id: str,
) -> None:
    """Store the concepts in the cache bucket."""
    key = os.path.join(
        config.document_target_prefix,
        f"{document_id}.json",
    )

    data = [concept.model_dump() for concept in concepts]
    body = BytesIO(json.dumps(data).encode("utf-8"))

    s3 = boto3.client("s3", region_name=config.bucket_region)
    s3.put_object(
        Bucket=config.cache_bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
    )


@task
async def run_convert_labelled_passage_to_concept(
    document_id: str, config: Config
) -> None:
    """Load a labelled passage from s3, extract an array of concepts and store it in s3."""
    logger = get_run_logger()

    logger.info(
        "Loading document from s3.", extra={"props": {"document_id": document_id}}
    )
    labelled_passage, timestamp = load_labelled_passage(config, document_id)
    labelled_passage_concept = Concept.model_validate(
        labelled_passage.metadata["concept"]
    )

    logger.info(
        "Converting labelled passage to concept.",
        extra={"props": {"document_id": document_id}},
    )
    concepts = []
    for i, span in enumerate(labelled_passage.spans):
        parent_concepts, parent_concept_ids_flat = (
            get_parent_concepts_from_labelled_passage(concept=labelled_passage_concept)
        )
        concepts.append(
            VespaConcept(
                id=f"{i}.{labelled_passage.metadata['text_block_id']}",
                name=labelled_passage_concept.preferred_label,
                parent_concepts=parent_concepts,
                parent_concept_ids_flat=parent_concept_ids_flat,
                model=span.labellers[0],
                end=span.end_index,
                start=span.start_index,
                timestamp=timestamp,
            )
        )

    logger.info(
        "Writing concepts to s3.", extra={"props": {"document_id": document_id}}
    )
    store_concepts(config, concepts, document_id)

    logger.info(
        "Finished converting labelled passage to concept.",
        extra={"props": {"document_id": document_id}},
    )


@flow
async def convert_labelled_passages_to_concepts(
    document_ids: Optional[list[str]] = None,
    config: Optional[Config] = None,
) -> None:
    """
    Convert labelled passages in s3 to concepts in s3.

    params:
    - document_ids: List of document ids to run inference on
    - config: A Config object, uses the default if not given. Usually
      there is no need to change this outside of local dev
    """
    logger = get_run_logger()

    if not config:
        config = await Config(
            document_source_prefix="labelled_passages",
            document_target_prefix="concepts",
        ).create()

    logger.info(
        "Running conversion of labelled passages to concepts.",
        extra={"props": {"config": config.__dict__}},
    )

    current_bucket_ids = list_bucket_doc_ids(config=config)
    validated_document_ids = determine_document_ids(
        requested_document_ids=document_ids,
        current_bucket_ids=current_bucket_ids,
    )

    subflows = [
        run_convert_labelled_passage_to_concept(
            document_id=document_id,
            config=config,
        )
        for document_id in validated_document_ids
    ]

    await asyncio.gather(*subflows)
