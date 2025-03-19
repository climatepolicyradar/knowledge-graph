import asyncio
import os
from collections import Counter
from dataclasses import dataclass
from typing import Any

from cpr_sdk.models.search import Concept as VespaConcept
from cpr_sdk.models.search import Passage as VespaPassage
from cpr_sdk.search_adaptors import VespaSearchAdapter
from prefect import flow
from prefect.logging import get_run_logger

from flows.boundary import (
    DEFAULT_DOCUMENTS_BATCH_SIZE,
    DEFAULT_INDEXING_TASK_BATCH_SIZE,
    ConceptModel,
    DocumentImporter,
    TextBlockId,
    calculate_concepts_counts_from_results,
    convert_labelled_passage_to_concepts,
    get_vespa_search_adapter,
    index_by_s3,
    load_labelled_passages_by_uri,
    partial_update_text_block,
    s3_paths_or_s3_prefixes,
    update_s3_with_latest_concepts_counts,
)
from flows.inference import DOCUMENT_TARGET_PREFIX_DEFAULT
from flows.utils import (
    SlackNotify,
    iterate_batch,
    remove_translated_suffix,
)
from scripts.cloud import (
    AwsEnv,
    ClassifierSpec,
    get_prefect_job_variable,
)
from scripts.update_classifier_spec import parse_spec_file

CONCEPTS_COUNTS_PREFIX_DEFAULT: str = "concepts_counts"


@dataclass()
class Config:
    """Configuration used across flow runs."""

    cache_bucket: str | None = None
    document_source_prefix: str = DOCUMENT_TARGET_PREFIX_DEFAULT
    concepts_counts_prefix: str = CONCEPTS_COUNTS_PREFIX_DEFAULT
    bucket_region: str = "eu-west-1"
    # An instance of VespaSearchAdapter.
    #
    # E.g.
    #
    # VespaSearchAdapter(
    #   instance_url="https://vespa-instance-url.com",
    #   cert_directory="certs/"
    # )
    vespa_search_adapter: VespaSearchAdapter | None = None
    aws_env: AwsEnv = AwsEnv(os.environ["AWS_ENV"])
    as_deployment: bool = True

    @classmethod
    async def create(cls) -> "Config":
        """Create a new Config instance with initialized values."""
        logger = get_run_logger()

        config = cls()

        if not config.cache_bucket:
            logger.info(
                "no cache bucket provided, getting it from Prefect job variable"
            )
            config.cache_bucket = await get_prefect_job_variable(
                "pipeline_cache_bucket_name"
            )

        return config


def update_concepts_on_existing_vespa_concepts(
    passage: VespaPassage,
    concepts: list[VespaConcept],
) -> list[dict[str, Any]]:
    """
    Update a passage's concepts with the new concepts.

    During the update we remove all the old concepts related to a model. This is as it
    was decided that holding out dated concepts/spans on the passage in Vespa for a
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


# FIXME: From what I can tell this function should be identical to the related function
# deindex.py. In deindex.py we need to remove_translated_suffix from the
# document_import_id and in this function I'm assuming we should add the updates to how
# we calculate concepts counts from results.
@flow
async def run_partial_updates_of_concepts_for_document_passages__update(
    document_importer: DocumentImporter,
    cache_bucket: str,
    concepts_counts_prefix: str,
    vespa_search_adapter: VespaSearchAdapter | None = None,
) -> Counter[ConceptModel]:
    """
    Run partial update for VespaConcepts on text blocks for a document.

    This is done in the document_passage index.

    Assumptions:

    - The ID field of the VespaConcept object holds the
    context of the text block that it relates to. E.g. the concept ID
    1.10 would relate to the text block ID 10.
    """
    logger = get_run_logger()

    cm, vespa_search_adapter = get_vespa_search_adapter(vespa_search_adapter)

    logger.info("getting S3 labelled passages generator")
    document_labelled_passages = load_labelled_passages_by_uri(document_importer[1])

    with cm:
        logger.info("converting labelled passages to Vespa concepts")
        grouped_concepts: dict[TextBlockId, list[VespaConcept]] = {
            labelled_passage.id: convert_labelled_passage_to_concepts(labelled_passage)
            for labelled_passage in document_labelled_passages
        }

        logger.info(
            f"starting partial updates for {len(grouped_concepts)} grouped concepts"
        )

        batches = iterate_batch(
            list(grouped_concepts.items()),
            batch_size=DEFAULT_DOCUMENTS_BATCH_SIZE,
        )

        concepts_counts: Counter[ConceptModel] = Counter()

        for batch_num, batch in enumerate(batches, start=1):
            logger.info(f"processing partial updates batch {batch_num}")

            # We query vespa for document passages that contain a matching import id.
            # The document imported contains the file stem which could contain a
            # translated suffix. We remove this suffix to get the document import id.
            # E.g. CCLW.executive.1.1_translated_en -> CCLW.executive.1.1
            document_import_id = remove_translated_suffix(document_importer[0])

            partial_update_tasks = [
                partial_update_text_block(
                    text_block_id=text_block_id,
                    concepts=concepts,
                    document_import_id=document_import_id,
                    vespa_search_adapter=vespa_search_adapter,
                    update_function=update_concepts_on_existing_vespa_concepts,
                )
                for text_block_id, concepts in batch
            ]

            logger.info(f"gathering partial updates tasks for batch {batch_num}")
            results = await asyncio.gather(
                *partial_update_tasks, return_exceptions=True
            )
            logger.info(
                f"gathered partial {len(results)} updates tasks for batch {batch_num}"
            )

            concepts_counts = calculate_concepts_counts_from_results(results, batch)
            concepts_counts.update(concepts_counts)

            await update_s3_with_latest_concepts_counts(
                document_importer=document_importer,
                concepts_counts=concepts_counts,
                cache_bucket=cache_bucket,
                concepts_counts_prefix=concepts_counts_prefix,
                document_labelled_passages=document_labelled_passages,
            )

    return concepts_counts


@flow(
    on_failure=[SlackNotify.message],
    on_crashed=[SlackNotify.message],
)
async def index_labelled_passages_from_s3_to_vespa(
    classifier_specs: list[ClassifierSpec] | None = None,
    document_ids: list[str] | None = None,
    config: Config | None = None,
    batch_size: int = DEFAULT_DOCUMENTS_BATCH_SIZE,
    indexing_task_batch_size: int = DEFAULT_INDEXING_TASK_BATCH_SIZE,
) -> None:
    """
    Asynchronously index concepts from S3 into Vespa.

    This function retrieves concept documents from files stored in an
    S3 path and indexes them in a Vespa instance. The name of each
    file in the specified S3 path is expected to represent the
    document's import ID.
    """
    logger = get_run_logger()

    if not config:
        logger.info("no config provided, creating one")

        config = await Config.create()
    else:
        logger.info("config provided")
    assert config.cache_bucket

    logger.info(f"running with config: {config}")

    if classifier_specs is None:
        logger.info("no classifier specs. passed in, loading from file")
        classifier_specs = parse_spec_file(config.aws_env)

    logger.info(f"running with classifier specs: {classifier_specs}")

    s3_accessor = s3_paths_or_s3_prefixes(
        classifier_specs,
        document_ids,
        config.cache_bucket,
        config.document_source_prefix,
    )

    logger.info(f"s3_prefixes: {s3_accessor.prefixes}, s3_paths: {s3_accessor.paths}")

    await index_by_s3(
        partial_update_flow=run_partial_updates_of_concepts_for_document_passages__update,
        aws_env=config.aws_env,
        vespa_search_adapter=config.vespa_search_adapter,
        s3_prefixes=s3_accessor.prefixes,
        s3_paths=s3_accessor.paths,
        batch_size=batch_size,
        indexing_task_batch_size=indexing_task_batch_size,
        as_deployment=config.as_deployment,
        cache_bucket=config.cache_bucket,
        concepts_counts_prefix=config.concepts_counts_prefix,
    )
