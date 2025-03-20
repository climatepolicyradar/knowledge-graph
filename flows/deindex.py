import asyncio
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
from cpr_sdk.models.search import Concept as VespaConcept
from cpr_sdk.models.search import Passage as VespaPassage
from cpr_sdk.search_adaptors import VespaSearchAdapter
from prefect import flow
from prefect.logging import get_run_logger

from flows.boundary import (
    ConceptModel,
    DocumentImporter,
    DocumentObjectUri,
    TextBlockId,
    convert_labelled_passage_to_concepts,
    get_vespa_search_adapter,
    index_by_s3,
    load_labelled_passages_by_uri,
    partial_update_text_block,
    s3_object_write_text,
    s3_paths_or_s3_prefixes,
)
from flows.inference import DOCUMENT_TARGET_PREFIX_DEFAULT
from flows.utils import SlackNotify, iterate_batch
from scripts.cloud import (
    AwsEnv,
    ClassifierSpec,
    get_prefect_job_variable,
)
from src.identifiers import WikibaseID
from src.labelled_passage import LabelledPassage
from src.span import Span

DEFAULT_DOCUMENTS_BATCH_SIZE = 500
DEFAULT_INDEXING_TASK_BATCH_SIZE = 20
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


def save_labelled_passages_by_uri(
    document_object_uri: DocumentObjectUri,
    labelled_passages: list[LabelledPassage],
) -> None:
    """Save LabelledPassages objects to S3."""
    object_json = json.dumps(
        [labelled_passage.model_dump_json() for labelled_passage in labelled_passages]
    )

    _ = s3_object_write_text(
        s3_uri=document_object_uri,
        text=object_json,
    )


def remove_concepts_from_existing_vespa_concepts(
    passage: VespaPassage,
    concepts_to_remove: list[VespaConcept],
) -> list[dict[str, Any]]:
    """
    Update a passage's concepts with the updated/removed concepts.

    During the update we remove all the old concepts related to a model. This is as it
    was decided that holding out dated concepts/spans on the passage in Vespa for a
    model is not useful.

    It is also, not possible to duplicate a Concept object in the concepts array as we
    are removing all instances where the model is the same.
    """
    # Get the models to remove
    concepts_to_remove__models = [concept.model for concept in concepts_to_remove]

    # It's an optional sequence at the moment, so massage it
    concepts_in_vespa: list[VespaConcept] = (
        list(passage.concepts) if passage.concepts is not None else []
    )

    # We'll be removing all of the listed concepts, so filter them out
    concepts_in_vespa_to_keep = [
        concept
        for concept in concepts_in_vespa
        if concept.model not in concepts_to_remove__models
    ]

    return [concept_.model_dump(mode="json") for concept_ in concepts_in_vespa_to_keep]


@flow
async def run_partial_updates_of_concepts_for_document_passages__removal(
    document_importer: DocumentImporter,
    cache_bucket: str,
    concepts_counts_prefix: str,
    vespa_search_adapter: VespaSearchAdapter | None = None,
) -> None:
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
        logger.info(
            (
                "getting document passages from Vespa for document "
                f"import ID {document_importer[0]}"
            )
        )

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

        for batch_num, batch in enumerate(batches, start=1):
            logger.info(f"processing partial updates batch {batch_num}")

            partial_update_tasks = [
                partial_update_text_block(
                    text_block_id=text_block_id,
                    document_import_id=document_importer[0],
                    concepts=concepts,
                    vespa_search_adapter=vespa_search_adapter,
                    update_function=remove_concepts_from_existing_vespa_concepts,
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

            await update_s3_with_latest_concepts_counts(
                document_importer=document_importer,
                concepts_counts=concepts_counts,
                cache_bucket=cache_bucket,
                concepts_counts_prefix=concepts_counts_prefix,
                document_labelled_passages=document_labelled_passages,
            )


def calculate_concepts_counts_from_results(
    results: list[BaseException | None],
    batch: list[tuple[TextBlockId, list[VespaConcept]]],
) -> Counter[ConceptModel]:
    logger = get_run_logger()

    # This can handle multiple concepts, but, in practice at the
    # moment, this function is operating on a DocumentImporter,
    # which represents a labelled passages object, which is per
    # concept.
    concepts_counts: Counter[ConceptModel] = Counter()

    for i, result in enumerate(results):
        _text_block_id, concepts = batch[i]

        # Example:
        #
        # ..
        # "labellers": [
        #   "KeywordClassifier(\"professional services sector\")"
        # ],
        # ...
        concepts_models = [
            ConceptModel(wikibase_id=WikibaseID(concept.id), model_name=concept.model)
            for concept in concepts
        ]

        # Set 0s in the counter for all seen concepts. This ensures
        # all concepts are represented in the counter even if they're
        # not updated.
        for concept_model in concepts_models:
            if concept_model not in concepts_counts:
                concepts_counts[concept_model] = 0

        if isinstance(result, Exception):
            # Since we failed to remove them from the spans, make sure
            # they're accounted for as remaining.
            logger.info(f"partial update failed: {str(result)}")
            concepts_counts.update(concepts_models)

    return concepts_counts


async def update_s3_with_latest_concepts_counts(
    document_importer: DocumentImporter,
    concepts_counts: Counter[ConceptModel],
    cache_bucket: str,
    concepts_counts_prefix: str,
    document_labelled_passages: list[LabelledPassage],
) -> None:
    logger = get_run_logger()

    # Ideally, we'd remove the concepts count file entirely, but, we may fail above in updating
    # 1 or more document passages in Vespa, which means that they'd still have the concept present.
    #
    # To avoid a mismatch of the family documents' concepts counts, and what's _still_ reflected on
    # document passages due to failed partial updates, still write an updated concepts counts to
    # S3.
    #
    # However, if we successfully removed all of the concepts from the document passages, then we can
    # delete it. Then, also update the family document's concepts counts to remove it from there.

    # Remove entries with a value of 0 from the counter
    concepts_counts_filtered = Counter(
        {k: v for k, v in concepts_counts.items() if v != 0}
    )

    # If after filtering out, there's no concepts, that means we
    # succeeded in all the partial updates to the document
    # passages.
    if len(concepts_counts_filtered) == 0:
        logger.info("successfully updated all concepts")
        update_s3_with_all_successes(
            document_object_uri=document_importer[1],
            cache_bucket=cache_bucket,
            concepts_counts_prefix=concepts_counts_prefix,
        )
    # We didn't succeed with all, so write the concepts counts still
    else:
        logger.info("only updated some concepts")
        update_s3_with_some_successes(
            document_object_uri=document_importer[1],
            concepts_counts_filtered=concepts_counts_filtered,
            document_labelled_passages=document_labelled_passages,
            cache_bucket=cache_bucket,
            concepts_counts_prefix=concepts_counts_prefix,
        )

    return None


def update_s3_with_all_successes(
    document_object_uri: DocumentObjectUri,
    cache_bucket: str,
    concepts_counts_prefix: str,
) -> None:
    logger = get_run_logger()

    logger.info("updating S3 with all successes")

    s3 = boto3.client("s3")

    s3_uri = Path(document_object_uri)

    # First, delete the concepts counts object
    # Get all parts after the prefix (e.g. "Q787/v4/CCLW.executive.1813.2418.json")
    key_parts = "/".join(s3_uri.parts[3:])  # Skip s3://bucket/labelled_passages/

    concepts_counts_key = f"{concepts_counts_prefix}/{key_parts}"

    _ = s3.delete_object(Bucket=cache_bucket, Key=concepts_counts_key)

    logger.info("updated S3 with deleted concepts counts")

    # Second, delete the labelled passages
    # Get all parts except for the bucket (e.g. "labelled_passages/Q787/v4/CCLW.executive.1813.2418.json")
    labelled_passages_key = "/".join(s3_uri.parts[2:])  # Skip s3://bucket/

    _ = s3.delete_object(Bucket=cache_bucket, Key=labelled_passages_key)

    logger.info("updated S3 with deleted labelled passages")

    logger.info("updated S3 with all successes")

    return None


def serialise_concepts_counts(concepts_counts: Counter[ConceptModel]) -> str:
    return json.dumps({str(k): v for k, v in concepts_counts.items()})


def update_s3_with_some_successes(
    document_object_uri: DocumentObjectUri,
    concepts_counts_filtered: Counter[ConceptModel],
    document_labelled_passages: list[LabelledPassage],
    cache_bucket: str,
    concepts_counts_prefix: str,
) -> None:
    logger = get_run_logger()

    logger.info("updating S3 with partial successes")

    # First, update the concepts counts object
    serialised_concepts_counts = serialise_concepts_counts(concepts_counts_filtered)

    s3_uri = Path(document_object_uri)

    # Get all parts after the prefix (e.g. "Q787/v4/CCLW.executive.1813.2418.json")
    key_parts = "/".join(s3_uri.parts[3:])  # Skip s3://bucket/labelled_passages/

    concepts_counts_uri = f"s3://{cache_bucket}/{concepts_counts_prefix}/{key_parts}"

    _ = s3_object_write_text(
        s3_uri=concepts_counts_uri,
        text=serialised_concepts_counts,
    )

    logger.info("updated S3 with updated concepts counts")

    # Second, update the labelled passages
    concept_ids_to_keep: list[WikibaseID] = [
        concept_model.wikibase_id for concept_model in concepts_counts_filtered
    ]

    filtered_labelled_passages: list[LabelledPassage] = []

    for labelled_passage in document_labelled_passages:
        # It doesn't matter if this list is empty, as it
        # emulates an empty result from the inference
        # pipeline.
        updated_spans: list[Span] = [
            span
            for span in labelled_passage.spans
            if span.concept_id in concept_ids_to_keep
        ]

        labelled_passage.spans = updated_spans

        filtered_labelled_passages.append(labelled_passage)

    _ = save_labelled_passages_by_uri(
        document_object_uri=document_object_uri,
        labelled_passages=filtered_labelled_passages,
    )

    logger.info("updated S3 with updated labelled passages")

    logger.info("updated S3 with partial successes")

    return None


@flow(
    on_failure=[SlackNotify.message],
    on_crashed=[SlackNotify.message],
)
async def deindex_labelled_passages_from_s3_to_vespa(
    classifier_specs: list[ClassifierSpec],
    document_ids: list[str] | None = None,
    config: Config | None = None,
    batch_size: int = DEFAULT_DOCUMENTS_BATCH_SIZE,
    indexing_task_batch_size: int = DEFAULT_INDEXING_TASK_BATCH_SIZE,
) -> None:
    """
    Asynchronously de-index concepts from S3 into Vespa.

    This function retrieves inference results of concepts in documents
    from S3, "undoes" them in a Vespa instance, and deletes the
    appropriate objects from S3.

    The undoing is relative to the doing in the index pipeline. It's
    resilient to de-indexing per document failing, so that it can be
    retried.

    The name of each file in the specified S3 path is expected to
    represent the document's import ID.
    """
    logger = get_run_logger()

    if not config:
        logger.info("no config provided, creating one")

        config = await Config.create()
    else:
        logger.info("config provided")
    assert config.cache_bucket

    logger.info(f"running with config: {config}")

    logger.info(f"running with classifier specs: {classifier_specs}")

    s3_accessor = s3_paths_or_s3_prefixes(
        classifier_specs,
        document_ids,
        config.cache_bucket,
        config.document_source_prefix,
    )

    logger.info(f"s3_prefixes: {s3_accessor.prefixes}, s3_paths: {s3_accessor.paths}")

    await index_by_s3(
        partial_update_flow=run_partial_updates_of_concepts_for_document_passages__removal,
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
