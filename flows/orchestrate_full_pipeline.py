from collections.abc import Sequence

from prefect import flow, get_run_logger
from pydantic import BaseModel, PositiveInt

from flows.aggregate_inference_results import (
    DEFAULT_N_DOCUMENTS_IN_BATCH,
    RunOutputIdentifier,
    aggregate_inference_results,
)
from flows.aggregate_inference_results import Config as AggregationConfig
from flows.boundary import DEFAULT_DOCUMENTS_BATCH_SIZE
from flows.index_from_aggregate_results import (
    DEFAULT_INDEXER_CONCURRENCY_LIMIT,
    DEFAULT_VESPA_MAX_CONNECTIONS_AGG_INDEXER,
    run_indexing_from_aggregate_results,
)
from flows.inference import CLASSIFIER_CONCURRENCY_LIMIT, classifier_inference
from flows.inference import Config as InferenceConfig
from flows.utils import DocumentImportId
from scripts.cloud import ClassifierSpec

# TODO: Update imports to import constants with clearer naming.


class OrchestrateFullPipelineConfig(BaseModel):
    # TODO: Add documentation on params.
    """Configuration for the full pipeline orchestration."""

    # Inference Params
    # ---------------

    classifier_specs: Sequence[ClassifierSpec] | None = None
    document_ids: Sequence[DocumentImportId] | None = None
    use_new_and_updated: bool = False
    inference_config: InferenceConfig | None = None
    inference_batch_size: int = 1000
    inference_classifier_concurrency_limit: PositiveInt = CLASSIFIER_CONCURRENCY_LIMIT

    # Aggregation Params
    # ------------------

    # Duplicate
    # document_ids: None | list[DocumentImportId] = None,
    aggregation_config: AggregationConfig | None = None
    n_documents_in_batch: PositiveInt = DEFAULT_N_DOCUMENTS_IN_BATCH
    n_batches: PositiveInt = 5

    # Indexing Params
    # ---------------

    # We get this from the aggregation run but it can also be set in config...
    # In this instance I'd say we don't want to expose it, if someone wants to run
    # indexing on a different run they can just use the indexing flow.
    # run_output_identifier: RunOutputIdentifier

    # Returned from inference.
    # document_stems: list[DocumentStem] | None = None

    # Duplicate of aggregation_config.
    # indexing_config: AggregationConfig | None = None

    indexing_batch_size: int = DEFAULT_DOCUMENTS_BATCH_SIZE
    indexer_concurrency_limit: PositiveInt = DEFAULT_INDEXER_CONCURRENCY_LIMIT

    # Not used in the flow
    # indexer_document_passages_concurrency_limit: PositiveInt = (
    #   INDEXER_DOCUMENT_PASSAGES_CONCURRENCY_LIMIT
    # )

    indexer_max_vespa_connections: PositiveInt = (
        DEFAULT_VESPA_MAX_CONNECTIONS_AGG_INDEXER
    )


@flow(log_prints=True)
async def orchestrate_full_pipeline(
    config: OrchestrateFullPipelineConfig | None = None,
):
    # TODO: Add documentation on params.
    """
    Orchestrate the full KG pipeline.

    This flow orchestrates the full KG pipeline, including:
    1. Inference
    2. Aggregation
    3. Indexing
    """

    logger = get_run_logger()

    if not config:
        logger.info("No config provided, creating one...")
        config = OrchestrateFullPipelineConfig()

    # TODO: Any checking of the two config objects for inference and aggregation?
    # For example, do we want to assert that the buckets and regions are correct?
    # Variables that should match:
    # cache_bucket, bucket_region, aws_env
    # This could be a post instantiation check on the OrchestrateFullPipelineConfig
    # object.

    logger.info(f"Orchestrating full pipeline with config: {config}")

    try:
        logger.info("Starting inference...")
        document_stems = await classifier_inference(
            classifier_specs=config.classifier_specs,
            document_ids=config.document_ids,
            use_new_and_updated=config.use_new_and_updated,
            config=config.inference_config,
            batch_size=config.inference_batch_size,
            classifier_concurrency_limit=config.inference_classifier_concurrency_limit,
        )
        logger.info("Inference complete.")
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise

    try:
        logger.info("Starting aggregation...")
        run_output_identifier: RunOutputIdentifier = await aggregate_inference_results(
            document_stems=document_stems,
            config=config.aggregation_config,
            n_documents_in_batch=config.n_documents_in_batch,
            n_batches=config.n_batches,
        )
        logger.info(
            f"Aggregation complete. Run output identifier: {run_output_identifier}"
        )
    except Exception as e:
        logger.error(f"Aggregation failed: {e}")
        raise

    try:
        logger.info("Starting indexing...")
        await run_indexing_from_aggregate_results(
            run_output_identifier=run_output_identifier,
            # If we run aggregation and store results under this unique prefix then we
            #   should just index everything from within this sub directory as this
            #   should match the document_stems?
            # document_stems=document_stems,
            config=config.aggregation_config,
            batch_size=config.indexing_batch_size,
            indexer_concurrency_limit=config.indexer_concurrency_limit,
            indexer_max_vespa_connections=config.indexer_max_vespa_connections,
        )
        logger.info("Indexing complete.")
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise


# TODO: Add run artifact.
#   What needs to be in this one as we have the child artifacts?
# TODO: Add tests.
#   What to mock and what to let run?
# TODO: Add docs.
