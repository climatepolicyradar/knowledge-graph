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
    INDEXER_DOCUMENT_PASSAGES_CONCURRENCY_LIMIT,
    run_indexing_from_aggregate_results,
)
from flows.inference import CLASSIFIER_CONCURRENCY_LIMIT, classifier_inference
from flows.inference import Config as InferenceConfig
from flows.utils import DocumentImportId, DocumentStem
from scripts.cloud import ClassifierSpec

# TODO: Update imports to import constants with clearer naming.
# TODO: Check that the imports are coming from the correct module.


class OrchestrateFullPipelineConfig(BaseModel):
    """Configuration for the full pipeline orchestration."""

    # Inference Params

    classifier_specs: Sequence[ClassifierSpec] | None = None
    document_ids: Sequence[DocumentImportId] | None = None
    use_new_and_updated: bool = False
    inference_config: InferenceConfig | None = None
    inference_batch_size: int = 1000
    inference_classifier_concurrency_limit: PositiveInt = CLASSIFIER_CONCURRENCY_LIMIT

    # Aggregation Params

    # Duplicate
    # document_ids: None | list[DocumentImportId] = None,
    aggregation_config: AggregationConfig | None = None
    n_documents_in_batch: PositiveInt = DEFAULT_N_DOCUMENTS_IN_BATCH
    n_batches: PositiveInt = 5

    # Indexing Params

    # TODO: We get this from the aggregation run but it can also be set in config...
    # In this instance I'd say we don't want to expose it, if someone wants to run
    # indexing on a different run they can just use the indexing flow.
    # run_output_identifier: RunOutputIdentifier
    # TODO: How to pass these in as we take import ids elsewhere?
    document_stems: list[DocumentStem] | None = None
    indexing_config: AggregationConfig | None = None
    indexing_batch_size: int = DEFAULT_DOCUMENTS_BATCH_SIZE
    indexer_concurrency_limit: PositiveInt = DEFAULT_INDEXER_CONCURRENCY_LIMIT
    indexer_document_passages_concurrency_limit: PositiveInt = (
        INDEXER_DOCUMENT_PASSAGES_CONCURRENCY_LIMIT
    )
    indexer_max_vespa_connections: PositiveInt = (
        DEFAULT_VESPA_MAX_CONNECTIONS_AGG_INDEXER
    )


@flow(log_prints=True)
async def orchestrate_full_pipeline(
    config: OrchestrateFullPipelineConfig,
):
    """
    Orchestrate the full KG pipeline.

    This flow orchestrates the full KG pipeline, including:
    1. Inference
    2. Aggregation
    3. Indexing
    """

    logger = get_run_logger()

    # TODO: Any checking of the two config objects for inference and aggregation?
    # For example, do we want to assert that the buckets and regions are correct?
    # Variables that should match:
    # cache_bucket, bucket_region, aws_env
    # This could be a post instantiation check on the OrchestrateFullPipelineConfig
    # object.

    logger.info(f"Orchestrating full pipeline with config: {config}")

    # TODO: If document_ids are not set we don't want to comb over s3 twice for all
    # docs so could we define here and pass in? Also alot of hidden logic in the steps
    # as to whether inference has run for example or not.

    # Inference
    #  - Identifies file stems to run on from a combination of document_ids,
    #       new_and_updated and all the bucket files.
    # - document_ids as an input are correct here.
    #  - Identifies the classifiers from the local specs.
    #  - Also, some hidden logic to filter Sabin docs and not run on the latest classifier.

    # Aggregation
    # - Simply runs on the document ids provided or all the file stems in the
    #      bucket/${classifier_spec.name}/${classifier_spec.alias}.
    # It really should run on file stems but needs refactoring.
    # - Get's the classifiers to run on from the local specs (no latest filtering)

    # Indexing
    # - Runs on the document stems provided or all the file stems in the unique run prefix.

    # Wondering whether we can do the following:
    # Are there any scaling concerns if we were passing around 30k file stems?
    #
    # 1. Update inference to return the classifiers and document stems that inference
    #    was run on.
    # inference_run_file_stems, inference_run_classifiers = await classifier_inference(...)
    # 2. Pass in these results into aggregation and indexing.

    try:
        logger.info("Starting inference...")
        await classifier_inference(
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
            document_ids=list(config.document_ids) if config.document_ids else None,
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
            document_stems=config.document_stems,
            config=config.indexing_config,
            batch_size=config.indexing_batch_size,
            indexer_concurrency_limit=config.indexer_concurrency_limit,
            indexer_document_passages_concurrency_limit=config.indexer_document_passages_concurrency_limit,
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
