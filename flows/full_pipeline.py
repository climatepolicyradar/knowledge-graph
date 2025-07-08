from collections.abc import Sequence

from prefect import flow, get_run_logger
from pydantic import PositiveInt

from flows.aggregate_inference_results import (
    DEFAULT_N_DOCUMENTS_IN_BATCH as AGGREGATION_DEFAULT_N_DOCUMENTS_IN_BATCH,
)
from flows.aggregate_inference_results import (
    Config as AggregationConfig,
)
from flows.aggregate_inference_results import (
    RunOutputIdentifier,
    aggregate_inference_results,
)
from flows.boundary import (
    DEFAULT_DOCUMENTS_BATCH_SIZE as INDEXING_DEFAULT_DOCUMENTS_BATCH_SIZE,
)
from flows.index_from_aggregate_results import (
    DEFAULT_INDEXER_CONCURRENCY_LIMIT,
    DEFAULT_VESPA_MAX_CONNECTIONS_AGG_INDEXER,
    INDEXER_DOCUMENT_PASSAGES_CONCURRENCY_LIMIT,
    run_indexing_from_aggregate_results,
)
from flows.inference import CLASSIFIER_CONCURRENCY_LIMIT, classifier_inference
from flows.inference import Config as InferenceConfig
from flows.utils import DocumentImportId
from scripts.cloud import ClassifierSpec


def check_sub_config_fields_match(
    aggregation_config: AggregationConfig,
    inference_config: InferenceConfig,
) -> None:
    """
    Check that the sub config fields match.

    This is as they have fields that should match and we want to catch this before we
    run the flows.

    Raises:
        ValueError: If the config fields don't match.
    """

    if aggregation_config.cache_bucket_str != inference_config.cache_bucket:
        raise ValueError(
            f"Cache bucket mismatch: {aggregation_config.cache_bucket_str} != {inference_config.cache_bucket}"
        )
    if (
        aggregation_config.document_source_prefix
        != inference_config.document_target_prefix
    ):
        raise ValueError(
            "Inference target prefix does not match aggregation source prefix"
        )
    if aggregation_config.bucket_region != inference_config.bucket_region:
        raise ValueError("Bucket region mismatch")
    if aggregation_config.aws_env != inference_config.aws_env:
        raise ValueError("AWS environment mismatch")


@flow(log_prints=True)
async def full_pipeline(
    inference_config: InferenceConfig | None = None,
    inference_classifier_specs: Sequence[ClassifierSpec] | None = None,
    inference_document_ids: Sequence[DocumentImportId] | None = None,
    inference_use_new_and_updated: bool = False,
    inference_batch_size: int = 1000,
    inference_classifier_concurrency_limit: PositiveInt = CLASSIFIER_CONCURRENCY_LIMIT,
    aggregation_config: AggregationConfig | None = None,
    aggregation_n_documents_in_batch: PositiveInt = AGGREGATION_DEFAULT_N_DOCUMENTS_IN_BATCH,
    aggregation_n_batches: PositiveInt = 5,
    indexing_batch_size: int = INDEXING_DEFAULT_DOCUMENTS_BATCH_SIZE,
    indexer_concurrency_limit: PositiveInt = DEFAULT_INDEXER_CONCURRENCY_LIMIT,
    indexer_document_passages_concurrency_limit: PositiveInt = INDEXER_DOCUMENT_PASSAGES_CONCURRENCY_LIMIT,
    indexer_max_vespa_connections: PositiveInt = DEFAULT_VESPA_MAX_CONNECTIONS_AGG_INDEXER,
):
    """
    Orchestrate the full KG pipeline.

    This flow orchestrates the full KG pipeline, including:
    1. Inference
    2. Aggregation
    3. Indexing

    We run inference either on all documents or based upon the configured document ids
    and use_new_and_updated flag. This step identifies the document stems to process.

    We return the document stems that inference ran on and use this as a parameter
    within the aggregation step.

    Aggregation then returns a unique run identifier which we use as a parameter within
    the indexing step. Thus, we index all documents from within the aggregation run
    directory.

    Indexing includes both passage level concept indexing as well as
    family level indexing of concept counts.

    We expose the listed parameters such that we can configure the
    sub pipelines.

    - If config is not provided for the inference and aggregation sub pipelines then we
      create these with default values. We could have let the sub pipelines create the
      config objects but it's preferred creating first and validating before starting the
      run so we can check that the configs are correct and so we don't for
      example run inference for OOM hrs and then fail at config instantiation and
      validation.

    - The aggregation and indexing configs are the same so we only expose the aggregation
      config.
    """

    logger = get_run_logger()

    if not aggregation_config:
        logger.info("No aggregation config provided, creating default...")
        aggregation_config = await AggregationConfig.create()

    if not inference_config:
        logger.info("No inference config provided, creating default...")
        inference_config = await InferenceConfig.create()

    check_sub_config_fields_match(aggregation_config, inference_config)

    logger.info(
        f"Orchestrating full pipeline with aggregation config: {aggregation_config}, "
        + f"inference config: {inference_config}"
    )

    try:
        logger.info("Starting inference...")
        document_stems = await classifier_inference(
            classifier_specs=inference_classifier_specs,
            document_ids=inference_document_ids,
            use_new_and_updated=inference_use_new_and_updated,
            config=inference_config,
            batch_size=inference_batch_size,
            classifier_concurrency_limit=inference_classifier_concurrency_limit,
        )
        logger.info("Inference complete.")
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise

    try:
        logger.info("Starting aggregation...")
        run_output_identifier: RunOutputIdentifier = await aggregate_inference_results(
            document_stems=document_stems,
            config=aggregation_config,
            n_documents_in_batch=aggregation_n_documents_in_batch,
            n_batches=aggregation_n_batches,
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
            config=aggregation_config,
            batch_size=indexing_batch_size,
            indexer_concurrency_limit=indexer_concurrency_limit,
            indexer_document_passages_concurrency_limit=indexer_document_passages_concurrency_limit,
            indexer_max_vespa_connections=indexer_max_vespa_connections,
        )
        logger.info("Indexing complete.")
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise


# TODO: Add run artifact.
#   What needs to be in this one as we have the child artifacts?
#       document_stem count, run_output_identifier etc. these are already captured.
# TODO: Add tests.
#   What to mock and what to let run?
# TODO: Add docs.
