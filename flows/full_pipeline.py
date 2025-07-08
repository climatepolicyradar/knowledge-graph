from collections.abc import Sequence

from prefect import State, flow, get_run_logger
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
from flows.inference import (
    CLASSIFIER_CONCURRENCY_LIMIT,
    INFERENCE_BATCH_SIZE_DEFAULT,
    classifier_inference,
)
from flows.inference import Config as InferenceConfig
from flows.utils import DocumentImportId, DocumentStem
from scripts.cloud import ClassifierSpec


def validate_aggregation_inference_configs(
    aggregation_config: AggregationConfig,
    inference_config: InferenceConfig,
) -> None:
    """
    Check that the aggregation and inference config fields match.

    This is as they have fields that should match and we want to catch this before we
    run the flows.

    Raises:
        ValueError: If the config fields don't match.
    """

    if aggregation_config.cache_bucket_str != inference_config.cache_bucket:
        raise ValueError(
            f"Cache bucket mismatch: {aggregation_config.cache_bucket_str} != "
            + f"{inference_config.cache_bucket}"
        )
    if (
        aggregation_config.document_source_prefix
        != inference_config.document_target_prefix
    ):
        raise ValueError(
            "Inference target prefix does not match aggregation source prefix: "
            + f"{inference_config.document_target_prefix} != "
            + f"{aggregation_config.document_source_prefix}"
        )
    if aggregation_config.bucket_region != inference_config.bucket_region:
        raise ValueError(
            f"Bucket region mismatch: {aggregation_config.bucket_region} != "
            + f"{inference_config.bucket_region}"
        )
    if aggregation_config.aws_env != inference_config.aws_env:
        raise ValueError(
            f"AWS environment mismatch: {aggregation_config.aws_env} != "
            + f"{inference_config.aws_env}"
        )


@flow(log_prints=True)
async def full_pipeline(
    inference_config: InferenceConfig | None = None,
    inference_classifier_specs: Sequence[ClassifierSpec] | None = None,
    inference_document_ids: Sequence[DocumentImportId] | None = None,
    inference_use_new_and_updated: bool = False,
    inference_batch_size: int = INFERENCE_BATCH_SIZE_DEFAULT,
    inference_classifier_concurrency_limit: PositiveInt = CLASSIFIER_CONCURRENCY_LIMIT,
    aggregation_config: AggregationConfig | None = None,
    aggregation_n_documents_in_batch: PositiveInt = AGGREGATION_DEFAULT_N_DOCUMENTS_IN_BATCH,
    aggregation_n_batches: PositiveInt = 5,
    indexing_batch_size: int = INDEXING_DEFAULT_DOCUMENTS_BATCH_SIZE,
    indexer_concurrency_limit: PositiveInt = DEFAULT_INDEXER_CONCURRENCY_LIMIT,
    indexer_document_passages_concurrency_limit: PositiveInt = INDEXER_DOCUMENT_PASSAGES_CONCURRENCY_LIMIT,
    indexer_max_vespa_connections: PositiveInt = DEFAULT_VESPA_MAX_CONNECTIONS_AGG_INDEXER,
) -> None:
    """
    Full KG pipeline.

    This flow orchestrates the full KG pipeline, including:
    1. Inference
    2. Aggregation
    3. Indexing

    Args:
        inference_config: Configuration for the inference stage. If None, creates default.
        inference_classifier_specs: Classifier specifications to use for inference.
        inference_document_ids: Specific document IDs to process. If None, processes all.
        inference_use_new_and_updated: Whether to process only new/updated documents.
        inference_batch_size: Number of documents to process in each batch.
        inference_classifier_concurrency_limit: Maximum concurrent classifiers.
        aggregation_config: Configuration for the aggregation stage. If None, creates default.
        aggregation_n_documents_in_batch: Number of documents per aggregation batch.
        aggregation_n_batches: Number of aggregation batches to run.
        indexing_batch_size: Number of documents to index in each batch.
        indexer_concurrency_limit: Maximum concurrent indexers.
        indexer_document_passages_concurrency_limit: Max concurrent passage indexers.
        indexer_max_vespa_connections: Maximum Vespa connections for indexing.

    Returns:
        None

    Raises:
        ValueError: If inference and aggregation configs are incompatible.
    """

    logger = get_run_logger()

    if not aggregation_config:
        logger.info("No aggregation config provided, creating default...")
        aggregation_config = await AggregationConfig.create()

    if not inference_config:
        logger.info("No inference config provided, creating default...")
        inference_config = await InferenceConfig.create()

    validate_aggregation_inference_configs(aggregation_config, inference_config)

    logger.info(
        f"Running the full pipeline with aggregation config: {aggregation_config}, "
        + f"inference config: {inference_config}"
    )

    classifier_inference_run: State = await classifier_inference(
        classifier_specs=inference_classifier_specs,
        document_ids=inference_document_ids,
        use_new_and_updated=inference_use_new_and_updated,
        config=inference_config,
        batch_size=inference_batch_size,
        classifier_concurrency_limit=inference_classifier_concurrency_limit,
        return_state=True,
    )  # pyright: ignore[reportGeneralTypeIssues]
    classifier_inference_result: Sequence[DocumentStem] | Exception = (
        await classifier_inference_run.result(raise_on_failure=False)  # pyright: ignore[reportGeneralTypeIssues]
    )

    if isinstance(classifier_inference_result, Exception):
        logger.error("Inference failed.")
        raise classifier_inference_result
    logger.info("Inference complete.")

    # TODO: Update run_classifier_inference_on_batch_of_documents to return a result with
    # the document stems that were successfully processed. Then update classifier_inference
    # to return all of the successful document stems from all the inference batches so that
    # we can use this as the input to the aggregation step.
    # Currently using filtered_file_stems as the input to the aggregation step, some of
    # these may not have inference results due to failures and thus we expect further
    # failures in the aggregation step.

    logger.info("Starting aggregation...")
    aggregation_run: State = await aggregate_inference_results(
        document_stems=classifier_inference_result,
        config=aggregation_config,
        n_documents_in_batch=aggregation_n_documents_in_batch,
        n_batches=aggregation_n_batches,
        return_state=True,
    )  # pyright: ignore[reportGeneralTypeIssues]
    aggregation_result: RunOutputIdentifier | Exception = (
        await aggregation_run.result(raise_on_failure=False)  # pyright: ignore[reportGeneralTypeIssues]
    )

    if isinstance(aggregation_result, Exception):
        logger.error("Aggregation failed.")
        raise aggregation_result
    logger.info("Aggregation complete.")

    logger.info("Starting indexing...")
    indexing_run: State = (
        await run_indexing_from_aggregate_results(
            run_output_identifier=aggregation_result,
            config=aggregation_config,
            batch_size=indexing_batch_size,
            indexer_concurrency_limit=indexer_concurrency_limit,
            indexer_document_passages_concurrency_limit=indexer_document_passages_concurrency_limit,
            indexer_max_vespa_connections=indexer_max_vespa_connections,
            return_state=True,
        )  # pyright: ignore[reportGeneralTypeIssues]
    )
    indexing_result: Exception = (
        await indexing_run.result(raise_on_failure=False)  # pyright: ignore[reportGeneralTypeIssues]
    )

    if isinstance(indexing_result, Exception):
        logger.error("Indexing failed.")
        raise indexing_result
    logger.info("Indexing complete.")
