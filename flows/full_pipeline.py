from collections.abc import Sequence

from prefect import State, flow, get_run_logger
from pydantic import PositiveInt

from flows.aggregate import (
    DEFAULT_N_DOCUMENTS_IN_BATCH as AGGREGATION_DEFAULT_N_DOCUMENTS_IN_BATCH,
)
from flows.aggregate import RunOutputIdentifier, aggregate
from flows.boundary import (
    DEFAULT_DOCUMENTS_BATCH_SIZE as INDEXING_DEFAULT_DOCUMENTS_BATCH_SIZE,
)
from flows.config import Config
from flows.index import (
    DEFAULT_INDEXER_CONCURRENCY_LIMIT,
    DEFAULT_VESPA_MAX_CONNECTIONS_AGG_INDEXER,
    INDEXER_DOCUMENT_PASSAGES_CONCURRENCY_LIMIT,
    index,
)
from flows.inference import (
    CLASSIFIER_CONCURRENCY_LIMIT,
    INFERENCE_BATCH_SIZE_DEFAULT,
    InferenceResult,
    inference,
)
from flows.utils import DocumentImportId, Fault
from src.cloud import ClassifierSpec


# pyright: reportCallIssue=false, reportGeneralTypeIssues=false
@flow(log_prints=True)
async def full_pipeline(
    classifier_specs: Sequence[ClassifierSpec] | None = None,
    document_ids: Sequence[DocumentImportId] | None = None,
    inference_use_new_and_updated: bool = False,
    inference_batch_size: int = INFERENCE_BATCH_SIZE_DEFAULT,
    inference_classifier_concurrency_limit: PositiveInt = CLASSIFIER_CONCURRENCY_LIMIT,
    config: Config | None = None,
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
        classifier_specs: Classifier specifications to use for inference.
        document_ids: Specific document IDs to process. If None, processes all.
        config: Configuration for the inference, aggregation and index flows. If None, creates default.
        inference_use_new_and_updated: Whether to process only new/updated documents.
        inference_batch_size: Number of documents to process in each batch.
        inference_classifier_concurrency_limit: Maximum concurrent classifiers.
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

    if not config:
        logger.info("No pipeline config provided, creating default...")
        config = await Config.create()

    logger.info(f"Running the full pipeline with the config: {config}, ")

    inference_run: State = await inference(
        classifier_specs=classifier_specs,
        document_ids=document_ids,
        use_new_and_updated=inference_use_new_and_updated,
        config=config,
        batch_size=inference_batch_size,
        classifier_concurrency_limit=inference_classifier_concurrency_limit,
        return_state=True,
    )

    inference_result_raw: (
        InferenceResult | Fault | Exception
    ) = await inference_run.result(raise_on_failure=False)

    match inference_result_raw:
        case Exception() if not isinstance(inference_result_raw, Fault):
            logger.error("Inference failed.")
            raise inference_result_raw
        case Fault():
            assert isinstance(inference_result_raw.data, InferenceResult), (
                "Expected data field of the Fault to contain an InferenceResult object,"
                + f"got type: {type(inference_result_raw.data)}"
            )
            inference_result: InferenceResult = inference_result_raw.data
        case InferenceResult():
            inference_result: InferenceResult = inference_result_raw
        case _:
            raise ValueError(
                f"Unexpected inference result type: {type(inference_result_raw)}"
            )

    success_ratio: str = (
        f"{len(inference_result.fully_successfully_classified_document_stems)}/"
        + f"{len(inference_result.document_stems)}"
    )
    logger.info(
        f"Inference complete. Successfully classified {success_ratio} documents."
    )

    if len(inference_result.fully_successfully_classified_document_stems) == 0:
        raise ValueError(
            "Inference successfully ran on 0 documents, skipping aggregation and indexing."
        )

    aggregation_run: State = await aggregate(
        document_stems=list(
            inference_result.fully_successfully_classified_document_stems
        ),
        config=config,
        n_documents_in_batch=aggregation_n_documents_in_batch,
        n_batches=aggregation_n_batches,
        return_state=True,
    )
    aggregation_result: RunOutputIdentifier | Exception = await aggregation_run.result(
        raise_on_failure=False
    )

    if isinstance(aggregation_result, Exception):
        logger.error("Aggregation failed.")
        raise aggregation_result
    logger.info(f"Aggregation complete. Run output identifier: {aggregation_result}")

    indexing_run: State = await index(
        run_output_identifier=aggregation_result,
        config=config,
        batch_size=indexing_batch_size,
        indexer_concurrency_limit=indexer_concurrency_limit,
        indexer_document_passages_concurrency_limit=indexer_document_passages_concurrency_limit,
        indexer_max_vespa_connections=indexer_max_vespa_connections,
        return_state=True,
    )
    indexing_result: None | Exception = await indexing_run.result(
        raise_on_failure=False
    )

    if isinstance(indexing_result, Exception):
        logger.error("Indexing failed.")
        raise indexing_result

    logger.info("Full pipeline run completed!")
