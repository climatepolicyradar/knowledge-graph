import json
from collections.abc import Sequence

from prefect import State, flow
from prefect.artifacts import create_markdown_artifact
from pydantic import PositiveInt

from flows.aggregate import (
    DEFAULT_N_DOCUMENTS_IN_BATCH as AGGREGATION_DEFAULT_N_DOCUMENTS_IN_BATCH,
)
from flows.aggregate import (
    RunOutputIdentifier,
    aggregate,
)
from flows.boundary import (
    DEFAULT_DOCUMENTS_BATCH_SIZE as INDEXING_DEFAULT_DOCUMENTS_BATCH_SIZE,
)
from flows.classifier_specs.spec_interface import ClassifierSpec
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
    build_inference_result_s3_uri,
    inference,
)
from flows.utils import DocumentImportId, DocumentStem, Fault, get_logger
from knowledge_graph.cloud import get_async_session


async def create_full_pipeline_summary_artifact(
    config: Config,
    successful_document_stems: set[DocumentStem],
) -> None:
    """Create an artifact with summary information about the full pipeline successful run."""

    # Format the overview information as a string for the description
    full_pipeline_report = f"""# Full Pipeline Summary

## Overview
- **Environment**: {config.aws_env.value}
- **Inference: successful documents**: {len(successful_document_stems)}
"""

    await create_markdown_artifact(  # pyright: ignore[reportGeneralTypeIssues]
        key=f"full-pipeline-results-summary-{config.aws_env.value}",
        description="Summary of the full pipeline successful run.",
        markdown=full_pipeline_report,
    )


# pyright: reportCallIssue=false, reportGeneralTypeIssues=false
@flow()
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

    logger = get_logger()

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
        return_pointer=True,
        return_state=True,
    )

    inference_result_raw: (
        RunOutputIdentifier | Fault | Exception
    ) = await inference_run.result(raise_on_failure=False)

    match inference_result_raw:
        case Exception() if not isinstance(inference_result_raw, Fault):
            logger.error("Inference failed.")
            raise inference_result_raw
        case Fault():
            if not isinstance(inference_result_raw.data, dict):
                raise ValueError(
                    "Expected data field of the Fault to contain a dict with successful_document_stems and run_output_identifier,"
                    + f"got type: {type(inference_result_raw.data)}"
                )
            successful_document_stems: set[DocumentStem] = inference_result_raw.data[
                "successful_document_stems"
            ]
            run_output_identifier: RunOutputIdentifier = inference_result_raw.data[
                "run_output_identifier"
            ]
            logger.info(
                f"Inference complete with partial failures. Successfully classified {len(successful_document_stems)} documents."
            )
        case str():
            run_output_identifier: RunOutputIdentifier = inference_result_raw
            s3_uri = build_inference_result_s3_uri(config, run_output_identifier)
            session = get_async_session(
                region_name=config.bucket_region,
                aws_env=config.aws_env,
            )
            async with session.client("s3") as s3_client:
                response = await s3_client.get_object(
                    Bucket=s3_uri.bucket, Key=s3_uri.key
                )
                body = await response["Body"].read()
                result_data = json.loads(body.decode("utf-8"))
                successful_document_stems: set[DocumentStem] = set(
                    result_data["successful_document_stems"]
                )
            logger.info(
                f"Inference complete. Successfully classified {len(successful_document_stems)} documents."
            )
        case _:
            raise ValueError(f"unexpected result {type(inference_result_raw)}")

    aggregation_run: State = await aggregate(
        run_output_identifier=run_output_identifier,
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

    await create_full_pipeline_summary_artifact(
        config=config,
        successful_document_stems=successful_document_stems,
    )

    logger.info("Full pipeline run completed!")
