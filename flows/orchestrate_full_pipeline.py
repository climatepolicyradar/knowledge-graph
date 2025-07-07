from collections.abc import Sequence

from prefect import flow, get_run_logger
from pydantic import BaseModel, PositiveInt, field_validator, model_validator

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
    run_indexing_from_aggregate_results,
)
from flows.inference import CLASSIFIER_CONCURRENCY_LIMIT, classifier_inference
from flows.inference import Config as InferenceConfig
from flows.utils import DocumentImportId
from scripts.cloud import ClassifierSpec


class OrchestrateFullPipelineConfig(BaseModel):
    """
    Configuration for the full knowledge graph pipeline orchestration.

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

    # Inference Params

    inference_classifier_specs: Sequence[ClassifierSpec] | None = None
    inference_document_ids: Sequence[DocumentImportId] | None = None
    inference_use_new_and_updated: bool = False
    inference_config: InferenceConfig | None = None
    inference_batch_size: int = 1000
    inference_classifier_concurrency_limit: PositiveInt = CLASSIFIER_CONCURRENCY_LIMIT

    # Aggregation Params

    aggregation_config: AggregationConfig | None = None
    aggregation_n_documents_in_batch: PositiveInt = (
        AGGREGATION_DEFAULT_N_DOCUMENTS_IN_BATCH
    )
    aggregation_n_batches: PositiveInt = 5

    # Indexing Params

    indexing_batch_size: int = INDEXING_DEFAULT_DOCUMENTS_BATCH_SIZE
    indexer_concurrency_limit: PositiveInt = DEFAULT_INDEXER_CONCURRENCY_LIMIT
    indexer_max_vespa_connections: PositiveInt = (
        DEFAULT_VESPA_MAX_CONNECTIONS_AGG_INDEXER
    )

    @field_validator("inference_config", mode="before")
    @classmethod
    async def set_inference_config_if_none(
        cls, v: InferenceConfig | None
    ) -> InferenceConfig:
        """Set the inference config if it is not provided."""
        if v is None:
            return await InferenceConfig.create()
        return v

    @field_validator("aggregation_config", mode="before")
    @classmethod
    async def set_aggregation_config_if_none(
        cls, v: AggregationConfig | None
    ) -> AggregationConfig:
        """Set the aggregation config if it is not provided."""
        if v is None:
            return await AggregationConfig.create()
        return v

    @model_validator(mode="after")
    def check_sub_config_fields_match(self) -> "OrchestrateFullPipelineConfig":
        """
        Check that the sub config fields match.

        This is a post instantiation check to ensure that the sub config fields match.
        This is as they have fields that should match and we want to catch this before
        we run the flows.
        """

        if self.aggregation_config is None or self.inference_config is None:
            raise ValueError("Sub config fields are not set.")

        if (
            self.aggregation_config.cache_bucket_str
            != self.inference_config.cache_bucket
        ):
            raise ValueError("Cache bucket mismatch")
        if (
            self.aggregation_config.document_source_prefix
            != self.inference_config.document_source_prefix
        ):
            raise ValueError("Document source prefix mismatch")
        if (
            self.aggregation_config.document_source_prefix
            != self.inference_config.document_target_prefix
        ):
            raise ValueError("Document target prefix mismatch")
        if self.aggregation_config.bucket_region != self.inference_config.bucket_region:
            raise ValueError("Bucket region mismatch")
        if self.aggregation_config.aws_env != self.inference_config.aws_env:
            raise ValueError("AWS environment mismatch")
        return self


@flow(log_prints=True)
async def orchestrate_full_pipeline(
    config: OrchestrateFullPipelineConfig | None = None,
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
    """

    logger = get_run_logger()

    if not config:
        logger.info("No config provided, creating one...")
        config = OrchestrateFullPipelineConfig()

    logger.info(f"Orchestrating full pipeline with config: {config}")

    try:
        logger.info("Starting inference...")
        document_stems = await classifier_inference(
            classifier_specs=config.inference_classifier_specs,
            document_ids=config.inference_document_ids,
            use_new_and_updated=config.inference_use_new_and_updated,
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
            n_documents_in_batch=config.aggregation_n_documents_in_batch,
            n_batches=config.aggregation_n_batches,
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
