from unittest.mock import AsyncMock, patch

import pytest

from flows.aggregate_inference_results import (
    DEFAULT_N_BATCHES,
    DEFAULT_N_DOCUMENTS_IN_BATCH,
    RunOutputIdentifier,
)
from flows.aggregate_inference_results import Config as AggregationConfig
from flows.boundary import DEFAULT_DOCUMENTS_BATCH_SIZE
from flows.full_pipeline import (
    full_pipeline,
    validate_aggregation_inference_configs,
)
from flows.inference import INFERENCE_BATCH_SIZE_DEFAULT
from flows.inference import (
    Config as InferenceConfig,
)
from flows.utils import DocumentImportId
from scripts.cloud import AwsEnv, ClassifierSpec


def test_validate_aggregation_inference_configs() -> None:
    """Test the validate_aggregation_inference_configs function."""

    config = validate_aggregation_inference_configs(
        inference_config=InferenceConfig(
            cache_bucket="test",
            bucket_region="test",
            document_target_prefix="test",
        ),
        aggregation_config=AggregationConfig(
            cache_bucket="test",
            bucket_region="test",
            document_source_prefix="test",
        ),
    )
    assert config is None


@pytest.mark.parametrize(
    "inference_config, aggregation_config, expected_error",
    [
        # Cache bucket mismatch
        (
            InferenceConfig(
                cache_bucket="bucket-does-not-exist",
                bucket_region="test",
                document_target_prefix="test",
            ),
            AggregationConfig(
                cache_bucket="test", bucket_region="test", document_source_prefix="test"
            ),
            "Cache bucket mismatch",
        ),
        # Prefix mismatch
        (
            InferenceConfig(
                cache_bucket="test",
                bucket_region="test",
                document_target_prefix="inference_target",
            ),
            AggregationConfig(
                cache_bucket="test",
                bucket_region="test",
                document_source_prefix="aggregation_source",
            ),
            "Inference target prefix does not match aggregation source prefix",
        ),
        # Region mismatch
        (
            InferenceConfig(
                cache_bucket="test",
                bucket_region="eu-west-1",
                document_target_prefix="test",
            ),
            AggregationConfig(
                cache_bucket="test",
                bucket_region="us-east-1",
                document_source_prefix="test",
            ),
            "Bucket region mismatch",
        ),
        # AWS env mismatch
        (
            InferenceConfig(
                cache_bucket="test",
                bucket_region="test",
                document_target_prefix="test",
                aws_env=AwsEnv.sandbox,
            ),
            AggregationConfig(
                cache_bucket="test",
                bucket_region="test",
                document_source_prefix="test",
                aws_env=AwsEnv.production,
            ),
            "AWS environment mismatch",
        ),
    ],
)
def test_validate_aggregation_inference_configs_raises_value_error(
    inference_config: InferenceConfig,
    aggregation_config: AggregationConfig,
    expected_error: str,
) -> None:
    """Test the validate_aggregation_inference_configs function raises a ValueError."""

    with pytest.raises(ValueError) as e:
        validate_aggregation_inference_configs(
            inference_config=inference_config,
            aggregation_config=aggregation_config,
        )
    assert expected_error in str(e.value)


@pytest.mark.asyncio
async def test_full_pipeline_no_config_provided(
    test_config: InferenceConfig,
    test_aggregate_config: AggregationConfig,
    aggregate_inference_results_document_stems,
    mock_run_output_identifier_str,
) -> None:
    """Test the flow when no aggregation or inference config is provided - should create default configs."""

    # Mock the sub-flows
    with (
        patch(
            "flows.full_pipeline.classifier_inference",
            new_callable=AsyncMock,
        ) as mock_inference,
        patch(
            "flows.full_pipeline.aggregate_inference_results",
            new_callable=AsyncMock,
        ) as mock_aggregate,
        patch(
            "flows.full_pipeline.run_indexing_from_aggregate_results",
            new_callable=AsyncMock,
        ) as mock_indexing,
        patch(
            "flows.full_pipeline.InferenceConfig.create",
            new_callable=AsyncMock,
        ) as mock_inference_create,
        patch(
            "flows.full_pipeline.AggregationConfig.create",
            new_callable=AsyncMock,
        ) as mock_aggregate_create,
    ):
        # Setup mocks
        mock_inference_create.return_value = test_config
        mock_aggregate_create.return_value = test_aggregate_config
        mock_inference.return_value = aggregate_inference_results_document_stems
        mock_aggregate.return_value = RunOutputIdentifier(
            mock_run_output_identifier_str
        )
        mock_indexing.return_value = None

        # Run the flow
        await full_pipeline()

        # Verify default configs were created
        mock_inference_create.assert_called_once()
        mock_aggregate_create.assert_called_once()

        # Verify sub-flows were called with correct parameters
        mock_inference.assert_called_once()
        call_args = mock_inference.call_args
        assert call_args.kwargs["config"] == test_config
        assert call_args.kwargs["classifier_specs"] is None
        assert call_args.kwargs["document_ids"] is None
        assert call_args.kwargs["use_new_and_updated"] is False
        assert call_args.kwargs["batch_size"] == INFERENCE_BATCH_SIZE_DEFAULT

        mock_aggregate.assert_called_once()
        call_args = mock_aggregate.call_args
        assert (
            call_args.kwargs["document_stems"]
            == aggregate_inference_results_document_stems
        )
        assert call_args.kwargs["config"] == test_aggregate_config
        assert call_args.kwargs["n_documents_in_batch"] == DEFAULT_N_DOCUMENTS_IN_BATCH
        assert call_args.kwargs["n_batches"] == DEFAULT_N_BATCHES

        mock_indexing.assert_called_once()
        call_args = mock_indexing.call_args
        assert call_args.kwargs["run_output_identifier"] == RunOutputIdentifier(
            mock_run_output_identifier_str
        )
        assert call_args.kwargs["config"] == test_aggregate_config
        assert call_args.kwargs["batch_size"] == DEFAULT_DOCUMENTS_BATCH_SIZE


@pytest.mark.asyncio
async def test_full_pipeline_with_full_config(
    test_config,
    test_aggregate_config,
    aggregate_inference_results_document_stems,
    mock_run_output_identifier_str,
):
    """Test the flow with complete config provided."""

    # Mock the sub-flows
    with (
        patch(
            "flows.full_pipeline.classifier_inference",
            new_callable=AsyncMock,
        ) as mock_inference,
        patch(
            "flows.full_pipeline.aggregate_inference_results",
            new_callable=AsyncMock,
        ) as mock_aggregate,
        patch(
            "flows.full_pipeline.run_indexing_from_aggregate_results",
            new_callable=AsyncMock,
        ) as mock_indexing,
    ):
        # Setup mocks
        mock_inference.return_value = aggregate_inference_results_document_stems
        mock_aggregate.return_value = RunOutputIdentifier(
            mock_run_output_identifier_str
        )
        mock_indexing.return_value = None

        # Run the flow
        await full_pipeline(
            inference_config=test_config,
            aggregation_config=test_aggregate_config,
            inference_classifier_specs=[ClassifierSpec(name="Q123", alias="v1")],
            inference_document_ids=[
                DocumentImportId("test.doc.1"),
                DocumentImportId("test.doc.2"),
            ],
            inference_use_new_and_updated=True,
            inference_batch_size=500,
            inference_classifier_concurrency_limit=5,
            aggregation_n_documents_in_batch=50,
            aggregation_n_batches=3,
            indexing_batch_size=200,
            indexer_concurrency_limit=2,
            indexer_document_passages_concurrency_limit=4,
            indexer_max_vespa_connections=8,
        )

        # Verify sub-flows were called with correct parameters
        mock_inference.assert_called_once_with(
            classifier_specs=[ClassifierSpec(name="Q123", alias="v1")],
            document_ids=[
                DocumentImportId("test.doc.1"),
                DocumentImportId("test.doc.2"),
            ],
            use_new_and_updated=True,
            config=test_config,
            batch_size=500,
            classifier_concurrency_limit=5,
        )

        mock_aggregate.assert_called_once_with(
            document_stems=aggregate_inference_results_document_stems,
            config=test_aggregate_config,
            n_documents_in_batch=50,
            n_batches=3,
        )

        mock_indexing.assert_called_once_with(
            run_output_identifier=RunOutputIdentifier(mock_run_output_identifier_str),
            config=test_aggregate_config,
            batch_size=200,
            indexer_concurrency_limit=2,
            indexer_document_passages_concurrency_limit=4,
            indexer_max_vespa_connections=8,
        )
