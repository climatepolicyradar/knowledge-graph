from unittest.mock import AsyncMock, patch

import pytest

from flows.aggregate_inference_results import Config as AggregationConfig
from flows.aggregate_inference_results import RunOutputIdentifier
from flows.inference import Config as InferenceConfig
from flows.orchestrate_full_pipeline import (
    check_sub_config_fields_match,
    orchestrate_full_pipeline,
)
from flows.utils import DocumentImportId, DocumentStem
from scripts.cloud import AwsEnv, ClassifierSpec

# Test the check_sub_config_fields_match function


def test_check_sub_config_fields_match() -> None:
    """Test the check_sub_config_fields_match function."""

    config = check_sub_config_fields_match(
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
def test_check_sub_config_fields_match_raises_value_error(
    inference_config: InferenceConfig,
    aggregation_config: AggregationConfig,
    expected_error: str,
) -> None:
    """Test the check_sub_config_fields_match function raises a ValueError."""

    with pytest.raises(ValueError) as e:
        check_sub_config_fields_match(
            inference_config=inference_config,
            aggregation_config=aggregation_config,
        )
    assert expected_error in str(e.value)


# Test the orchestrate_full_pipeline flow with mocked sub-flows


@pytest.mark.asyncio
async def test_orchestrate_full_pipeline_no_config_provided(
    test_config: InferenceConfig, test_aggregate_config: AggregationConfig
) -> None:
    """Test the flow when no aggregation or inference config is provided - should create default configs."""

    # Mock the sub-flows
    mock_document_stems = [DocumentStem("test.doc.1"), DocumentStem("test.doc.2")]
    mock_run_identifier = RunOutputIdentifier("test-run-123")

    with (
        patch(
            "flows.orchestrate_full_pipeline.classifier_inference",
            new_callable=AsyncMock,
        ) as mock_inference,
        patch(
            "flows.orchestrate_full_pipeline.aggregate_inference_results",
            new_callable=AsyncMock,
        ) as mock_aggregate,
        patch(
            "flows.orchestrate_full_pipeline.run_indexing_from_aggregate_results",
            new_callable=AsyncMock,
        ) as mock_indexing,
        patch(
            "flows.orchestrate_full_pipeline.InferenceConfig.create",
            new_callable=AsyncMock,
        ) as mock_inference_create,
        patch(
            "flows.orchestrate_full_pipeline.AggregationConfig.create",
            new_callable=AsyncMock,
        ) as mock_aggregate_create,
    ):
        # Setup mocks
        mock_inference_create.return_value = test_config
        mock_aggregate_create.return_value = test_aggregate_config
        mock_inference.return_value = mock_document_stems
        mock_aggregate.return_value = mock_run_identifier
        mock_indexing.return_value = None

        # Run the flow
        await orchestrate_full_pipeline()

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
        assert call_args.kwargs["batch_size"] == 1000

        mock_aggregate.assert_called_once()
        call_args = mock_aggregate.call_args
        assert call_args.kwargs["document_stems"] == mock_document_stems
        assert call_args.kwargs["config"] == test_aggregate_config
        assert call_args.kwargs["n_documents_in_batch"] == 20
        assert call_args.kwargs["n_batches"] == 5

        mock_indexing.assert_called_once()
        call_args = mock_indexing.call_args
        assert call_args.kwargs["run_output_identifier"] == mock_run_identifier
        assert call_args.kwargs["config"] == test_aggregate_config
        assert call_args.kwargs["batch_size"] == 50


@pytest.mark.asyncio
async def test_orchestrate_full_pipeline_with_full_config(
    test_config, test_aggregate_config
):
    """Test the flow with complete config provided."""

    mock_document_stems = [DocumentStem("test.doc.1"), DocumentStem("test.doc.2")]
    mock_run_identifier = RunOutputIdentifier("test-run-789")

    with (
        patch(
            "flows.orchestrate_full_pipeline.classifier_inference",
            new_callable=AsyncMock,
        ) as mock_inference,
        patch(
            "flows.orchestrate_full_pipeline.aggregate_inference_results",
            new_callable=AsyncMock,
        ) as mock_aggregate,
        patch(
            "flows.orchestrate_full_pipeline.run_indexing_from_aggregate_results",
            new_callable=AsyncMock,
        ) as mock_indexing,
    ):
        # Setup mocks
        mock_inference.return_value = mock_document_stems
        mock_aggregate.return_value = mock_run_identifier
        mock_indexing.return_value = None

        # Run the flow
        await orchestrate_full_pipeline(
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
            document_stems=mock_document_stems,
            config=test_aggregate_config,
            n_documents_in_batch=50,
            n_batches=3,
        )

        mock_indexing.assert_called_once_with(
            run_output_identifier=mock_run_identifier,
            config=test_aggregate_config,
            batch_size=200,
            indexer_concurrency_limit=2,
            indexer_document_passages_concurrency_limit=4,
            indexer_max_vespa_connections=8,
        )


# Test the orchestrate_full_pipeline flow with real sub-flows


@pytest.mark.asyncio
async def test_orchestrate_full_pipeline_with_real_sub_flows(
    mock_bucket_documents, test_config, test_aggregate_config
) -> None:
    """Test the flow with real sub-flows."""

    # Run the flow and sub flows against mock bucket documents
    await orchestrate_full_pipeline(
        inference_config=test_config,
        aggregation_config=test_aggregate_config,
    )
