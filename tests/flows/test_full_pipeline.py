from unittest.mock import AsyncMock, patch

import pytest
from prefect.client.schemas.objects import State, StateType
from prefect.exceptions import FailedRun
from prefect.states import Completed, Failed

from flows.aggregate import (
    DEFAULT_N_BATCHES,
    DEFAULT_N_DOCUMENTS_IN_BATCH,
    RunOutputIdentifier,
)
from flows.aggregate import Config as AggregationConfig
from flows.boundary import DEFAULT_DOCUMENTS_BATCH_SIZE
from flows.full_pipeline import (
    full_pipeline,
    validate_aggregation_inference_configs,
)
from flows.inference import (
    INFERENCE_BATCH_SIZE_DEFAULT,
    BatchInferenceResult,
    InferenceResult,
)
from flows.inference import (
    Config as InferenceConfig,
)
from flows.utils import DocumentImportId, DocumentStem, Fault
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
            "flows.full_pipeline.inference",
            new_callable=AsyncMock,
        ) as mock_inference,
        patch(
            "flows.full_pipeline.aggregate",
            new_callable=AsyncMock,
        ) as mock_aggregate,
        patch(
            "flows.full_pipeline.index",
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

        mock_inference.return_value = Completed(
            message="Successfully ran inference on all batches!",
            data=InferenceResult(
                batch_inference_results=[
                    BatchInferenceResult(
                        successful_document_stems=list(
                            aggregate_inference_results_document_stems
                        ),
                        failed_document_stems=[],
                        classifier_name="Q100",
                        classifier_alias="v1",
                    ),
                ],
                unexpected_failures=[],
            ).model_dump(),
        )
        mock_aggregate.return_value = State(
            type=StateType.COMPLETED,
            data=RunOutputIdentifier(mock_run_output_identifier_str),
        )
        mock_indexing.return_value = State(
            type=StateType.COMPLETED, data={"message": "Indexing complete."}
        )

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
        assert sorted(call_args.kwargs["document_stems"]) == sorted(
            aggregate_inference_results_document_stems
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
            "flows.full_pipeline.inference",
            new_callable=AsyncMock,
        ) as mock_inference,
        patch(
            "flows.full_pipeline.aggregate",
            new_callable=AsyncMock,
        ) as mock_aggregate,
        patch(
            "flows.full_pipeline.index",
            new_callable=AsyncMock,
        ) as mock_indexing,
    ):
        # Setup mocks
        mock_inference.return_value = Completed(
            message="Successfully ran inference on all batches!",
            data=InferenceResult(
                batch_inference_results=[
                    BatchInferenceResult(
                        successful_document_stems=aggregate_inference_results_document_stems,
                        failed_document_stems=[],
                        classifier_name="Q100",
                        classifier_alias="v1",
                    ),
                ],
                unexpected_failures=[],
            ).model_dump(),
        )
        mock_aggregate.return_value = State(
            type=StateType.COMPLETED,
            data=RunOutputIdentifier(mock_run_output_identifier_str),
        )
        mock_indexing.return_value = State(
            type=StateType.COMPLETED, data={"message": "Indexing complete."}
        )

        # Run the flow
        await full_pipeline(
            inference_config=test_config,
            aggregation_config=test_aggregate_config,
            classifier_specs=[ClassifierSpec(name="Q123", alias="v1")],
            document_ids=[
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
        mock_inference.assert_called_once()
        call_args = mock_inference.call_args
        assert call_args.kwargs["classifier_specs"] == [
            ClassifierSpec(name="Q123", alias="v1")
        ]
        assert sorted(call_args.kwargs["document_ids"]) == sorted(
            [
                DocumentImportId("test.doc.1"),
                DocumentImportId("test.doc.2"),
            ]
        )
        assert call_args.kwargs["use_new_and_updated"] is True
        assert call_args.kwargs["config"] == test_config
        assert call_args.kwargs["batch_size"] == 500
        assert call_args.kwargs["classifier_concurrency_limit"] == 5

        mock_aggregate.assert_called_once()
        call_args = mock_aggregate.call_args
        assert sorted(call_args.kwargs["document_stems"]) == sorted(
            aggregate_inference_results_document_stems
        )
        assert call_args.kwargs["config"] == test_aggregate_config
        assert call_args.kwargs["n_documents_in_batch"] == 50
        assert call_args.kwargs["n_batches"] == 3

        mock_indexing.assert_called_once_with(
            run_output_identifier=RunOutputIdentifier(mock_run_output_identifier_str),
            config=test_aggregate_config,
            batch_size=200,
            indexer_concurrency_limit=2,
            indexer_document_passages_concurrency_limit=4,
            indexer_max_vespa_connections=8,
            return_state=True,
        )


@pytest.mark.asyncio
async def test_full_pipeline_with_inference_failure(
    test_config,
    test_aggregate_config,
    mock_run_output_identifier_str,
):
    """Test the flows handling of inference failures modes."""

    # Mock the sub-flows
    with (
        patch(
            "flows.full_pipeline.inference",
            new_callable=AsyncMock,
        ) as mock_inference,
        patch(
            "flows.full_pipeline.aggregate",
            new_callable=AsyncMock,
        ) as mock_aggregate,
        patch(
            "flows.full_pipeline.index",
            new_callable=AsyncMock,
        ) as mock_indexing,
    ):
        document_ids = [
            DocumentImportId("CCLW.executive.1.1"),
            DocumentImportId("CCLW.executive.2.2"),
        ]
        document_stems_failed = [
            (DocumentStem("CCLW.executive.1.1"), Exception("Test error"))
        ]
        document_stems_successful = [DocumentStem("CCLW.executive.2.2")]
        classifier_spec = ClassifierSpec(name="Q100", alias="v1")

        # Setup mocks
        mock_inference.return_value = Failed(
            message="Some inference batches had failures!",
            data=Fault(
                msg="Some inference batches had failures!",
                metadata={},
                data=InferenceResult(
                    batch_inference_results=[
                        BatchInferenceResult(
                            successful_document_stems=document_stems_successful,
                            failed_document_stems=document_stems_failed,
                            classifier_name=classifier_spec.name,
                            classifier_alias=classifier_spec.alias,
                        ),
                    ],
                    unexpected_failures=[],
                ).model_dump(),
            ),
        )
        mock_aggregate.return_value = State(
            type=StateType.COMPLETED,
            data=RunOutputIdentifier(mock_run_output_identifier_str),
        )
        mock_indexing.return_value = State(
            type=StateType.COMPLETED, data={"message": "Indexing complete."}
        )

        # Run the flow expecting aggregation and indexing to run on successful documents.
        await full_pipeline(
            inference_config=test_config,
            aggregation_config=test_aggregate_config,
            classifier_specs=[classifier_spec],
            document_ids=document_ids,
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
        mock_inference.assert_called_once()
        call_args = mock_inference.call_args
        assert call_args.kwargs["classifier_specs"] == [classifier_spec]
        assert sorted(call_args.kwargs["document_ids"]) == sorted(
            [
                DocumentImportId("CCLW.executive.1.1"),
                DocumentImportId("CCLW.executive.2.2"),
            ]
        )
        assert call_args.kwargs["use_new_and_updated"] is True
        assert call_args.kwargs["config"] == test_config
        assert call_args.kwargs["batch_size"] == 500
        assert call_args.kwargs["classifier_concurrency_limit"] == 5

        mock_aggregate.assert_called_once()
        call_args = mock_aggregate.call_args
        assert sorted(call_args.kwargs["document_stems"]) == sorted(
            document_stems_successful
        )
        assert call_args.kwargs["config"] == test_aggregate_config
        assert call_args.kwargs["n_documents_in_batch"] == 50
        assert call_args.kwargs["n_batches"] == 3

        mock_indexing.assert_called_once_with(
            run_output_identifier=RunOutputIdentifier(mock_run_output_identifier_str),
            config=test_aggregate_config,
            batch_size=200,
            indexer_concurrency_limit=2,
            indexer_document_passages_concurrency_limit=4,
            indexer_max_vespa_connections=8,
            return_state=True,
        )

        # Run the flow expecting aggregation and indexing not to run.
        mock_inference.reset_mock()
        mock_aggregate.reset_mock()
        mock_indexing.reset_mock()

        mock_inference.return_value = Failed(
            message="Test error", result=Exception("Test exception")
        )

        with pytest.raises(FailedRun, match="Test error"):
            await full_pipeline(
                inference_config=test_config,
                aggregation_config=test_aggregate_config,
                classifier_specs=[classifier_spec],
                document_ids=document_ids,
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

        assert mock_inference.call_count == 1
        assert mock_aggregate.call_count == 0
        assert mock_indexing.call_count == 0
