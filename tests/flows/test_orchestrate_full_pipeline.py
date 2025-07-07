from unittest.mock import AsyncMock, patch

import pytest

from flows.aggregate_inference_results import Config as AggregationConfig
from flows.aggregate_inference_results import RunOutputIdentifier
from flows.inference import Config as InferenceConfig
from flows.orchestrate_full_pipeline import (
    OrchestrateFullPipelineConfig,
    orchestrate_full_pipeline,
)
from flows.utils import DocumentImportId, DocumentStem
from scripts.cloud import AwsEnv, ClassifierSpec


def test_orchestrate_full_pipeline_config() -> None:
    """Test the OrchestrateFullPipelineConfig object."""

    config = OrchestrateFullPipelineConfig(
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
    assert config is not None
    assert isinstance(config.inference_config, InferenceConfig)


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
def test_orchestrate_full_pipeline_config_validation(
    inference_config: InferenceConfig,
    aggregation_config: AggregationConfig,
    expected_error: str,
) -> None:
    """Test the OrchestrateFullPipelineConfig object validation."""

    with pytest.raises(ValueError) as e:
        _ = OrchestrateFullPipelineConfig(
            inference_config=inference_config,
            aggregation_config=aggregation_config,
        )
    assert expected_error in str(e.value)


@pytest.mark.asyncio
async def test_orchestrate_full_pipeline_no_config_provided(
    test_config: InferenceConfig, test_aggregate_config: AggregationConfig
) -> None:
    """Test the flow when no config is provided - should create default configs."""

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
async def test_orchestrate_full_pipeline_with_partial_config(
    test_config, test_aggregate_config
):
    """Test the flow when no config is provided - should create default configs."""

    mock_document_stems = [DocumentStem("test.doc.1")]
    mock_run_identifier = RunOutputIdentifier("test-run-456")

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

        # Run the flow with no config - it should create default configs
        await orchestrate_full_pipeline()

        # Verify default configs were created
        mock_inference_create.assert_called_once()
        mock_aggregate_create.assert_called_once()

        # Verify sub-flows were called with correct parameters
        mock_inference.assert_called_once_with(
            classifier_specs=None,
            document_ids=None,
            use_new_and_updated=False,
            config=test_config,
            batch_size=1000,
            classifier_concurrency_limit=20,
        )


@pytest.mark.asyncio
async def test_orchestrate_full_pipeline_with_full_config(
    test_config, test_aggregate_config
):
    """Test the flow with complete config provided."""

    # Create a complete config
    config = OrchestrateFullPipelineConfig(
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
        await orchestrate_full_pipeline(config=config)

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


@pytest.mark.asyncio
async def test_orchestrate_full_pipeline_inference_failure(
    test_config, test_aggregate_config
):
    """Test the flow when inference step fails."""

    config = OrchestrateFullPipelineConfig(
        inference_config=test_config,
        aggregation_config=test_aggregate_config,
    )

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
        # Setup inference to fail
        mock_inference.side_effect = Exception("Inference failed")

        # Run the flow and expect it to fail
        with pytest.raises(Exception, match="Inference failed"):
            await orchestrate_full_pipeline(config=config)

        # Verify other steps were not called
        mock_aggregate.assert_not_called()
        mock_indexing.assert_not_called()


@pytest.mark.asyncio
async def test_orchestrate_full_pipeline_aggregation_failure(
    test_config, test_aggregate_config
):
    """Test the flow when aggregation step fails."""

    config = OrchestrateFullPipelineConfig(
        inference_config=test_config,
        aggregation_config=test_aggregate_config,
    )

    mock_document_stems = [DocumentStem("test.doc.1")]

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
        mock_aggregate.side_effect = Exception("Aggregation failed")

        # Run the flow and expect it to fail
        with pytest.raises(Exception, match="Aggregation failed"):
            await orchestrate_full_pipeline(config=config)

        # Verify inference was called but indexing was not
        mock_inference.assert_called_once()
        mock_indexing.assert_not_called()


@pytest.mark.asyncio
async def test_orchestrate_full_pipeline_indexing_failure(
    test_config, test_aggregate_config
):
    """Test the flow when indexing step fails."""

    config = OrchestrateFullPipelineConfig(
        inference_config=test_config,
        aggregation_config=test_aggregate_config,
    )

    mock_document_stems = [DocumentStem("test.doc.1")]
    mock_run_identifier = RunOutputIdentifier("test-run-999")

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
        mock_indexing.side_effect = Exception("Indexing failed")

        # Run the flow and expect it to fail
        with pytest.raises(Exception, match="Indexing failed"):
            await orchestrate_full_pipeline(config=config)

        # Verify inference and aggregation were called
        mock_inference.assert_called_once()
        mock_aggregate.assert_called_once()


@pytest.mark.asyncio
async def test_orchestrate_full_pipeline_data_flow(test_config, test_aggregate_config):
    """Test that data flows correctly between pipeline stages."""

    config = OrchestrateFullPipelineConfig(
        inference_config=test_config,
        aggregation_config=test_aggregate_config,
    )

    # Create realistic test data
    input_document_stems = [
        DocumentStem("CCLW.executive.10061.4515"),
        DocumentStem("UNFCCC.party.492.0"),
        DocumentStem("CPR.document.i00000549.n0000"),
    ]
    output_run_identifier = RunOutputIdentifier("2024-01-15T10-30-45-test-run")

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
        mock_inference.return_value = input_document_stems
        mock_aggregate.return_value = output_run_identifier
        mock_indexing.return_value = None

        # Run the flow
        await orchestrate_full_pipeline(config=config)

        # Verify data flows correctly between stages
        mock_inference.assert_called_once()

        # Verify aggregation receives the document stems from inference
        mock_aggregate.assert_called_once()
        aggregate_call_args = mock_aggregate.call_args
        assert aggregate_call_args.kwargs["document_stems"] == input_document_stems

        # Verify indexing receives the run identifier from aggregation
        mock_indexing.assert_called_once()
        indexing_call_args = mock_indexing.call_args
        assert (
            indexing_call_args.kwargs["run_output_identifier"] == output_run_identifier
        )


@pytest.mark.asyncio
async def test_orchestrate_full_pipeline_with_classifier_specs(
    test_config, test_aggregate_config
):
    """Test the flow with specific classifier specs."""

    classifier_specs = [
        ClassifierSpec(name="Q123", alias="v4"),
        ClassifierSpec(name="Q223", alias="v3"),
        ClassifierSpec(name="Q218", alias="v5"),
    ]

    config = OrchestrateFullPipelineConfig(
        inference_config=test_config,
        aggregation_config=test_aggregate_config,
        inference_classifier_specs=classifier_specs,
    )

    mock_document_stems = [DocumentStem("test.doc.1")]
    mock_run_identifier = RunOutputIdentifier("test-run-specs")

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
        await orchestrate_full_pipeline(config=config)

        # Verify classifier specs were passed correctly
        mock_inference.assert_called_once()
        call_args = mock_inference.call_args
        assert call_args.kwargs["classifier_specs"] == classifier_specs


@pytest.mark.asyncio
async def test_orchestrate_full_pipeline_with_document_ids(
    test_config, test_aggregate_config
):
    """Test the flow with specific document IDs."""

    document_ids = ["CCLW.executive.10061.4515", "UNFCCC.party.492.0"]

    config = OrchestrateFullPipelineConfig(
        inference_config=test_config,
        aggregation_config=test_aggregate_config,
        inference_document_ids=[DocumentImportId(doc_id) for doc_id in document_ids],
    )

    mock_document_stems = [DocumentStem(doc_id) for doc_id in document_ids]
    mock_run_identifier = RunOutputIdentifier("test-run-docs")

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
        await orchestrate_full_pipeline(config=config)

        # Verify document IDs were passed correctly
        mock_inference.assert_called_once()
        call_args = mock_inference.call_args
        assert call_args.kwargs["document_ids"] == [
            DocumentImportId(doc_id) for doc_id in document_ids
        ]


@pytest.mark.asyncio
async def test_orchestrate_full_pipeline_with_new_and_updated_flag(
    test_config, test_aggregate_config
):
    """Test the flow with use_new_and_updated flag set."""

    config = OrchestrateFullPipelineConfig(
        inference_config=test_config,
        aggregation_config=test_aggregate_config,
        inference_use_new_and_updated=True,
    )

    mock_document_stems = [DocumentStem("new.doc.1"), DocumentStem("updated.doc.2")]
    mock_run_identifier = RunOutputIdentifier("test-run-new-updated")

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
        await orchestrate_full_pipeline(config=config)

        # Verify new_and_updated flag was passed correctly
        mock_inference.assert_called_once()
        call_args = mock_inference.call_args
        assert call_args.kwargs["use_new_and_updated"] is True
