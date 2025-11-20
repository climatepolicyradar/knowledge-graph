from unittest.mock import AsyncMock, Mock, patch

import pytest
from prefect.artifacts import Artifact
from prefect.client.schemas.objects import State, StateType
from prefect.exceptions import FailedRun
from prefect.states import Completed, Failed

from flows.aggregate import (
    DEFAULT_N_BATCHES,
    DEFAULT_N_DOCUMENTS_IN_BATCH,
    AggregateResult,
    RunOutputIdentifier,
)
from flows.boundary import DEFAULT_DOCUMENTS_BATCH_SIZE
from flows.classifier_specs.spec_interface import (
    ClassifierSpec,
    WikibaseID,
)
from flows.config import Config
from flows.full_pipeline import full_pipeline
from flows.inference import (
    INFERENCE_BATCH_SIZE_DEFAULT,
)
from flows.utils import DocumentImportId, DocumentStem, Fault


@pytest.mark.asyncio
async def test_full_pipeline_no_config_provided(
    test_config: Config,
    mock_run_output_identifier_str,
    aggregate_inference_results_document_stems,
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
            "flows.full_pipeline.Config.create",
            new_callable=AsyncMock,
        ) as mock_pipeline_config_create,
        patch(
            "flows.full_pipeline.get_async_session",
        ) as mock_get_session,
    ):
        # Setup mocks
        mock_pipeline_config_create.return_value = test_config

        # Mock S3 loading for document stems
        mock_s3_client = AsyncMock()
        mock_response = {
            "Body": AsyncMock(
                read=AsyncMock(
                    return_value=b'{"successful_document_stems": ["CCLW.executive.4934.1571", "CCLW.executive.10014.4470_translated_en"]}'
                )
            )
        }
        mock_s3_client.get_object = AsyncMock(return_value=mock_response)
        mock_client_context = AsyncMock()
        mock_client_context.__aenter__ = AsyncMock(return_value=mock_s3_client)
        mock_client_context.__aexit__ = AsyncMock(return_value=None)
        mock_session = Mock()
        mock_session.client = Mock(return_value=mock_client_context)
        mock_get_session.return_value = mock_session

        mock_inference.return_value = Completed(
            message="Successfully ran inference on all batches!",
            data=mock_run_output_identifier_str,
        )
        mock_aggregate.return_value = State(
            type=StateType.COMPLETED,
            data=AggregateResult(
                run_output_identifier=mock_run_output_identifier_str, errors=None
            ),
        )
        mock_indexing.return_value = State(
            type=StateType.COMPLETED, data={"message": "Indexing complete."}
        )

        # Run the flow
        await full_pipeline()

        # Verify default configs were created
        mock_pipeline_config_create.assert_called_once()

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
        assert call_args.kwargs["run_output_identifier"] == RunOutputIdentifier(
            mock_run_output_identifier_str
        )
        assert call_args.kwargs["config"] == test_config
        assert call_args.kwargs["n_documents_in_batch"] == DEFAULT_N_DOCUMENTS_IN_BATCH
        assert call_args.kwargs["n_batches"] == DEFAULT_N_BATCHES

        mock_indexing.assert_called_once()
        call_args = mock_indexing.call_args
        assert call_args.kwargs["run_output_identifier"] == RunOutputIdentifier(
            mock_run_output_identifier_str
        )
        assert call_args.kwargs["config"] == test_config
        assert call_args.kwargs["batch_size"] == DEFAULT_DOCUMENTS_BATCH_SIZE

        # Assert that the summary artifact was created
        summary_artifact = await Artifact.get("full-pipeline-results-summary-sandbox")
        print(f"Summary artifact {summary_artifact}")
        assert summary_artifact and summary_artifact.description
        assert (
            summary_artifact.description
            == "Summary of the full pipeline successful run."
        )


@pytest.mark.asyncio
async def test_full_pipeline_with_full_config(
    test_config,
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
        patch(
            "flows.full_pipeline.get_async_session",
        ) as mock_get_session,
    ):
        classifier_spec = ClassifierSpec(
            wikibase_id=WikibaseID("Q100"),
            classifier_id="zzzz9999",
            wandb_registry_version="v1",
        )

        # Mock S3 loading for document stems
        mock_s3_client = AsyncMock()
        mock_response = {
            "Body": AsyncMock(
                read=AsyncMock(
                    return_value=b'{"successful_document_stems": ["CCLW.executive.4934.1571", "CCLW.executive.10014.4470_translated_en"]}'
                )
            )
        }
        mock_s3_client.get_object = AsyncMock(return_value=mock_response)
        mock_client_context = AsyncMock()
        mock_client_context.__aenter__ = AsyncMock(return_value=mock_s3_client)
        mock_client_context.__aexit__ = AsyncMock(return_value=None)
        mock_session = Mock()
        mock_session.client = Mock(return_value=mock_client_context)
        mock_get_session.return_value = mock_session

        # Setup mocks
        mock_inference.return_value = Completed(
            message="Successfully ran inference on all batches!",
            data=mock_run_output_identifier_str,
        )
        mock_aggregate.return_value = State(
            type=StateType.COMPLETED,
            data=AggregateResult(
                run_output_identifier=mock_run_output_identifier_str, errors=None
            ),
        )
        mock_indexing.return_value = State(
            type=StateType.COMPLETED, data={"message": "Indexing complete."}
        )

        # Run the flow
        await full_pipeline(
            config=test_config,
            classifier_specs=[classifier_spec],
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
        assert call_args.kwargs["classifier_specs"] == [classifier_spec]
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
        assert (
            call_args.kwargs["run_output_identifier"] == mock_run_output_identifier_str
        )
        assert call_args.kwargs["config"] == test_config
        assert call_args.kwargs["n_documents_in_batch"] == 50
        assert call_args.kwargs["n_batches"] == 3

        mock_indexing.assert_called_once_with(
            run_output_identifier=RunOutputIdentifier(mock_run_output_identifier_str),
            config=test_config,
            batch_size=200,
            indexer_concurrency_limit=2,
            indexer_document_passages_concurrency_limit=4,
            indexer_max_vespa_connections=8,
            return_state=True,
        )

        # Assert that the summary artifact was created
        summary_artifact = await Artifact.get("full-pipeline-results-summary-sandbox")
        print(f"Summary artifact {summary_artifact}")
        assert summary_artifact and summary_artifact.description
        assert (
            summary_artifact.description
            == "Summary of the full pipeline successful run."
        )


@pytest.mark.asyncio
async def test_full_pipeline_with_inference_failure(
    test_config,
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
        document_stems_successful = [DocumentStem("CCLW.executive.2.2")]
        classifier_spec = ClassifierSpec(
            wikibase_id=WikibaseID("Q100"),
            classifier_id="zzzz9999",
            wandb_registry_version="v1",
        )

        # Setup mocks
        mock_inference.return_value = Failed(
            message="Some inference batches had failures!",
            data=Fault(
                msg="Some inference batches had failures!",
                metadata={},
                data={
                    "successful_document_stems": set(document_stems_successful),
                    "run_output_identifier": mock_run_output_identifier_str,
                },
            ),
        )
        mock_aggregate.return_value = State(
            type=StateType.COMPLETED,
            data=AggregateResult(
                run_output_identifier=mock_run_output_identifier_str, errors=None
            ),
        )
        mock_indexing.return_value = State(
            type=StateType.COMPLETED, data={"message": "Indexing complete."}
        )

        # Run the flow expecting aggregation and indexing to run on successful documents.
        await full_pipeline(
            config=test_config,
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
        assert (
            call_args.kwargs["run_output_identifier"] == mock_run_output_identifier_str
        )
        assert call_args.kwargs["n_documents_in_batch"] == 50
        assert call_args.kwargs["n_batches"] == 3

        mock_indexing.assert_called_once_with(
            run_output_identifier=RunOutputIdentifier(mock_run_output_identifier_str),
            config=test_config,
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
                config=test_config,
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


@pytest.mark.asyncio
async def test_full_pipeline_completes_after_some_docs_fail_inference_and_aggregation(
    test_config,
    mock_run_output_identifier_str,
):
    """Test that indexing flows completes after some documents fail on aggregation."""

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
        document_stems_successful = [DocumentStem("CCLW.executive.2.2")]
        classifier_spec = ClassifierSpec(
            wikibase_id=WikibaseID("Q100"),
            classifier_id="zzzz9999",
            wandb_registry_version="v1",
        )

        # Setup mocks
        mock_inference.return_value = Failed(
            message="Some inference batches had failures!",
            data=Fault(
                msg="Some inference batches had failures!",
                metadata={},
                data={
                    "successful_document_stems": set(document_stems_successful),
                    "run_output_identifier": mock_run_output_identifier_str,
                },
            ),
        )

        # aggregation state contains failed documents
        mock_aggregate.return_value = State(
            type=StateType.COMPLETED,
            data=AggregateResult(
                run_output_identifier=mock_run_output_identifier_str,
                errors="1/2 Documents failed",
            ),
        )

        mock_indexing.return_value = State(
            type=StateType.COMPLETED, data={"message": "Indexing complete."}
        )

        # Run the flow
        await full_pipeline(
            config=test_config,
            classifier_specs=[classifier_spec],
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
        assert call_args.kwargs["classifier_specs"] == [classifier_spec]
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
        assert (
            call_args.kwargs["run_output_identifier"] == mock_run_output_identifier_str
        )

        assert call_args.kwargs["config"] == test_config
        assert call_args.kwargs["n_documents_in_batch"] == 50
        assert call_args.kwargs["n_batches"] == 3

        mock_indexing.assert_called_once_with(
            run_output_identifier=RunOutputIdentifier(mock_run_output_identifier_str),
            config=test_config,
            batch_size=200,
            indexer_concurrency_limit=2,
            indexer_document_passages_concurrency_limit=4,
            indexer_max_vespa_connections=8,
            return_state=True,
        )

        # Assert that the summary artifact was created
        summary_artifact = await Artifact.get("full-pipeline-results-summary-sandbox")
        print(f"Summary artifact {summary_artifact}")
        assert summary_artifact and summary_artifact.description
        assert (
            summary_artifact.description
            == "Summary of the full pipeline successful run."
        )

        # assert pipeline completed all three flows despite inference and aggregation failures
        assert mock_inference.call_count == 1
        assert mock_aggregate.call_count == 1
        assert mock_indexing.call_count == 1
