import json
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
from flows.full_pipeline import topic_pipeline
from flows.inference import (
    INFERENCE_BATCH_SIZE_DEFAULT,
)
from flows.utils import DocumentImportId, DocumentStem, Fault


@pytest.mark.asyncio
async def test_topic_pipeline_no_config_provided(
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
        # Index returns None on success, need to mock the result() method
        mock_indexing_state = AsyncMock()
        mock_indexing_state.result = AsyncMock(return_value=None)
        mock_indexing.return_value = mock_indexing_state

        # Run the flow
        await topic_pipeline()

        # Verify default configs were created
        mock_pipeline_config_create.assert_called_once()

        # Verify sub-flows were called with correct parameters
        mock_inference.assert_called_once()
        call_args = mock_inference.call_args
        assert call_args.kwargs["config"] == test_config
        assert call_args.kwargs["classifier_specs"] is None
        assert call_args.kwargs["document_ids"] is None
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
        summary_artifact = await Artifact.get("topic-pipeline-results-summary-sandbox")
        print(f"Summary artifact {summary_artifact}")
        assert summary_artifact and summary_artifact.description
        assert (
            summary_artifact.description
            == "Summary of the topic pipeline successful run."
        )


@pytest.mark.asyncio
async def test_topic_pipeline_with_full_config(
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
        # Index returns None on success, need to mock the result() method
        mock_indexing_state = AsyncMock()
        mock_indexing_state.result = AsyncMock(return_value=None)
        mock_indexing.return_value = mock_indexing_state

        # Run the flow
        await topic_pipeline(
            config=test_config,
            classifier_specs=[classifier_spec],
            document_ids=[
                DocumentImportId("test.doc.1"),
                DocumentImportId("test.doc.2"),
            ],
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
            enable_v2_concepts=None,
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
async def test_topic_pipeline_with_inference_failure(
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
        # Index returns None on success, need to mock the result() method
        mock_indexing_state = AsyncMock()
        mock_indexing_state.result = AsyncMock(return_value=None)
        mock_indexing.return_value = mock_indexing_state

        # Run the flow expecting aggregation and indexing to run on successful documents.
        with pytest.raises(Fault, match="Some inference batches had failures!"):
            await topic_pipeline(
                config=test_config,
                classifier_specs=[classifier_spec],
                document_ids=document_ids,
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
            enable_v2_concepts=None,
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
            await topic_pipeline(
                config=test_config,
                classifier_specs=[classifier_spec],
                document_ids=document_ids,
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
async def test_topic_pipeline_completes_after_some_docs_fail_inference_and_aggregation(
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
            data=Fault(
                msg="1/2 Documents failed",
                metadata={},
                data=AggregateResult(
                    run_output_identifier=mock_run_output_identifier_str,
                    errors="1/2 Documents failed",
                ),
            ),
        )

        # Index returns None on success, need to mock the result() method
        mock_indexing_state = AsyncMock()
        mock_indexing_state.result = AsyncMock(return_value=None)
        mock_indexing.return_value = mock_indexing_state

        # Run the flow and expect an exception to be returned
        with pytest.raises(
            Fault,
            match="Some inference batches had failures!",
        ):
            await topic_pipeline(
                config=test_config,
                classifier_specs=[classifier_spec],
                document_ids=[
                    DocumentImportId("test.doc.1"),
                    DocumentImportId("test.doc.2"),
                ],
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
            enable_v2_concepts=None,
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


@pytest.mark.asyncio
async def test_topic_pipeline_with_document_ids_s3_path(
    test_config,
    mock_run_output_identifier_str,
    aggregate_inference_results_document_stems,
    mock_async_bucket,
    mock_s3_async_client,
):
    """Test topic_pipeline flow with document_ids_s3_path parameter."""
    classifier_spec = ClassifierSpec(
        wikibase_id=WikibaseID("Q100"),
        classifier_id="zzzz9999",
        wandb_registry_version="v1",
    )

    s3_key: str = "test-document-ids.txt"
    s3_path: str = f"s3://{test_config.cache_bucket}/" + s3_key

    document_ids = [
        DocumentImportId("test.doc.1"),
        DocumentImportId("test.doc.2"),
    ]
    file_content: str = json.dumps(document_ids)

    await mock_s3_async_client.put_object(
        Bucket=test_config.cache_bucket,
        Key=s3_key,
        Body=file_content.encode("utf-8"),
    )

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
        # Mock S3 loading for document stems
        mock_s3_client = AsyncMock()
        mock_response = {
            "Body": AsyncMock(
                read=AsyncMock(
                    return_value=b'{"successful_document_stems": ["test.doc.1", "test.doc.2"]}'
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
        # Index returns None on success, need to mock the result() method
        mock_indexing_state = AsyncMock()
        mock_indexing_state.result = AsyncMock(return_value=None)
        mock_indexing.return_value = mock_indexing_state

        # Run the flow with document_ids_s3_path
        await topic_pipeline(
            config=test_config,
            classifier_specs=[classifier_spec],
            document_ids_s3_path=s3_path,
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
        assert call_args.kwargs["document_ids_s3_path"] == s3_path
        assert call_args.kwargs["config"] == test_config
        assert call_args.kwargs["batch_size"] == 500
        assert call_args.kwargs["classifier_concurrency_limit"] == 5

        mock_aggregate.assert_called_once()
        mock_indexing.assert_called_once()


@pytest.mark.asyncio
async def test_topic_pipeline_uses_aggregation_run_output_identifier_for_indexing(
    test_config,
    mock_run_output_identifier_str,
):
    """
    Test that index stage receives run_output_identifier from aggregation, not inference.

    This test verifies the fix for the bug where index was receiving the inference
    run_output_identifier instead of the aggregation run_output_identifier, causing
    it to look for documents in the wrong S3 location.
    """

    # Create distinct identifiers for each stage
    inference_run_id = "2025-12-17T14:49-inference-id"
    aggregation_run_id = "2025-12-17T16:12-aggregation-id"

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
        # Mock S3 loading for document stems from inference
        mock_s3_client = AsyncMock()
        mock_response = {
            "Body": AsyncMock(
                read=AsyncMock(
                    return_value=b'{"successful_document_stems": ["CCLW.executive.1.1", "CCLW.executive.2.2"]}'
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

        # Create mock State objects with properly mocked .result() methods
        # Inference returns its own run_output_identifier
        mock_inference_state = AsyncMock()
        mock_inference_state.result = AsyncMock(return_value=inference_run_id)
        mock_inference.return_value = mock_inference_state

        # Aggregation returns a DIFFERENT run_output_identifier (as it does in production)
        aggregation_result = AggregateResult(
            run_output_identifier=aggregation_run_id,  # Different from inference!
            errors=None,
        )
        mock_aggregate_state = AsyncMock()
        mock_aggregate_state.result = AsyncMock(return_value=aggregation_result)
        mock_aggregate.return_value = mock_aggregate_state

        # Index returns None on success
        mock_indexing_state = AsyncMock()
        mock_indexing_state.result = AsyncMock(return_value=None)
        mock_indexing.return_value = mock_indexing_state

        # Run the flow
        await topic_pipeline(config=test_config)

        # Verify that aggregation was called with inference's run_output_identifier
        aggregate_call_args = mock_aggregate.call_args
        assert aggregate_call_args.kwargs["run_output_identifier"] == inference_run_id

        # Verify that index was called with AGGREGATION's run_output_identifier
        # NOT inference's run_output_identifier
        index_call_args = mock_indexing.call_args
        assert index_call_args.kwargs["run_output_identifier"] == RunOutputIdentifier(
            aggregation_run_id
        ), (
            f"Index should use aggregation's run_output_identifier ({aggregation_run_id}), "
            f"not inference's ({inference_run_id})"
        )

        # This is the bug we're testing for:
        #
        # If this assertion fails, index is looking in the wrong S3 location
        assert index_call_args.kwargs["run_output_identifier"] != RunOutputIdentifier(
            inference_run_id
        ), "Index should NOT use inference's run_output_identifier"
