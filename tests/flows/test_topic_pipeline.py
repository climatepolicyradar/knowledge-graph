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
from flows.classifier_specs.spec_interface import (
    ClassifierSpec,
    WikibaseID,
)
from flows.config import Config
from flows.inference import (
    INFERENCE_BATCH_SIZE_DEFAULT,
)
from flows.topic_pipeline import topic_pipeline
from flows.utils import AwsEnv, DocumentImportId, DocumentStem, Fault


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
            "flows.topic_pipeline.inference",
            new_callable=AsyncMock,
        ) as mock_inference,
        patch(
            "flows.topic_pipeline.aggregate",
            new_callable=AsyncMock,
        ) as mock_aggregate,
        patch(
            "flows.topic_pipeline.Config.create",
            new_callable=AsyncMock,
        ) as mock_pipeline_config_create,
        patch(
            "flows.topic_pipeline.get_async_session",
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

        # Assert that the summary artifact was created
        summary_artifact = await Artifact.get("topic-pipeline-results-summary-sandbox")
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
            "flows.topic_pipeline.inference",
            new_callable=AsyncMock,
        ) as mock_inference,
        patch(
            "flows.topic_pipeline.aggregate",
            new_callable=AsyncMock,
        ) as mock_aggregate,
        patch(
            "flows.topic_pipeline.get_async_session",
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

        # Run the flow
        await topic_pipeline(
            config=test_config,
            classifier_specs=[classifier_spec],
            document_ids=[
                DocumentImportId("test.doc.1"),
                DocumentImportId("test.doc.2"),
            ],
            inference_batch_size=500,
            inference_cpu_concurrency_limit=5,
            inference_gpu_concurrency_limit=5,
            aggregation_n_documents_in_batch=50,
            aggregation_n_batches=3,
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
        assert call_args.kwargs["classifier_cpu_concurrency_limit"] == 5
        assert call_args.kwargs["classifier_gpu_concurrency_limit"] == 5

        mock_aggregate.assert_called_once()
        call_args = mock_aggregate.call_args
        assert (
            call_args.kwargs["run_output_identifier"] == mock_run_output_identifier_str
        )
        assert call_args.kwargs["config"] == test_config
        assert call_args.kwargs["n_documents_in_batch"] == 50
        assert call_args.kwargs["n_batches"] == 3
        # config.aws_env is not production in test_config, so classifier_specs
        # should be forwarded through to aggregate unchanged.
        assert call_args.kwargs["classifier_specs"] == [classifier_spec]

        # Assert that the summary artifact was created
        summary_artifact = await Artifact.get("topic-pipeline-results-summary-sandbox")
        assert summary_artifact and summary_artifact.description
        assert (
            summary_artifact.description
            == "Summary of the topic pipeline successful run."
        )


@pytest.mark.asyncio
async def test_topic_pipeline_does_not_pass_classifier_specs_to_aggregate_in_production(
    test_config,
    aggregate_inference_results_document_stems,
    mock_run_output_identifier_str,
):
    """
    Test that classifier_specs are never forwarded to aggregate in production,
    even when explicitly provided - this is a deliberate safety behaviour so
    that a manually-triggered run can never wipe existing concepts for users.
    """
    # NOTE: this assumes `Config` is a Pydantic model supporting
    # `.model_copy(update=...)`. If `Config` is constructed differently,
    # adjust how `production_config` below is built accordingly.
    production_config = test_config.model_copy(update={"aws_env": AwsEnv.production})

    with (
        patch(
            "flows.topic_pipeline.inference",
            new_callable=AsyncMock,
        ) as mock_inference,
        patch(
            "flows.topic_pipeline.aggregate",
            new_callable=AsyncMock,
        ) as mock_aggregate,
        patch(
            "flows.topic_pipeline.get_async_session",
        ) as mock_get_session,
    ):
        classifier_spec = ClassifierSpec(
            wikibase_id=WikibaseID("Q100"),
            classifier_id="zzzz9999",
            wandb_registry_version="v1",
        )

        mock_s3_client = AsyncMock()
        mock_response = {
            "Body": AsyncMock(
                read=AsyncMock(
                    return_value=b'{"successful_document_stems": ["CCLW.executive.4934.1571"]}'
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

        await topic_pipeline(
            config=production_config,
            classifier_specs=[classifier_spec],
            document_ids=[DocumentImportId("test.doc.1")],
        )

        mock_aggregate.assert_called_once()
        call_args = mock_aggregate.call_args
        # Explicitly provided classifier_specs must be nulled out for
        # production runs.
        assert call_args.kwargs["classifier_specs"] is None


@pytest.mark.asyncio
async def test_topic_pipeline_with_inference_failure(
    test_config,
    mock_run_output_identifier_str,
):
    """Test the flows handling of inference failures modes."""

    # Mock the sub-flows
    with (
        patch(
            "flows.topic_pipeline.inference",
            new_callable=AsyncMock,
        ) as mock_inference,
        patch(
            "flows.topic_pipeline.aggregate",
            new_callable=AsyncMock,
        ) as mock_aggregate,
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
                loggable_data={},
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
        # Run the flow expecting aggregation to run on successful documents.
        with pytest.raises(Fault, match="Some inference batches had failures!"):
            await topic_pipeline(
                config=test_config,
                classifier_specs=[classifier_spec],
                document_ids=document_ids,
                inference_batch_size=500,
                inference_cpu_concurrency_limit=5,
                inference_gpu_concurrency_limit=5,
                aggregation_n_documents_in_batch=50,
                aggregation_n_batches=3,
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
        assert call_args.kwargs["classifier_cpu_concurrency_limit"] == 5
        assert call_args.kwargs["classifier_gpu_concurrency_limit"] == 5

        mock_aggregate.assert_called_once()
        call_args = mock_aggregate.call_args
        assert (
            call_args.kwargs["run_output_identifier"] == mock_run_output_identifier_str
        )
        assert call_args.kwargs["n_documents_in_batch"] == 50
        assert call_args.kwargs["n_batches"] == 3

        # Run the flow expecting aggregation and indexing not to run.
        mock_inference.reset_mock()
        mock_aggregate.reset_mock()

        mock_inference.return_value = Failed(
            message="Test error", result=Exception("Test exception")
        )

        with pytest.raises(FailedRun, match="Test error"):
            await topic_pipeline(
                config=test_config,
                classifier_specs=[classifier_spec],
                document_ids=document_ids,
                inference_batch_size=500,
                inference_cpu_concurrency_limit=5,
                inference_gpu_concurrency_limit=5,
                aggregation_n_documents_in_batch=50,
                aggregation_n_batches=3,
            )

        assert mock_inference.call_count == 1
        assert mock_aggregate.call_count == 0


@pytest.mark.asyncio
async def test_topic_pipeline_with_inference_unexpected_result_type(
    test_config,
):
    """
    Test the flow's inference match statement raises ValueError when
    inference resolves to a result type that is neither a Fault, an
    Exception, nor a str run_output_identifier.
    """
    with (
        patch(
            "flows.topic_pipeline.inference",
            new_callable=AsyncMock,
        ) as mock_inference,
        patch(
            "flows.topic_pipeline.aggregate",
            new_callable=AsyncMock,
        ) as mock_aggregate,
    ):
        # An int is not a valid inference result shape.
        mock_inference.return_value = Completed(
            message="Unexpected result shape", data=12345
        )

        with pytest.raises(ValueError, match="unexpected result"):
            await topic_pipeline(config=test_config)

        mock_inference.assert_called_once()
        mock_aggregate.assert_not_called()


@pytest.mark.asyncio
async def test_topic_pipeline_with_inference_fault_missing_dict_data(
    test_config,
):
    """
    Test the flow raises ValueError when an inference Fault's data field
    does not contain the expected dict shape.
    """
    with (
        patch(
            "flows.topic_pipeline.inference",
            new_callable=AsyncMock,
        ) as mock_inference,
        patch(
            "flows.topic_pipeline.aggregate",
            new_callable=AsyncMock,
        ) as mock_aggregate,
    ):
        mock_inference.return_value = Failed(
            message="Some inference batches had failures!",
            data=Fault(
                msg="Some inference batches had failures!",
                loggable_data={},
                data="not a dict",
            ),
        )

        with pytest.raises(
            ValueError, match="Expected data field of the Fault to contain a dict"
        ):
            await topic_pipeline(config=test_config)

        mock_inference.assert_called_once()
        mock_aggregate.assert_not_called()


@pytest.mark.asyncio
async def test_topic_pipeline_raises_on_aggregation_failure_despite_partial_inference_failure(
    test_config,
    mock_run_output_identifier_str,
):
    """
    Test that the pipeline raises when aggregation fails, even when inference
    also had partial (non-fatal) failures. Aggregation no longer has explicit
    Fault handling - `aggregation_run.result()` is called with the default
    `raise_on_failure=True`, so a Fault/Exception stored as a Completed
    state's data raises automatically. The summary artifact is created
    before `aggregate` runs, so it should still exist despite the failure.
    """
    # Mock the sub-flows
    with (
        patch(
            "flows.topic_pipeline.inference",
            new_callable=AsyncMock,
        ) as mock_inference,
        patch(
            "flows.topic_pipeline.aggregate",
            new_callable=AsyncMock,
        ) as mock_aggregate,
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
                loggable_data={},
                data={
                    "successful_document_stems": set(document_stems_successful),
                    "run_output_identifier": mock_run_output_identifier_str,
                },
            ),
        )

        # aggregation state contains failed documents - stored as a Fault on
        # an otherwise COMPLETED state, matching the codebase's convention.
        mock_aggregate.return_value = State(
            type=StateType.COMPLETED,
            data=Fault(
                msg="1/2 Documents failed",
                loggable_data={},
                data=AggregateResult(
                    run_output_identifier=mock_run_output_identifier_str,
                    errors="1/2 Documents failed",
                ),
            ),
        )

        # aggregation_run.result() raises the Fault directly (default
        # raise_on_failure=True), so this surfaces before the deferred
        # inference Fault check at the end of the flow is ever reached.
        with pytest.raises(
            Fault,
            match="1/2 Documents failed",
        ):
            await topic_pipeline(
                config=test_config,
                classifier_specs=[classifier_spec],
                document_ids=[
                    DocumentImportId("test.doc.1"),
                    DocumentImportId("test.doc.2"),
                ],
                inference_batch_size=500,
                inference_cpu_concurrency_limit=5,
                inference_gpu_concurrency_limit=5,
                aggregation_n_documents_in_batch=50,
                aggregation_n_batches=3,
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
        assert call_args.kwargs["classifier_cpu_concurrency_limit"] == 5
        assert call_args.kwargs["classifier_gpu_concurrency_limit"] == 5

        mock_aggregate.assert_called_once()
        call_args = mock_aggregate.call_args
        assert (
            call_args.kwargs["run_output_identifier"] == mock_run_output_identifier_str
        )
        assert call_args.kwargs["config"] == test_config
        assert call_args.kwargs["n_documents_in_batch"] == 50
        assert call_args.kwargs["n_batches"] == 3

        # The summary artifact is created right after the inference match,
        # before aggregate() is even called - so it should exist despite
        # the subsequent aggregation failure.
        summary_artifact = await Artifact.get("topic-pipeline-results-summary-sandbox")
        assert summary_artifact and summary_artifact.description
        assert (
            summary_artifact.description
            == "Summary of the topic pipeline successful run."
        )

        assert mock_inference.call_count == 1
        assert mock_aggregate.call_count == 1


@pytest.mark.asyncio
async def test_topic_pipeline_raises_on_aggregation_subflow_crash(
    test_config,
    mock_run_output_identifier_str,
):
    """
    Test that a genuine aggregation subflow crash (not a partial-failure
    Fault, but the subflow itself failing) propagates out of
    aggregation_run.result() as a FailedRun, distinct from the
    Fault-wrapped partial-failure case above.
    """
    with (
        patch(
            "flows.topic_pipeline.inference",
            new_callable=AsyncMock,
        ) as mock_inference,
        patch(
            "flows.topic_pipeline.aggregate",
            new_callable=AsyncMock,
        ) as mock_aggregate,
    ):
        mock_inference.return_value = Completed(
            message="Successfully ran inference on all batches!",
            data=mock_run_output_identifier_str,
        )
        mock_aggregate.return_value = Failed(
            message="Aggregation crashed",
            result=Exception("Test aggregation exception"),
        )

        with pytest.raises(FailedRun, match="Aggregation crashed"):
            await topic_pipeline(config=test_config)

        mock_inference.assert_called_once()
        mock_aggregate.assert_called_once()

        # The summary artifact should still have been created, since it is
        # created before aggregate() runs.
        summary_artifact = await Artifact.get("topic-pipeline-results-summary-sandbox")
        assert summary_artifact and summary_artifact.description


@pytest.mark.asyncio
async def test_topic_pipeline_summary_artifact_created_before_aggregate(
    test_config,
    mock_run_output_identifier_str,
):
    """
    Test the ordering guarantee that the summary artifact is created before
    aggregate() is invoked. This ordering is what makes the artifact
    survive an aggregation failure in the tests above, so it is worth
    locking down directly rather than only inferring it indirectly.
    """
    call_order = []

    with (
        patch(
            "flows.topic_pipeline.inference",
            new_callable=AsyncMock,
        ) as mock_inference,
        patch(
            "flows.topic_pipeline.aggregate",
            new_callable=AsyncMock,
        ) as mock_aggregate,
        patch(
            "flows.topic_pipeline.create_topic_pipeline_summary_artifact",
            new_callable=AsyncMock,
        ) as mock_create_artifact,
    ):
        mock_inference.return_value = Completed(
            message="Successfully ran inference on all batches!",
            data=mock_run_output_identifier_str,
        )

        async def record_artifact_call(*args, **kwargs):
            call_order.append("create_summary_artifact")

        async def record_aggregate_call(*args, **kwargs):
            call_order.append("aggregate")
            return State(
                type=StateType.COMPLETED,
                data=AggregateResult(
                    run_output_identifier=mock_run_output_identifier_str,
                    errors=None,
                ),
            )

        mock_create_artifact.side_effect = record_artifact_call
        mock_aggregate.side_effect = record_aggregate_call

        await topic_pipeline(config=test_config)

        assert call_order == ["create_summary_artifact", "aggregate"]


@pytest.mark.asyncio
async def test_create_topic_pipeline_summary_artifact_content(test_config):
    """
    Unit test for create_topic_pipeline_summary_artifact directly, asserting
    on the markdown body content (environment name and successful document
    count), not just the artifact's static description string.
    """
    from flows.topic_pipeline import create_topic_pipeline_summary_artifact

    successful_document_stems = {
        DocumentStem("CCLW.executive.1.1"),
        DocumentStem("CCLW.executive.2.2"),
        DocumentStem("CCLW.executive.3.3"),
    }

    await create_topic_pipeline_summary_artifact(
        config=test_config,
        successful_document_stems=successful_document_stems,
    )

    summary_artifact = await Artifact.get(
        f"topic-pipeline-results-summary-{test_config.aws_env.value}"
    )
    assert summary_artifact is not None
    assert summary_artifact.data is not None
    markdown = summary_artifact.data
    assert test_config.aws_env.value in markdown
    assert str(len(successful_document_stems)) in markdown


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
            "flows.topic_pipeline.inference",
            new_callable=AsyncMock,
        ) as mock_inference,
        patch(
            "flows.topic_pipeline.aggregate",
            new_callable=AsyncMock,
        ) as mock_aggregate,
        patch(
            "flows.topic_pipeline.get_async_session",
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

        # Run the flow with document_ids_s3_path
        await topic_pipeline(
            config=test_config,
            classifier_specs=[classifier_spec],
            document_ids_s3_path=s3_path,
            inference_batch_size=500,
            inference_cpu_concurrency_limit=5,
            inference_gpu_concurrency_limit=5,
            aggregation_n_documents_in_batch=50,
            aggregation_n_batches=3,
        )

        # Verify sub-flows were called with correct parameters
        mock_inference.assert_called_once()
        call_args = mock_inference.call_args
        assert call_args.kwargs["classifier_specs"] == [classifier_spec]
        assert call_args.kwargs["document_ids_s3_path"] == s3_path
        assert call_args.kwargs["config"] == test_config
        assert call_args.kwargs["batch_size"] == 500
        assert call_args.kwargs["classifier_cpu_concurrency_limit"] == 5
        assert call_args.kwargs["classifier_gpu_concurrency_limit"] == 5

        mock_aggregate.assert_called_once()
