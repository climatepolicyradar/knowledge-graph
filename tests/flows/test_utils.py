import asyncio
import os
import re
import time
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import boto3
import pytest
from prefect.flows import flow

from flows.utils import (
    DocumentStem,
    SlackNotify,
    collect_unique_file_stems_under_prefix,
    file_name_from_path,
    filter_non_english_language_file_stems,
    fn_is_async,
    gather_and_report,
    get_file_stems_for_document_id,
    get_labelled_passage_paths,
    iterate_batch,
    remove_translated_suffix,
    s3_file_exists,
)
from scripts.cloud import ClassifierSpec


@pytest.mark.parametrize(
    "path, expected",
    [
        ("Q1.json", "Q1"),
        ("test/Q2.json", "Q2"),
        ("test/test/test/Q3.json", "Q3"),
    ],
)
def test_file_name_from_path(path, expected):
    assert file_name_from_path(path) == expected


@pytest.mark.asyncio
async def test_message(mock_prefect_slack_webhook, mock_flow, mock_flow_run):
    with (
        patch.object(SlackNotify, "environment", "prod"),
        patch.object(
            SlackNotify, "slack_block_name", "slack-webhook-platform-prefect-mvp-prod"
        ),
    ):
        await SlackNotify.message(mock_flow, mock_flow_run, mock_flow_run.state)

    mock_SlackWebhook, mock_prefect_slack_block = mock_prefect_slack_webhook

    # `.load`
    mock_SlackWebhook.load.assert_called_once_with(
        "slack-webhook-platform-prefect-mvp-prod"
    )

    # `.notify`
    mock_prefect_slack_block.notify.assert_called_once()
    kwargs = mock_prefect_slack_block.notify.call_args.kwargs
    message = kwargs.get("body", "")
    assert re.match(
        r"Flow run TestFlow/TestFlowRun observed in state `Completed` at 2025-01-28T12:00:00\+00:00\. For environment: prod\. Flow run URL: http://127\.0\.0\.1:\d+/flow-runs/flow-run/test-flow-run-id\. State message: message",
        message,
    )


@pytest.mark.parametrize(
    "file_name, expected",
    [
        ("CCLW.executive.1.1_translated_en", "CCLW.executive.1.1"),
        ("CCLW.executive.1.1", "CCLW.executive.1.1"),
        ("CCLW.executive.10083.rtl_190_translated_en", "CCLW.executive.10083.rtl_190"),
        ("CCLW.executive.10083.rtl_190_translated_fr", "CCLW.executive.10083.rtl_190"),
    ],
)
def test_remove_translated_suffix(file_name: str, expected: str) -> None:
    """Test that we can remove the translated suffix from a file name."""

    assert remove_translated_suffix(file_name) == expected


@pytest.mark.parametrize(
    "data, expected_lengths",
    [
        # Lists
        (list(range(50)), [50]),
        (list(range(850)), [400, 400, 50]),
        ([], [0]),
        # Generators
        ((x for x in range(50)), [50]),
        ((x for x in range(850)), [400, 400, 50]),
        ((x for x in []), [0]),
    ],
)
def test_iterate_batch(data, expected_lengths):
    for batch, expected in zip(list(iterate_batch(data, 400)), expected_lengths):
        assert len(batch) == expected


def test_s3_file_exists(test_config, mock_bucket_documents) -> None:
    """Test that we can check if a file exists in an S3 bucket."""

    key = os.path.join(test_config.document_source_prefix, "PDF.document.0.1.json")

    s3_file_exists(test_config.cache_bucket, key)

    assert not s3_file_exists(test_config.cache_bucket, "non_existent_key")


def test_get_file_stems_for_document_id(test_config, mock_bucket_documents) -> None:
    """Test that we can get the file stems for a document ID."""

    document_id = Path(mock_bucket_documents[0]).stem

    file_stems = get_file_stems_for_document_id(
        document_id,
        test_config.cache_bucket,
        test_config.document_source_prefix,
    )

    assert file_stems == [document_id]

    body = BytesIO('{"some_key": "some_value"}'.encode("utf-8"))
    key = os.path.join(
        test_config.document_source_prefix, f"{document_id}_translated_en.json"
    )
    s3_client = boto3.client("s3")

    s3_client.put_object(
        Bucket=test_config.cache_bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
    )

    file_stems = get_file_stems_for_document_id(
        document_id=document_id,
        bucket_name=test_config.cache_bucket,
        document_key=os.path.join(
            test_config.document_source_prefix, f"{document_id}.json"
        ),
    )

    assert file_stems == [f"{document_id}_translated_en"]


def test_collect_file_stems_under_prefix(test_config, mock_bucket) -> None:
    """Test that we can collect file stems under a prefix."""

    s3_paths = [
        "test_prefix/Q1/v1/CCLW.executive.1.1.json",
        "test_prefix/Q1/v1/CCLW.executive.2.2.json",
        "test_prefix/Q1/v1/CCLW.executive.2.2_translated_en.json",
        "test_prefix/Q1/v2/CCLW.executive.1.1.json",
        "test_prefix/Q1/v2/CCLW.executive.2.2.json",
        "test_prefix/Q2/v1/CCLW.executive.1.1.json",
        "test_prefix/Q2/v1/CCLW.executive.2.2.json",
        "test_prefix/Q3/v2/CCLW.executive.1.1.json",
        "test_prefix/Q3/v2/CCLW.executive.2.2.json",
        "test_prefix/Q3/v2/CCLW.executive.3.3.json",
        "some_other_prefix/Q1/v1/CCLW.some_other_doc.4.4.json",
    ]
    s3_client = boto3.client("s3")
    for s3_path in s3_paths:
        s3_client.put_object(Bucket=test_config.cache_bucket, Key=s3_path)

    file_stems = collect_unique_file_stems_under_prefix(
        bucket_name=test_config.cache_bucket,
        prefix="test_prefix",
    )

    assert set(file_stems) == set(
        [
            DocumentStem("CCLW.executive.1.1"),
            DocumentStem("CCLW.executive.2.2"),
            DocumentStem("CCLW.executive.2.2_translated_en"),
            DocumentStem("CCLW.executive.3.3"),
        ]
    )


def test_get_labelled_passage_paths(test_config, mock_s3_client, mock_bucket) -> None:
    """Test that we can get all document paths from a list of document IDs."""

    classifier_spec = ClassifierSpec(name="Q123", alias="v1")
    body = BytesIO('{"some_key": "some_value"}'.encode("utf-8"))
    s3_file_names = [
        "CCLW.executive.1.1_translated_en.json",
        "CCLW.executive.1.1.json",
        "CCLW.executive.10083.rtl_190.json",
    ]

    s3_client = boto3.client("s3")
    for file_name in s3_file_names:
        s3_client.put_object(
            Bucket=test_config.cache_bucket,
            Key=os.path.join(
                test_config.document_target_prefix,
                classifier_spec.name,
                classifier_spec.alias,
                file_name,
            ),
            Body=body,
            ContentType="application/json",
        )

    # Get all the document paths for classifiers and documents ids.
    document_paths = get_labelled_passage_paths(
        document_ids=["CCLW.executive.1.1", "CCLW.executive.10083.rtl_190"],
        classifier_specs=[classifier_spec],
        cache_bucket=test_config.cache_bucket,
        labelled_passages_prefix=test_config.document_target_prefix,
    )
    assert sorted(document_paths) == sorted(
        [
            f"s3://{test_config.cache_bucket}/{test_config.document_target_prefix}/{classifier_spec.name}/{classifier_spec.alias}/CCLW.executive.1.1_translated_en.json",
            f"s3://{test_config.cache_bucket}/{test_config.document_target_prefix}/{classifier_spec.name}/{classifier_spec.alias}/CCLW.executive.10083.rtl_190.json",
        ]
    )


def test_filter_non_english_file_stems() -> None:
    """Test that we can successfully filter out non-English file stems."""

    # Test we can filter out non-English file stems
    file_stems = [
        "AF.document.002MMUCR.n0000",
        "AF.document.AFRDG00038.n0000_translated_en",
        "AF.document.AFRDG00038.n0000",
        "CCLW.document.i00001313.n0000",
        "CCLW.executive.10491.5394",
        "CCLW.executive.rtl_98.rtl_338",
        "CCLW.executive.rtl_98.rtl_338_translated_en",
        "CCLW.legislative.1046.0",
        "CPR.document.i00000918.n0000",
        "GCF.document.FP018_13290.6302",
        "GEF.document.2480.n0000_translated_en",
        "GEF.document.2480.n0000",
        "OEP.document.i00000157.n0000",
        "UNFCCC.document.i00000546.n0000",
        "UNFCCC.non-party.1243.0",
    ]

    filtered_file_stems = filter_non_english_language_file_stems(file_stems=file_stems)

    assert filtered_file_stems == [
        "AF.document.002MMUCR.n0000",
        "AF.document.AFRDG00038.n0000_translated_en",
        "CCLW.document.i00001313.n0000",
        "CCLW.executive.10491.5394",
        "CCLW.executive.rtl_98.rtl_338_translated_en",
        "CCLW.legislative.1046.0",
        "CPR.document.i00000918.n0000",
        "GCF.document.FP018_13290.6302",
        "GEF.document.2480.n0000_translated_en",
        "OEP.document.i00000157.n0000",
        "UNFCCC.document.i00000546.n0000",
        "UNFCCC.non-party.1243.0",
    ]

    # Test that we can filter quickly even on very long lists
    file_stems = []
    for i in list(range(7_000)):
        file_stems.append(f"AF.document.{i}.n0000")
        if i % 2 == 0:
            file_stems.append(f"AF.document.{i}.n0000_translated_en")

    start_time = time.time()
    _ = filter_non_english_language_file_stems(file_stems=file_stems)
    end_time = time.time()

    assert end_time - start_time < 1, "Filtering took too long"


def test_fn_is_async():
    async def async_fn():
        return 1

    def sync_fn():
        return 1

    assert fn_is_async(async_fn)
    assert not fn_is_async(sync_fn)

    @flow
    async def async_flow():
        return 1

    @flow
    def sync_flow():
        return 1

    assert fn_is_async(async_flow)
    assert not fn_is_async(sync_flow)


@pytest.fixture
def mock_progress_artifacts():
    """Fixture to mock progress artifact functions and return their IDs."""
    with (
        patch(
            "flows.utils.create_progress_artifact", new_callable=AsyncMock
        ) as mock_create,
        patch(
            "flows.utils.update_progress_artifact", new_callable=AsyncMock
        ) as mock_update,
    ):
        mock_artifact_id = uuid4()
        mock_create.return_value = mock_artifact_id
        yield {
            "create": mock_create,
            "update": mock_update,
            "artifact_id": mock_artifact_id,
        }


@pytest.mark.asyncio
async def test_gather_and_report_calls_progress_artifacts(mock_progress_artifacts):
    """Test that gather_and_report calls create_progress_artifact and update_progress_artifact."""
    mock_create_progress_artifact = mock_progress_artifacts["create"]
    mock_update_progress_artifact = mock_progress_artifacts["update"]
    mock_artifact_id = mock_progress_artifacts["artifact_id"]

    # Create some async tasks to gather
    async def sample_task(value):
        await asyncio.sleep(0.01)  # Small delay to simulate work
        return value * 2

    tasks = [sample_task(i) for i in range(3)]

    # Call gather_and_report
    results = await gather_and_report(
        tasks=tasks,
        return_exceptions=False,
        key="test-progress-key",
        desc_create="Starting test tasks",
        desc_update_fn=lambda tasks,
        results: f"Completed {len(results)}/{len(tasks)} tasks",
    )

    # Verify results (order may vary due to asyncio.as_completed)
    assert sorted(results) == [0, 2, 4]

    # Verify create_progress_artifact was called once
    mock_create_progress_artifact.assert_called_once_with(
        progress=0.0,
        key="test-progress-key",
        description="Starting test tasks",
    )

    # Verify update_progress_artifact was called for each completed task
    assert mock_update_progress_artifact.call_count == 3

    # Check that all calls to update_progress_artifact used the correct artifact_id
    for call in mock_update_progress_artifact.call_args_list:
        _args, kwargs = call
        assert kwargs["artifact_id"] == mock_artifact_id


@pytest.mark.asyncio
async def test_gather_and_report_with_exceptions(mock_progress_artifacts):
    """Test that gather_and_report calls progress artifacts even when tasks raise exceptions."""
    mock_create_progress_artifact = mock_progress_artifacts["create"]
    mock_update_progress_artifact = mock_progress_artifacts["update"]

    # Create tasks that will raise exceptions
    async def failing_task():
        await asyncio.sleep(0.01)
        raise ValueError("Test error")

    async def success_task():
        await asyncio.sleep(0.01)
        return "success"

    tasks = [failing_task(), success_task(), failing_task()]

    # Call gather_and_report with return_exceptions=True
    results = await gather_and_report(
        tasks=tasks,
        return_exceptions=True,
        key="test-error-key",
        desc_create="Starting tasks with errors",
    )

    # Verify results contain both exceptions and successful values
    # (order may vary)
    assert len(results) == 3
    success_results = [r for r in results if r == "success"]
    error_results = [r for r in results if isinstance(r, ValueError)]
    assert len(success_results) == 1
    assert len(error_results) == 2

    # Verify create_progress_artifact was called once
    mock_create_progress_artifact.assert_called_once_with(
        progress=0.0,
        key="test-error-key",
        description="Starting tasks with errors",
    )

    # Verify update_progress_artifact was called for each task
    # (including failed ones)
    assert mock_update_progress_artifact.call_count == 3


@pytest.mark.asyncio
async def test_gather_and_report_matches_asyncio_gather(mock_progress_artifacts):
    """Test that gather_and_report produces the same results as asyncio.gather."""

    # Create test tasks with deterministic results
    async def task_add(x, y):
        await asyncio.sleep(0.001)  # Minimal delay
        return x + y

    async def task_multiply(x, y):
        await asyncio.sleep(0.001)
        return x * y

    async def task_power(x, y):
        await asyncio.sleep(0.001)
        return x**y

    tasks_for_gather = [task_add(2, 3), task_multiply(4, 5), task_power(2, 3)]
    tasks_for_gather_and_report = [
        task_add(2, 3),
        task_multiply(4, 5),
        task_power(2, 3),
    ]

    gather_results = await asyncio.gather(*tasks_for_gather, return_exceptions=False)
    gather_and_report_results = await gather_and_report(
        tasks=tasks_for_gather_and_report,
        return_exceptions=False,
        key="comparison-test",
        desc_create="Comparing with asyncio.gather",
    )

    assert sorted(gather_results) == sorted(gather_and_report_results) == [5, 8, 20]


@pytest.mark.asyncio
async def test_gather_and_report_matches_asyncio_gather_with_exceptions(
    mock_progress_artifacts,
):
    """Test that gather_and_report handles exceptions the same as asyncio.gather."""

    async def success_task(value):
        await asyncio.sleep(0.001)
        return value * 2

    async def failing_task(error_msg):
        await asyncio.sleep(0.001)
        raise ValueError(error_msg)

    tasks_for_gather = [
        success_task(10),
        failing_task("error1"),
        success_task(20),
        failing_task("error2"),
    ]
    tasks_for_gather_and_report = [
        success_task(10),
        failing_task("error1"),
        success_task(20),
        failing_task("error2"),
    ]

    gather_results = await asyncio.gather(*tasks_for_gather, return_exceptions=True)
    gather_and_report_results = await gather_and_report(
        tasks=tasks_for_gather_and_report,
        return_exceptions=True,
        key="exception-comparison-test",
        desc_create="Comparing exceptions with asyncio.gather",
    )

    assert len(gather_results) == len(gather_and_report_results) == 4

    gather_successes = [r for r in gather_results if not isinstance(r, Exception)]
    gather_exceptions = [r for r in gather_results if isinstance(r, Exception)]

    report_successes = [
        r for r in gather_and_report_results if not isinstance(r, Exception)
    ]
    report_exceptions = [
        r for r in gather_and_report_results if isinstance(r, Exception)
    ]

    assert len(gather_successes) == len(report_successes) == 2
    assert len(gather_exceptions) == len(report_exceptions) == 2

    assert sorted(gather_successes) == sorted(report_successes) == [20, 40]

    gather_error_messages = sorted([str(e) for e in gather_exceptions])
    report_error_messages = sorted([str(e) for e in report_exceptions])
    assert gather_error_messages == report_error_messages == ["error1", "error2"]
