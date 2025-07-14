import os
import re
import time
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID

import boto3
import pytest
from prefect.client.schemas.objects import FlowRun, State, StateType

from flows.utils import (
    DocumentStem,
    SlackNotify,
    collect_unique_file_stems_under_prefix,
    file_name_from_path,
    filter_non_english_language_file_stems,
    get_deployment_results,
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


@pytest.mark.asyncio
async def test_get_deployment_results():
    """Test that get_deployment_results correctly retrieves and processes flow run results."""

    # Create mock flow runs
    flow_run_1 = FlowRun(
        id=UUID("12345678-1234-5678-1234-567812345678"),
        name="test-flow-1",
        flow_id=UUID("87654321-4321-8765-4321-876543210987"),
        state=State(type=StateType.COMPLETED),
    )

    flow_run_2 = FlowRun(
        id=UUID("23456789-2345-6789-2345-678923456789"),
        name="test-flow-2",
        flow_id=UUID("87654321-4321-8765-4321-876543210987"),
        state=State(type=StateType.COMPLETED),
    )

    flow_runs = [flow_run_1, flow_run_2]

    # Mock data that would be returned by the flow run results
    mock_result_1 = {
        "successful_document_stems": ["doc1", "doc2"],
        "failed_document_stems": [],
        "classifier_name": "Q123",
        "classifier_alias": "v1",
    }

    mock_result_2 = {
        "successful_document_stems": ["doc3"],
        "failed_document_stems": [("doc4", "error")],
        "classifier_name": "Q456",
        "classifier_alias": "v2",
    }

    # Create mock completed states and set them on the flow runs
    mock_state_1 = MagicMock(spec=State)
    mock_state_1.type = StateType.COMPLETED
    mock_state_1.result = AsyncMock(return_value=mock_result_1)
    flow_run_1.state = mock_state_1

    mock_state_2 = MagicMock(spec=State)
    mock_state_2.type = StateType.COMPLETED
    mock_state_2.result = AsyncMock(return_value=mock_result_2)
    flow_run_2.state = mock_state_2

    results = await get_deployment_results(flow_runs)

    # Verify the results
    assert len(results) == 2

    # Check first result
    assert results[0][0] == flow_run_1
    assert results[0][1] == mock_result_1

    # Check second result
    assert results[1][0] == flow_run_2
    assert results[1][1] == mock_result_2

    # Assert that we handle returned exceptions.
    mock_result_2 = ValueError("Test error!")
    mock_state_2.result = AsyncMock(return_value=mock_result_2)

    results = await get_deployment_results(flow_runs)
    assert isinstance(results[1][1], Exception)

    # No completed state
    mock_state_2.type = StateType.FAILED
    with pytest.raises(ValueError) as e:
        _ = await get_deployment_results(flow_runs)
    assert "Expected COMPLETED flow run state" in str(e.value)
