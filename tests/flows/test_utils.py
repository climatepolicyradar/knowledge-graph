import os
from io import BytesIO
from pathlib import Path

import boto3
import pytest

from flows.utils import (
    SlackNotify,
    file_name_from_path,
    get_file_stems_for_document_id,
    remove_translated_suffix,
    s3_file_exists,
)


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


def test_message(mock_prefect_slack_webhook, mock_flow, mock_flow_run):
    SlackNotify.message(mock_flow, mock_flow_run, mock_flow_run.state)
    mock_SlackWebhook, mock_prefect_slack_block = mock_prefect_slack_webhook

    # `.load`
    mock_SlackWebhook.load.assert_called_once_with(
        "slack-webhook-platform-prefect-mvp-sandbox"
    )

    # `.notify`
    mock_prefect_slack_block.notify.assert_called_once()
    kwargs = mock_prefect_slack_block.notify.call_args.kwargs
    message = kwargs.get("body", "")
    assert message == (
        "Flow run TestFlow/TestFlowRun observed in state `Completed` at "
        "2025-01-28T12:00:00+00:00. For environment: sandbox. Flow run URL: "
        "None/flow-runs/flow-run/test-flow-run-id. State message: message"
    )


@pytest.mark.parametrize(
    "file_name, expected",
    [
        ("CCLW.executive.1.1_en_translated", "CCLW.executive.1.1"),
        ("CCLW.executive.1.1", "CCLW.executive.1.1"),
        ("CCLW.executive.10083.rtl_190_translated_en", "CCLW.executive.10083.rtl_190"),
        ("CCLW.executive.10083.rtl_190_translated_fr", "CCLW.executive.10083.rtl_190"),
    ],
)
def test_remove_translated_suffix(file_name: str, expected: str) -> None:
    """Test that we can remove the translated suffix from a file name."""

    assert remove_translated_suffix(file_name) == expected


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
        document_id,
        test_config.cache_bucket,
        test_config.document_source_prefix,
    )

    assert file_stems == [document_id, f"{document_id}_translated_en"]
