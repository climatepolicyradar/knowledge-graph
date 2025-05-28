import os
import re
import time
from io import BytesIO
from pathlib import Path

import boto3
import pytest

from flows.inference import DocumentStem
from flows.utils import (
    S3FileStemFetcher,
    SlackNotify,
    file_name_from_path,
    filter_non_english_language_file_stems,
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


async def test_message(mock_prefect_slack_webhook, mock_flow, mock_flow_run):
    await SlackNotify.message(mock_flow, mock_flow_run, mock_flow_run.state)
    mock_SlackWebhook, mock_prefect_slack_block = mock_prefect_slack_webhook

    # `.load`
    mock_SlackWebhook.load.assert_called_once_with(
        "slack-webhook-platform-prefect-mvp-sandbox"
    )

    # `.notify`
    mock_prefect_slack_block.notify.assert_called_once()
    kwargs = mock_prefect_slack_block.notify.call_args.kwargs
    message = kwargs.get("body", "")
    assert re.match(
        r"Flow run TestFlow/TestFlowRun observed in state `Completed` at 2025-01-28T12:00:00\+00:00\. For environment: sandbox\. Flow run URL: http://127\.0\.0\.1:\d+/flow-runs/flow-run/test-flow-run-id\. State message: message",
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


def test_list_bucket_file_stems(test_config, mock_bucket_documents):
    expected_ids = [Path(d).stem for d in mock_bucket_documents]
    got_ids = S3FileStemFetcher(
        bucket_region=test_config.bucket_region,
        cache_bucket=test_config.cache_bucket,
        document_source_prefix=test_config.document_source_prefix,
        use_new_and_updated=False,
        document_ids=[],
        pipeline_state_prefix=test_config.pipeline_state_prefix,
    ).list_bucket_file_stems()
    assert sorted(expected_ids) == sorted(got_ids)


@pytest.mark.parametrize(
    ("doc_ids", "bucket_ids", "expected"),
    [
        (
            ["AF.document.002MMUCR.n0000"],
            [
                "AF.document.002MMUCR.n0000",
                "AF.document.AFRDG00038.n0000",
                "CCLW.document.i00001313.n0000",
            ],
            ["AF.document.002MMUCR.n0000"],
        ),
        (None, ["AF.document.002MMUCR.n0000"], ["AF.document.002MMUCR.n0000"]),
    ],
)
def test_determine_file_stems(test_config, doc_ids, bucket_ids, expected):
    got = S3FileStemFetcher(
        bucket_region=test_config.bucket_region,
        cache_bucket=test_config.cache_bucket,
        document_source_prefix=test_config.document_source_prefix,
        pipeline_state_prefix=test_config.pipeline_state_prefix,
        use_new_and_updated=False,
        document_ids=doc_ids,
    ).determine_file_stems(
        use_new_and_updated=False,
        requested_document_ids=doc_ids,
        current_bucket_file_stems=bucket_ids,
    )
    assert got == expected


@pytest.mark.parametrize(
    "input_stems,expected_output",
    [
        ([], []),
        (
            ["CCLW.executive.12345.6789", "UNFCCC.document.1234.5678"],
            ["CCLW.executive.12345.6789", "UNFCCC.document.1234.5678"],
        ),
        (["Sabin.document.16944.17490", "Sabin.document.16945.17491"], []),
        (
            [
                "CCLW.executive.12345.6789",
                "Sabin.document.16944.17490",
                "UNFCCC.document.1234.5678",
                "Sabin.document.16945.17491",
            ],
            ["CCLW.executive.12345.6789", "UNFCCC.document.1234.5678"],
        ),
        (["sabin.document.16944.17490", "SABIN.document.16945.17491"], []),
        (
            ["SabinIndustries.document.1234.5678", "DocumentSabin.12345.6789"],
            ["DocumentSabin.12345.6789"],
        ),
    ],
)
def test_remove_sabin_file_stems(
    test_config, input_stems: list[DocumentStem], expected_output: list[DocumentStem]
):
    result = S3FileStemFetcher(
        bucket_region=test_config.bucket_region,
        cache_bucket=test_config.cache_bucket,
        document_source_prefix=test_config.document_source_prefix,
        pipeline_state_prefix=test_config.pipeline_state_prefix,
        use_new_and_updated=False,
        document_ids=[],
    ).remove_sabin_file_stems(input_stems)
    assert result == expected_output


def test_determine_file_stems__error(test_config):
    with pytest.raises(ValueError):
        S3FileStemFetcher(
            bucket_region=test_config.bucket_region,
            cache_bucket=test_config.cache_bucket,
            document_source_prefix=test_config.document_source_prefix,
            pipeline_state_prefix=test_config.pipeline_state_prefix,
            use_new_and_updated=False,
            document_ids=[],
        ).determine_file_stems(
            use_new_and_updated=False,
            requested_document_ids=[
                "AF.document.002MMUCR.n0000",
                "AF.document.AFRDG00038.n00002",
            ],
            current_bucket_file_stems=[
                "CCLW.document.i00001313.n0000",
                "AF.document.002MMUCR.n0000",
            ],
        )


def test_get_latest_ingest_documents(
    test_config, mock_bucket_new_and_updated_documents_json
):
    _, latest_docs = mock_bucket_new_and_updated_documents_json
    doc_ids = S3FileStemFetcher(
        bucket_region=test_config.bucket_region,
        cache_bucket=test_config.cache_bucket,
        document_source_prefix=test_config.document_source_prefix,
        pipeline_state_prefix=test_config.pipeline_state_prefix,
        use_new_and_updated=False,
        document_ids=[],
    ).get_latest_ingest_documents()
    assert set(doc_ids) == latest_docs


def test_get_latest_ingest_documents_no_latest(
    test_config,
    # Setup the empty bucket
    mock_bucket,
):
    with pytest.raises(
        ValueError,
        match="failed to find",
    ):
        S3FileStemFetcher(
            bucket_region=test_config.bucket_region,
            cache_bucket=test_config.cache_bucket,
            document_source_prefix=test_config.document_source_prefix,
            pipeline_state_prefix=test_config.pipeline_state_prefix,
            use_new_and_updated=False,
            document_ids=[],
        ).get_latest_ingest_documents()


def test_fetch__specific_document(test_config, mock_bucket_documents):
    # Specific document
    fetcher = S3FileStemFetcher(
        bucket_region=test_config.bucket_region,
        cache_bucket=test_config.cache_bucket,
        document_source_prefix=test_config.document_source_prefix,
        pipeline_state_prefix=test_config.pipeline_state_prefix,
        use_new_and_updated=False,
        document_ids=["HTML.document.0.1"],
    )
    assert fetcher.fetch() == ["HTML.document.0.1"]


def test_fetch__latest_documents(
    test_config, mock_bucket_documents, mock_bucket_new_and_updated_documents_json
):
    _, latest_docs = mock_bucket_new_and_updated_documents_json
    # Latest documents
    fetcher = S3FileStemFetcher(
        bucket_region=test_config.bucket_region,
        cache_bucket=test_config.cache_bucket,
        document_source_prefix=test_config.document_source_prefix,
        pipeline_state_prefix=test_config.pipeline_state_prefix,
        use_new_and_updated=True,
        document_ids=[],
    )
    assert set(fetcher.fetch()) == set(latest_docs)
