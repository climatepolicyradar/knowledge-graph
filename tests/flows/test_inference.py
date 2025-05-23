import json
from datetime import datetime
from pathlib import Path

import boto3
import pytest
from botocore.client import ClientError
from cpr_sdk.parser_models import BlockType
from prefect.testing.utilities import prefect_test_harness

from flows.inference import (
    ClassifierSpec,
    DocumentStem,
    _stringify,
    classifier_inference,
    determine_file_stems,
    document_passages,
    download_classifier_from_wandb_to_local,
    get_latest_ingest_documents,
    iterate_batch,
    list_bucket_file_stems,
    load_classifier,
    load_document,
    remove_sabin_file_stems,
    run_classifier_inference_on_document,
    store_labels,
    text_block_inference,
)
from src.labelled_passage import LabelledPassage
from src.span import Span


def helper_list_labels_in_bucket(test_config, bucket_name):
    # Find out what is now in the spans bucket
    s3 = boto3.client("s3", region_name=test_config.bucket_region)
    response = s3.list_objects_v2(
        Bucket=bucket_name, Prefix=test_config.document_target_prefix
    )
    labels = [c.get("Key") for c in response.get("Contents", [])]
    return labels


def test_list_bucket_file_stems(test_config, mock_bucket_documents):
    expected_ids = [Path(d).stem for d in mock_bucket_documents]
    got_ids = list_bucket_file_stems(test_config)
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
    got = determine_file_stems(
        config=test_config,
        use_new_and_updated=False,
        requested_document_ids=doc_ids,
        current_bucket_file_stems=bucket_ids,
    )
    assert got == expected


def test_determine_file_stems__error(test_config):
    with pytest.raises(ValueError):
        determine_file_stems(
            config=test_config,
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


@pytest.mark.asyncio
async def test_load_classifier__existing_classifier(
    mock_wandb, test_config, mock_classifiers_dir, local_classifier_id
):
    _, mock_run, _ = mock_wandb
    classifier = await load_classifier(
        mock_run, test_config, local_classifier_id, alias="latest"
    )
    assert local_classifier_id == classifier.concept.wikibase_id


def test_download_classifier_from_wandb_to_local(mock_wandb, test_config):
    _, mock_run, _ = mock_wandb
    classifier_id = "Qtest"
    _ = download_classifier_from_wandb_to_local(
        mock_run, test_config, classifier_id, alias="latest"
    )


def test_load_document(test_config, mock_bucket_documents):
    for doc_file_name in mock_bucket_documents:
        file_stem = Path(doc_file_name).stem
        doc = load_document(test_config, file_stem=file_stem)
        assert file_stem == doc.document_id


def test_stringify():
    text = ["a", " sequence", " of ", "text "]
    result = _stringify(text)
    assert result == "a sequence of text"


def test_document_passages__blocked_types(parser_output_pdf):
    # Add a page number block that should be filtered out
    from cpr_sdk.parser_models import TextBlock

    parser_output_pdf.pdf_data.text_blocks.append(
        TextBlock(
            text=["Page 1"],
            text_block_id="page_1",
            type=BlockType.PAGE_NUMBER,
            type_confidence=0.5,
        )
    )

    # Get all passages
    results = list(document_passages(parser_output_pdf))

    # Should only get the non-page-number block
    assert len(results) == 1
    assert results[0] == ("test pdf text", "2")
    # Verify the page number block was filtered out
    assert not any(block_id == "page_1" for _, block_id in results)


def test_document_passages__invalid_content_type(parser_output):
    # When the content type is none, empty list
    parser_output.document_content_type = None
    result = [i for i in document_passages(parser_output)]
    assert result == []


def test_document_passages__html(parser_output_html):
    html_result = document_passages(parser_output_html).__next__()
    assert html_result == ("test html text", "1")


def test_document_passages__pdf(parser_output_pdf):
    pdf_result = document_passages(parser_output_pdf).__next__()
    assert pdf_result == ("test pdf text", "2")


def test_store_labels(test_config, mock_bucket):
    text = "This is a test text block"
    spans = [Span(text=text, start_index=15, end_index=19)]
    labels = [LabelledPassage(text=text, spans=spans)]

    store_labels(test_config, labels, "TEST.DOC.0.1", "Q9081", "latest")

    labels = helper_list_labels_in_bucket(test_config, mock_bucket)

    assert len(labels) == 1
    assert labels[0] == "labelled_passages/Q9081/latest/TEST.DOC.0.1.json"


@pytest.mark.asyncio
async def test_text_block_inference_with_results(
    mock_wandb, test_config, mock_classifiers_dir, local_classifier_id
):
    _, mock_run, _ = mock_wandb
    test_config.local_classifier_dir = mock_classifiers_dir
    classifier = await load_classifier(
        mock_run, test_config, local_classifier_id, "latest"
    )

    text = "I love fishing. Aquaculture is the best."
    block_id = "fish_block"
    labels = text_block_inference(classifier=classifier, block_id=block_id, text=text)

    assert len(labels.spans) > 0
    assert labels.id == block_id
    assert labels.metadata != {}
    # Set the labelled passages as empty as we are removing them.
    expected_concept_metadata = classifier.concept.model_dump()
    expected_concept_metadata["labelled_passages"] = []
    assert labels.metadata["concept"] == expected_concept_metadata
    # check whether the timestamps are valid
    for span in labels.spans:
        assert isinstance(span.timestamps[0], datetime)


@pytest.mark.asyncio
async def test_text_block_inference_without_results(
    mock_wandb, test_config, mock_classifiers_dir, local_classifier_id
):
    _, mock_run, _ = mock_wandb
    test_config.local_classifier_dir = mock_classifiers_dir
    classifier = await load_classifier(
        mock_run, test_config, local_classifier_id, "latest"
    )

    text = "Rockets are cool. We should build more rockets."
    block_id = "fish_block"
    labels = text_block_inference(classifier=classifier, block_id=block_id, text=text)

    assert len(labels.spans) == 0
    assert labels.id == block_id
    assert labels.metadata == {}


@pytest.mark.asyncio
@pytest.mark.flaky_on_ci
async def test_classifier_inference(
    test_config, mock_classifiers_dir, mock_wandb, mock_bucket, mock_bucket_documents
):
    mock_wandb_init, _, _ = mock_wandb
    doc_ids = [Path(doc_file).stem for doc_file in mock_bucket_documents]
    with prefect_test_harness():
        await classifier_inference(
            classifier_specs=[ClassifierSpec(name="Q788", alias="latest")],
            document_ids=doc_ids,
            config=test_config,
        )

    mock_wandb_init.assert_called_once_with(
        entity="test_entity",
        job_type="concept_inference",
    )

    labels = helper_list_labels_in_bucket(test_config, mock_bucket)

    assert sorted(labels) == [
        "labelled_passages/Q788/latest/HTML.document.0.1.json",
        "labelled_passages/Q788/latest/PDF.document.0.1.json",
    ]

    for key in labels:
        s3 = boto3.client("s3", region_name=test_config.bucket_region)
        response = s3.get_object(Bucket=test_config.cache_bucket, Key=key)
        data = json.loads(response["Body"].read().decode("utf-8"))

        # Some spans where identified
        with_spans = [d for d in data if len(d["spans"]) > 0]
        assert len(with_spans) > 0


def test_get_latest_ingest_documents(
    test_config, mock_bucket_new_and_updated_documents_json
):
    _, latest_docs = mock_bucket_new_and_updated_documents_json
    doc_ids = get_latest_ingest_documents(test_config)
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
        get_latest_ingest_documents(test_config)


@pytest.mark.asyncio
async def test_run_classifier_inference_on_document(
    test_config, mock_classifiers_dir, mock_wandb, mock_bucket, mock_bucket_documents
):
    # Setup
    _, mock_run, _ = mock_wandb
    test_config.local_classifier_dir = mock_classifiers_dir
    classifier_name = "Q788"
    classifier_alias = "latest"

    # Load classifier
    classifier = await load_classifier(
        mock_run, test_config, classifier_name, classifier_alias
    )

    # Run the function on a document with no language
    document_id = Path(mock_bucket_documents[1]).stem
    with pytest.raises(ValueError) as exc_info:
        result = await run_classifier_inference_on_document(
            config=test_config,
            file_stem=document_id,
            classifier_name=classifier_name,
            classifier_alias=classifier_alias,
            classifier=classifier,
        )

    assert "Cannot run inference on" in str(exc_info.value)

    # Run the function on a document with english language
    document_id = Path(mock_bucket_documents[0]).stem
    result = await run_classifier_inference_on_document(
        config=test_config,
        file_stem=document_id,
        classifier_name=classifier_name,
        classifier_alias=classifier_alias,
        classifier=classifier,
    )

    assert result is None

    # Verify that labels were stored in S3
    labels = helper_list_labels_in_bucket(test_config, mock_bucket)
    expected_key = (
        f"labelled_passages/{classifier_name}/{classifier_alias}/{document_id}.json"
    )
    assert expected_key in labels

    # Verify the content of the stored labels
    s3 = boto3.client("s3", region_name=test_config.bucket_region)
    response = s3.get_object(Bucket=test_config.cache_bucket, Key=expected_key)
    data = json.loads(response["Body"].read().decode("utf-8"))

    # Verify we have at least one label
    assert len(data) > 0

    # Verify the structure of the labels
    for label in data:
        assert "id" in label
        assert "text" in label
        assert "spans" in label


@pytest.mark.asyncio
async def test_run_classifier_inference_on_document_missing(
    test_config,
    mock_classifiers_dir,
    mock_wandb,
    mock_bucket,
):
    # Setup
    _, mock_run, _ = mock_wandb
    test_config.local_classifier_dir = mock_classifiers_dir
    classifier_name = "Q788"
    classifier_alias = "latest"

    # Load classifier
    classifier = await load_classifier(
        mock_run, test_config, classifier_name, classifier_alias
    )

    document_id = "CCLW.executive.8133.0"
    with pytest.raises(ClientError) as excinfo:
        await run_classifier_inference_on_document(
            config=test_config,
            file_stem=document_id,
            classifier_name=classifier_name,
            classifier_alias=classifier_alias,
            classifier=classifier,
        )
    assert excinfo.value.response["Error"]["Code"] == "NoSuchKey"


@pytest.mark.parametrize(
    "data, expected_lengths",
    [
        (list(range(50)), [50]),
        (list(range(850)), [400, 400, 50]),
        ([], [0]),
    ],
)
def test_iterate_batch(data, expected_lengths):
    for batch, expected in zip(list(iterate_batch(data)), expected_lengths):
        assert len(batch) == expected


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
    input_stems: list[DocumentStem], expected_output: list[DocumentStem]
):
    result = remove_sabin_file_stems(input_stems)
    assert result == expected_output
