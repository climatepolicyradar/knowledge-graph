import json
from datetime import datetime
from pathlib import Path

import boto3
import pytest
from prefect.testing.utilities import prefect_test_harness

from flows.inference import (
    ClassifierSpec,
    _stringify,
    classifier_inference,
    determine_document_ids,
    document_passages,
    download_classifier_from_wandb_to_local,
    get_latest_ingest_documents,
    list_bucket_doc_ids,
    load_classifier,
    load_document,
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


def test_list_bucket_doc_ids(test_config, mock_bucket_documents):
    expected_ids = [Path(d).stem for d in mock_bucket_documents]
    got_ids = list_bucket_doc_ids(test_config)
    assert sorted(expected_ids) == sorted(got_ids)


@pytest.mark.parametrize(
    ("doc_ids", "bucket_ids", "expected"),
    [
        (["1"], ["1", "2", "3"], ["1"]),
        (None, ["1"], ["1"]),
    ],
)
def test_determine_document_ids(test_config, doc_ids, bucket_ids, expected):
    got = determine_document_ids(
        config=test_config,
        use_new_and_updated=False,
        requested_document_ids=doc_ids,
        current_bucket_ids=bucket_ids,
    )
    assert got == expected


def test_determine_document_ids__error(test_config):
    with pytest.raises(ValueError):
        determine_document_ids(
            config=test_config,
            use_new_and_updated=False,
            requested_document_ids=["1", "2"],
            current_bucket_ids=["3", "4"],
        )


@pytest.mark.asyncio
async def test_load_classifier__existing_classifier(
    mock_wandb, test_config, mock_classifiers_dir, local_classifier_id
):
    _, mock_run, _ = mock_wandb
    classifier = await load_classifier.fn(
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
        doc_id = Path(doc_file_name).stem
        doc = load_document(test_config, document_id=doc_id)
        assert doc_id == doc.document_id


def test_stringify():
    text = ["a", " sequence", " of ", "text "]
    result = _stringify(text)
    assert result == "a sequence of text"


def test_document_passages__invalid_content_type(parser_output):
    # When the content type is borked
    parser_output.document_content_type = None
    with pytest.raises(ValueError):
        document_passages(parser_output).__next__()


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

    store_labels.fn(test_config, labels, "TEST.DOC.0.1", "Q9081", "latest")

    labels = helper_list_labels_in_bucket(test_config, mock_bucket)

    assert len(labels) == 1
    assert labels[0] == "labelled_passages/Q9081/latest/TEST.DOC.0.1.json"


@pytest.mark.asyncio
async def test_text_block_inference(
    mock_wandb, test_config, mock_classifiers_dir, local_classifier_id
):
    _, mock_run, _ = mock_wandb
    test_config.local_classifier_dir = mock_classifiers_dir
    classifier = await load_classifier.fn(
        mock_run, test_config, local_classifier_id, "latest"
    )

    text = "I love fishing. Aquaculture is the best."
    block_id = "fish_block"
    labels = await text_block_inference.fn(
        classifier=classifier, block_id=block_id, text=text
    )

    assert len(labels.spans) > 0
    assert labels.id == block_id
    assert labels.metadata != {}
    # Set the labelled passages as empty as we are removing them.
    expected_concept_metadata = classifier.concept.model_dump()
    expected_concept_metadata["labelled_passages"] = []
    assert labels.metadata["concept"] == expected_concept_metadata
    datetime.fromisoformat(labels.metadata["inference_timestamp"])


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
