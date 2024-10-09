import json
from pathlib import Path
from unittest.mock import patch

import boto3
import pytest
from prefect.testing.utilities import prefect_test_harness

from flows.inference import (
    classifier_inference,
    config,
    determine_document_ids,
    list_bucket_doc_ids,
    load_classifier,
    load_document,
    store_labels,
    text_block_inference,
)
from src.labelled_passage import LabelledPassage
from src.span import Span


def helper_list_labels_in_bucket(bucket_name):
    # Find out what is now in the spans bucket
    s3 = boto3.client("s3", region_name=config.bucket_region)
    response = s3.list_objects_v2(
        Bucket=bucket_name, Prefix=config.document_target_prefix
    )
    labels = [c.get("Key") for c in response.get("Contents", [])]
    return labels


def test_list_bucket_doc_ids(mock_bucket_documents):
    expected_ids = [Path(d).stem for d in mock_bucket_documents]
    got_ids = list_bucket_doc_ids()
    assert sorted(expected_ids) == sorted(got_ids)


@pytest.mark.parametrize(
    ("doc_ids", "bucket_ids", "expected"),
    [
        (["1"], ["1", "2", "3"], ["1"]),
        (None, ["1"], ["1"]),
    ],
)
def test_determine_document_ids(doc_ids, bucket_ids, expected):
    got = determine_document_ids(
        requested_document_ids=doc_ids,
        current_bucket_ids=bucket_ids,
    )
    assert got == expected


def test_determine_document_ids__error():
    with pytest.raises(ValueError):
        determine_document_ids(
            requested_document_ids=["1", "2"],
            current_bucket_ids=["3", "4"],
        )


def test_load_classifier__existing_classifier(
    mock_classifiers_dir, local_classifier_id
):
    classifier = load_classifier(local_classifier_id)
    assert local_classifier_id == classifier.concept.wikibase_id


def test_download_classifier__wandb_classifier():
    # TODO mock the interface and test code path
    pass


def test_load_document(mock_bucket_documents):
    for doc_file_name in mock_bucket_documents:
        doc_id = Path(doc_file_name).stem
        doc = load_document(document_id=doc_id)
        assert doc_id == doc.document_id


def test_store_labels(mock_bucket):
    text = "This is a test text block"
    spans = [Span(text=text, start_index=15, end_index=19)]
    labels = [LabelledPassage(text=text, spans=spans)]

    store_labels(labels, "TEST.DOC.0.1", "Q9081")

    labels = helper_list_labels_in_bucket(mock_bucket)

    assert len(labels) == 1
    assert labels[0] == "spans/TEST.DOC.0.1.Q9081.json"


def test_text_block_inference(mock_classifiers_dir, local_classifier_id):
    with patch.object(config, "local_classifier_dir", new=mock_classifiers_dir):
        classifier = load_classifier(local_classifier_id)

    text = "I love fishing. Aquaculture is the best."
    block_id = "fish_block"
    labels = text_block_inference.fn(
        classifier=classifier, block_id=block_id, text=text
    )

    assert len(labels.spans) > 0
    assert labels.id == block_id


def test_classifier_inference(mock_classifiers_dir, mock_bucket, mock_bucket_documents):
    doc_ids = [Path(doc_file).stem for doc_file in mock_bucket_documents]
    with prefect_test_harness():
        classifier_inference(document_ids=doc_ids)

    labels = helper_list_labels_in_bucket(mock_bucket)

    assert sorted(labels) == [
        "spans/HTML.document.0.1.Q788.json",
        "spans/PDF.document.0.1.Q788.json",
    ]

    for key in labels:
        s3 = boto3.client("s3", region_name=config.bucket_region)
        response = s3.get_object(Bucket=config.cache_bucket, Key=key)
        data = json.loads(response["Body"].read().decode("utf-8"))

        # Some spans where identified
        with_spans = [d for d in data if len(d["spans"]) > 0]
        assert len(with_spans) > 0
