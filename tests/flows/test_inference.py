import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from uuid import UUID

import boto3
import pytest
from botocore.client import ClientError
from cpr_sdk.parser_models import BaseParserOutput, BlockType, HTMLData, HTMLTextBlock
from prefect.client.schemas.objects import FlowRun, State, StateType
from prefect.testing.utilities import prefect_test_harness

from flows.inference import (
    BatchInferenceException,
    ClassifierSpec,
    DocumentImportId,
    DocumentStem,
    _stringify,
    classifier_inference,
    determine_file_stems,
    document_passages,
    download_classifier_from_wandb_to_local,
    get_latest_ingest_documents,
    group_inference_results_into_states,
    list_bucket_file_stems,
    load_classifier,
    load_document,
    remove_sabin_file_stems,
    run_classifier_inference_on_batch_of_documents,
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
        _ = determine_file_stems(
            config=test_config,
            use_new_and_updated=False,
            requested_document_ids=[
                DocumentImportId("AF.document.002MMUCR.n0000"),
                DocumentImportId("AF.document.AFRDG00038.n00002"),
            ],
            current_bucket_file_stems=[
                DocumentStem("CCLW.document.i00001313.n0000"),
                DocumentStem("AF.document.002MMUCR.n0000"),
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
    doc_ids = [
        DocumentImportId(Path(doc_file).stem) for doc_file in mock_bucket_documents
    ]
    with prefect_test_harness():
        filtered_file_stems = await classifier_inference(
            classifier_specs=[ClassifierSpec(name="Q788", alias="latest")],
            document_ids=doc_ids,
            config=test_config,
        )
        assert filtered_file_stems == [DocumentStem(doc_id) for doc_id in doc_ids]

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
    document_stem = Path(mock_bucket_documents[1]).stem
    with pytest.raises(ValueError) as exc_info:
        result = await run_classifier_inference_on_document(
            config=test_config,
            file_stem=DocumentStem(document_stem),
            classifier_name=classifier_name,
            classifier_alias=classifier_alias,
            classifier=classifier,
        )

    assert "Cannot run inference on" in str(exc_info.value)

    # Run the function on a HTML document with has_valid_text=False
    document_stem = "HTML.document.0.1"
    with patch("flows.inference.load_document") as mock_load_document:
        html_document_invalid_text = BaseParserOutput(
            document_id=document_stem,
            document_metadata={},
            document_name="test document",
            document_description="test description",
            document_source_url=None,
            document_cdn_object=None,
            document_content_type="text/html",
            document_md5_sum=None,
            document_slug=document_stem,
            languages=None,
            translated=False,
            html_data=HTMLData(
                has_valid_text=False,
                text_blocks=[
                    HTMLTextBlock(
                        text=["This is an invalid text block."],
                        text_block_id="0",
                        type=BlockType.TEXT,
                    )
                ],
            ),
            pdf_data=None,
            pipeline_metadata={},
        )

        mock_load_document.return_value = html_document_invalid_text

        result = await run_classifier_inference_on_document(
            config=test_config,
            file_stem=DocumentStem(document_stem),
            classifier_name=classifier_name,
            classifier_alias=classifier_alias,
            classifier=classifier,
        )

        assert result is None

        # Load the stored labels from s3
        expected_key = f"labelled_passages/{classifier_name}/{classifier_alias}/{document_stem}.json"
        s3 = boto3.client("s3", region_name=test_config.bucket_region)
        response = s3.get_object(Bucket=test_config.cache_bucket, Key=expected_key)
        data = json.loads(response["Body"].read().decode("utf-8"))

        # Verify we stored NO labels
        assert len(data) == 0

    # Run the function on a document with english language
    document_stem = Path(mock_bucket_documents[0]).stem
    result = await run_classifier_inference_on_document(
        config=test_config,
        file_stem=DocumentStem(document_stem),
        classifier_name=classifier_name,
        classifier_alias=classifier_alias,
        classifier=classifier,
    )

    assert result is None

    # Verify that labels were stored in S3
    labels = helper_list_labels_in_bucket(test_config, mock_bucket)
    expected_key = (
        f"labelled_passages/{classifier_name}/{classifier_alias}/{document_stem}.json"
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

    document_stem = DocumentStem("CCLW.executive.8133.0")
    with pytest.raises(ClientError) as excinfo:
        await run_classifier_inference_on_document(
            config=test_config,
            file_stem=document_stem,
            classifier_name=classifier_name,
            classifier_alias=classifier_alias,
            classifier=classifier,
        )
    assert excinfo.value.response["Error"]["Code"] == "NoSuchKey"


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


def test_group_inference_results_into_states(snapshot):
    assert snapshot == group_inference_results_into_states(
        [
            FlowRun(
                name="1",
                id=UUID("09b81f2b-13c3-4d82-8afe-9d4a58971ef7"),
                flow_id=UUID("09b81f2b-13c3-4d82-8afe-9d4a58971ef7"),
                state=None,
            ),
            FlowRun(
                name="2",
                id=UUID("5c31d5a1-824f-42b2-ba7e-dab366ca5904"),
                flow_id=UUID("5c31d5a1-824f-42b2-ba7e-dab366ca5904"),
                state=State(type=StateType.CANCELLED),
            ),
            ValueError("2"),
            ValueError("3"),
            FlowRun(
                name="4",
                id=UUID("3a8fcdc1-f11e-4279-aee9-0624f91a2822"),
                flow_id=UUID("3a8fcdc1-f11e-4279-aee9-0624f91a2822"),
                state=State(type=StateType.COMPLETED),
                parameters={"classifier_name": "Q100", "classifier_alias": "v3"},
            ),
            FlowRun(
                name="5",
                id=UUID("c04c3798-b15e-427d-b51d-9e7b4870885f"),
                flow_id=UUID("c04c3798-b15e-427d-b51d-9e7b4870885f"),
                state=State(type=StateType.COMPLETED),
                parameters={"classifier_name": "Q200", "classifier_alias": "v5"},
            ),
        ]
    )


@pytest.mark.asyncio
async def test_run_classifier_inference_on_batch_of_documents(
    test_config,
    mock_classifiers_dir,
    mock_wandb,
    mock_bucket,
    mock_bucket_documents,
    snapshot,
):
    """Test successful batch processing of documents."""
    mock_wandb_init, mock_run, _ = mock_wandb
    test_config.local_classifier_dir = mock_classifiers_dir

    # Prepare test data - use only the PDF document which has languages field
    batch = [
        DocumentStem(Path(mock_bucket_documents[0]).stem)
    ]  # PDF.document.0.1 has languages
    classifier_name = "Q788"
    classifier_alias = "v7"
    config_json = {
        "cache_bucket": test_config.cache_bucket,
        "wandb_model_registry": test_config.wandb_model_registry,
        "wandb_entity": test_config.wandb_entity,
        "wandb_api_key": str(test_config.wandb_api_key.get_secret_value()),
        "aws_env": test_config.aws_env.value,
        "local_classifier_dir": str(test_config.local_classifier_dir),
    }

    # Should not raise any exceptions for successful processing
    await run_classifier_inference_on_batch_of_documents(
        batch=batch,
        config_json=config_json,
        classifier_name=classifier_name,
        classifier_alias=classifier_alias,
    )

    # Verify W&B was initialized
    mock_wandb_init.assert_called_once_with(
        entity=test_config.wandb_entity,
        job_type="concept_inference",
    )

    # Verify that a batch inference artifact was created
    from prefect.client.orchestration import get_client

    async with get_client() as client:
        artifacts = await client.read_artifacts()
        batch_artifacts = [a for a in artifacts if a.key and "batch-inference" in a.key]
        assert len(batch_artifacts) > 0, (
            "Expected at least one batch-inference artifact to be created"
        )

        # Sort artifacts by creation time and get the most recent one (this test's artifact)
        batch_artifacts.sort(key=lambda x: x.created, reverse=True)
        artifact = batch_artifacts[0]  # Most recently created

        assert artifact.description is not None, "Artifact should have a description"
        assert hasattr(artifact, "data"), "Artifact should have data"

        assert snapshot == artifact.data

    # Verify that inference outputs were stored in S3
    s3 = boto3.client("s3", region_name=test_config.bucket_region)
    expected_key = (
        f"labelled_passages/{classifier_name}/{classifier_alias}/{batch[0]}.json"
    )

    # Check that the S3 object exists
    response = s3.head_object(Bucket=test_config.cache_bucket, Key=expected_key)
    assert response["ContentLength"] > 0, (
        f"Expected S3 object {expected_key} to have content"
    )

    # Verify the content of the stored labels
    response = s3.get_object(Bucket=test_config.cache_bucket, Key=expected_key)
    data = json.loads(response["Body"].read().decode("utf-8"))

    # Verify we have at least one label for successful processing
    assert len(data) > 0, "Expected at least one labelled passage"


@pytest.mark.asyncio
async def test_run_classifier_inference_on_batch_of_documents_with_failures(
    test_config, mock_classifiers_dir, mock_wandb, mock_bucket, snapshot
):
    """Test batch processing with some document failures."""
    mock_wandb_init, mock_run, _ = mock_wandb
    test_config.local_classifier_dir = mock_classifiers_dir

    # Use non-existent document IDs to trigger failures
    batch = [DocumentStem("NonExistent.doc.1"), DocumentStem("AnotherMissing.doc.2")]
    classifier_name = "Q788"
    classifier_alias = "v8"
    config_json = {
        "cache_bucket": test_config.cache_bucket,
        "wandb_model_registry": test_config.wandb_model_registry,
        "wandb_entity": test_config.wandb_entity,
        "wandb_api_key": str(test_config.wandb_api_key.get_secret_value()),
        "aws_env": test_config.aws_env.value,
        "local_classifier_dir": str(test_config.local_classifier_dir),
    }

    with pytest.raises(BatchInferenceException) as exc_info:
        _ = await run_classifier_inference_on_batch_of_documents(
            batch=batch,
            config_json=config_json,
            classifier_name=classifier_name,
            classifier_alias=classifier_alias,
        )

        assert exc_info.value.message == "Failed to run inference on 2/2 documents."

    # Even with failures, an artifact should be created to track the failures
    from prefect.client.orchestration import get_client

    async with get_client() as client:
        artifacts = await client.read_artifacts()
        batch_artifacts = [a for a in artifacts if a.key and "batch-inference" in a.key]
        assert len(batch_artifacts) > 0, (
            "Expected artifact to be created even with failures"
        )

        # Sort artifacts by creation time and get the most recent one (this test's artifact)
        batch_artifacts.sort(key=lambda x: x.created, reverse=True)
        artifact = batch_artifacts[0]  # Most recently created

        assert artifact.description is not None, (
            "Failure artifact should have a description"
        )

        # Verify failure artifact data using snapshot
        assert snapshot == artifact.data

    # For failed documents, no S3 files should be created since the documents don't exist
    # The failure happens before store_labels is called
    s3 = boto3.client("s3", region_name=test_config.bucket_region)

    # Check that no labels were stored for the non-existent documents
    for doc_stem in batch:
        expected_key = (
            f"labelled_passages/{classifier_name}/{classifier_alias}/{doc_stem}.json"
        )
        with pytest.raises(ClientError) as exc_info:
            s3.head_object(Bucket=test_config.cache_bucket, Key=expected_key)
        assert exc_info.value.response["Error"]["Code"] == "404"


@pytest.mark.asyncio
async def test_run_classifier_inference_on_batch_of_documents_empty_batch(
    test_config, mock_classifiers_dir, mock_wandb, mock_bucket, snapshot
):
    """Test batch processing with empty batch."""
    mock_wandb_init, mock_run, _ = mock_wandb
    test_config.local_classifier_dir = mock_classifiers_dir

    batch: list[DocumentStem] = []
    classifier_name = "Q788"
    classifier_alias = "v12"
    config_json = {
        "cache_bucket": test_config.cache_bucket,
        "wandb_model_registry": test_config.wandb_model_registry,
        "wandb_entity": test_config.wandb_entity,
        "wandb_api_key": str(test_config.wandb_api_key.get_secret_value()),
        "aws_env": test_config.aws_env.value,
        "local_classifier_dir": str(test_config.local_classifier_dir),
    }

    # Should complete successfully with empty batch
    await run_classifier_inference_on_batch_of_documents(
        batch=batch,
        config_json=config_json,
        classifier_name=classifier_name,
        classifier_alias=classifier_alias,
    )

    # Verify W&B was still initialized
    mock_wandb_init.assert_called_once()

    # Verify artifact creation even for empty batch
    from prefect.client.orchestration import get_client

    async with get_client() as client:
        artifacts = await client.read_artifacts()
        batch_artifacts = [a for a in artifacts if a.key and "batch-inference" in a.key]
        assert len(batch_artifacts) > 0, (
            "Expected artifact to be created even for empty batch"
        )

        # Sort artifacts by creation time and get the most recent one (this test's artifact)
        batch_artifacts.sort(key=lambda x: x.created, reverse=True)
        artifact = batch_artifacts[0]  # Most recently created

        # Verify empty batch artifact data using snapshot
        assert snapshot == artifact.data

    # For empty batch, no S3 files should be created since there are no documents to process
    # Since batch is empty, we don't need to check any specific files - there should be none created
