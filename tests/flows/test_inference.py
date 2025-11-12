import json
import os
import random
import string
import tempfile
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from botocore.client import ClientError
from cpr_sdk.parser_models import (
    BaseParserOutput,
    BlockType,
    PDFData,
    PDFTextBlock,
)
from prefect.artifacts import Artifact
from prefect.client.schemas.objects import FlowRun
from prefect.context import FlowRunContext
from prefect.states import Completed, Running

from flows.classifier_specs.spec_interface import ClassifierSpec, DontRunOnEnum
from flows.inference import (
    BatchInferenceResult,
    Metadata,
    ParameterisedFlow,
    SingleDocumentInferenceResult,
    _inference_batch_of_documents,
    _stringify,
    determine_file_stems,
    did_inference_fail,
    document_passages,
    filter_document_batch,
    gather_successful_document_stems,
    get_existing_inference_results,
    get_inference_fault_metadata,
    get_latest_ingest_documents,
    inference,
    inference_batch_of_documents_cpu,
    list_bucket_file_stems,
    load_classifier_from_model_registry,
    load_document,
    parse_client_error_details,
    process_single_document_inference,
    run_classifier_inference_on_document,
    store_inference_result,
    store_metadata,
    text_block_inference,
)
from flows.utils import (
    DocumentImportId,
    DocumentStem,
    Fault,
    JsonDict,
    deserialise_pydantic_list_from_jsonl,
    deserialise_pydantic_list_with_fallback,
    serialise_pydantic_list_as_jsonl,
)
from knowledge_graph.identifiers import ClassifierID, ConceptID, Identifier, WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.span import Span


@pytest.fixture
def mock_deployment():
    """A `run_deployment` mock wrapper that lets result state be customised."""

    class MockDeployment:
        def __init__(self, state, name="test-flow-run"):
            """Mock run deployment, a state per call"""
            self.state = state
            self.name = name
            self._mock_patch = None

        async def mock_awaitable(self, *args, **kwargs):
            """Generate FlowRun with next state from the iterator"""
            flow_id = uuid.uuid4()
            flow_name = f"{self.name}-{str(flow_id)[:8]}"
            return FlowRun(flow_id=flow_id, name=flow_name, state=self.state)

        def __enter__(self):
            self._mock_patch = patch("flows.utils.run_deployment")
            mock_instance = self._mock_patch.__enter__()
            mock_instance.side_effect = self.mock_awaitable
            return mock_instance

        def __exit__(self, exc_type, exc_val, exc_tb):
            return self._mock_patch.__exit__(exc_type, exc_val, exc_tb)

    return MockDeployment


async def helper_list_labels_in_bucket(test_config, bucket_name, async_s3_client):
    # Find out what is now in the spans bucket

    response = await async_s3_client.list_objects_v2(
        Bucket=bucket_name, Prefix=test_config.inference_document_target_prefix
    )
    labels = [c.get("Key") for c in response.get("Contents", [])]
    return labels


@pytest.mark.asyncio
async def test_list_bucket_file_stems(test_config, mock_async_bucket_documents):
    expected_ids = [Path(d).stem for d in mock_async_bucket_documents]
    got_ids = await list_bucket_file_stems(test_config)
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
@pytest.mark.asyncio
async def test_determine_file_stems(
    mock_bucket_new_and_updated_documents_json,
    test_config,
    doc_ids,
    bucket_ids,
    expected,
):
    got = await determine_file_stems(
        config=test_config,
        use_new_and_updated=False,
        requested_document_ids=doc_ids,
        current_bucket_file_stems=bucket_ids,
    )
    assert got == expected


@pytest.mark.asyncio
async def test_determine_file_stems__error(
    mock_bucket_new_and_updated_documents_json, test_config
):
    with pytest.raises(ValueError):
        _ = await determine_file_stems(
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
    wikibase_id, classifier_id, wandb_registry_version = local_classifier_id
    _, mock_run, mock_artifact = mock_wandb
    spec = ClassifierSpec(
        wikibase_id=wikibase_id,
        classifier_id=classifier_id,  # no longer used but required for validation
        wandb_registry_version=wandb_registry_version,
    )
    classifier = await load_classifier_from_model_registry(
        mock_run,
        test_config,
        spec,
    )

    assert wikibase_id == classifier.concept.wikibase_id
    assert classifier.id == classifier_id


@pytest.mark.asyncio
async def test_load_document(
    test_config, mock_async_bucket_documents, mock_s3_async_client
):
    valid_doc, invalid_doc = mock_async_bucket_documents

    classifier_spec = ClassifierSpec(
        wikibase_id=WikibaseID("Q123"),
        classifier_id=ClassifierID("testabcd"),
        wandb_registry_version="v1",
    )

    # Invalid doc
    invalid_document_stem = Path(invalid_doc).stem
    invalid_document_result = SingleDocumentInferenceResult(
        document_stem=DocumentStem(invalid_document_stem),
        document=None,
        labelled_passages=[],
        classifier_spec=classifier_spec,
    )

    with pytest.raises(ValueError, match="non-English language"):
        await load_document(
            test_config,
            document_result=invalid_document_result,
            s3_client=mock_s3_async_client,
        )

    # Valid Doc
    valid_document_stem = Path(valid_doc).stem
    valid_document_result = SingleDocumentInferenceResult(
        document_stem=DocumentStem(valid_document_stem),
        document=None,
        labelled_passages=[],
        classifier_spec=classifier_spec,
    )
    result = await load_document(
        test_config,
        document_result=valid_document_result,
        s3_client=mock_s3_async_client,
    )
    assert result.document is not None


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


@pytest.mark.asyncio
async def test_text_block_inference_with_results(
    mock_wandb, test_config, mock_classifiers_dir
):
    _, mock_run, _ = mock_wandb
    test_config.local_classifier_dir = mock_classifiers_dir
    spec = ClassifierSpec(
        wikibase_id=WikibaseID("Q9081"),
        classifier_id=ClassifierID.generate("Q9081", "v3"),
        wandb_registry_version="v3",
    )
    classifier = await load_classifier_from_model_registry(
        mock_run,
        test_config,
        spec,
    )

    text = "I love fishing. Aquaculture is the best."
    block_id = "fish_block"
    labels = text_block_inference(
        classifier=classifier, classifier_spec=spec, block_id=block_id, text=text
    )

    assert len(labels.spans) > 0
    assert labels.id == block_id
    assert labels.metadata != {}
    assert "classifier_spec" in labels.metadata
    assert labels.metadata["classifier_spec"] == spec.model_dump()
    # Set the labelled passages as empty as we are removing them.
    expected_concept_metadata = classifier.concept.model_dump()
    expected_concept_metadata["labelled_passages"] = []
    assert labels.metadata["concept"] == expected_concept_metadata
    # check whether the timestamps are valid
    for span in labels.spans:
        assert isinstance(span.timestamps[0], datetime)


@pytest.mark.asyncio
async def test_text_block_inference_without_results(
    mock_wandb, test_config, mock_classifiers_dir
):
    _, mock_run, _ = mock_wandb
    test_config.local_classifier_dir = mock_classifiers_dir
    spec = ClassifierSpec(
        wikibase_id=WikibaseID("Q9081"),
        classifier_id=ClassifierID.generate("Q9081", "v3"),
        wandb_registry_version="v3",
    )
    classifier = await load_classifier_from_model_registry(
        mock_run,
        test_config,
        spec,
    )

    text = "Rockets are cool. We should build more rockets."
    block_id = "fish_block"
    labels = text_block_inference(
        classifier=classifier, classifier_spec=spec, block_id=block_id, text=text
    )

    assert len(labels.spans) == 0
    assert labels.id == block_id
    # When there are no spans, metadata should be empty
    assert labels.metadata == {}


@pytest.mark.asyncio
async def test_inference_with_dont_run_on_filter(
    test_config,
    mock_classifiers_dir,
    mock_wandb,
    mock_async_bucket,
    mock_async_bucket_multiple_sources,
    mock_deployment,
):
    input_doc_ids = [
        DocumentImportId(Path(doc).stem) for doc in mock_async_bucket_multiple_sources
    ]
    gef_doc_id, cpr_doc_id, sabin_doc_id = input_doc_ids

    spec = ClassifierSpec(
        wikibase_id=WikibaseID("Q788"),
        classifier_id="bvaw9xxm",
        wandb_registry_version="v13",
        dont_run_on=["cpr", "sabin"],
    )

    state = Completed(
        data=BatchInferenceResult(
            batch_document_stems=[gef_doc_id],
            successful_document_stems=[gef_doc_id],
            classifier_spec=spec,
        ),
    )
    with mock_deployment(state) as mock_inference_run_deployment:
        # run the inference flow
        _ = await inference(
            classifier_specs=[spec],
            document_ids=input_doc_ids,
            config=test_config,
        )

        mock_inference_run_deployment.call_args.kwargs["parameters"]["batch"] == [
            gef_doc_id
        ]

        summary_artifact = await Artifact.get("removal-details-sandbox")
        assert summary_artifact and summary_artifact.description
        assert json.loads(summary_artifact.data) == [
            {
                "Wikibase ID": spec.wikibase_id,
                "Classifier ID": spec.classifier_id,
                "Dont Run Ons": ["cpr", "sabin"],
                "Removals": 2,
                "Cached (Skipped)": 0,
            }
        ]


@pytest.mark.asyncio
async def test_inference_with_gpu_enabled(
    test_config,
    mock_classifiers_dir,
    mock_wandb,
    mock_async_bucket,
    mock_async_bucket_multiple_sources,
    mock_deployment,
):
    input_doc_ids = [
        DocumentImportId(Path(doc).stem) for doc in mock_async_bucket_multiple_sources
    ]
    output_doc_ids = [DocumentStem(doc) for doc in input_doc_ids]

    spec = ClassifierSpec(
        wikibase_id=WikibaseID("Q788"),
        classifier_id="bvaw9xxm",
        wandb_registry_version="v13",
        compute_environment=ClassifierSpec.ComputeEnvironment(gpu=True),
    )

    state = Completed(
        data=BatchInferenceResult(
            batch_document_stems=output_doc_ids,
            successful_document_stems=output_doc_ids,
            classifier_spec=spec,
        ),
    )
    with mock_deployment(state) as mock_inference_run_deployment:
        # run the inference flow
        _ = await inference(
            classifier_specs=[spec],
            document_ids=input_doc_ids,
            config=test_config,
        )
        called_deployment = mock_inference_run_deployment.call_args.kwargs["name"]

    assert called_deployment == (
        "inference-batch-of-documents-gpu/kg-inference-batch-of-documents-gpu-sandbox"
    )


@pytest.mark.asyncio
async def test_inference_flow_returns_successful_batch_inference_result_with_docs(
    test_config,
    mock_classifiers_dir,
    mock_wandb,
    mock_async_bucket_documents,
    mock_deployment,
):
    """Test inference flow when creating batches of inference results"""
    input_doc_ids = [
        DocumentImportId(Path(doc_file).stem)
        for doc_file in mock_async_bucket_documents
    ]

    expected_classifier_spec = ClassifierSpec(
        wikibase_id=WikibaseID("Q788"),
        classifier_id="bvaw9xxm",
        wandb_registry_version="v13",
    )

    state = Completed(
        data=BatchInferenceResult(
            batch_document_stems=list(input_doc_ids),
            successful_document_stems=list(input_doc_ids),
            classifier_spec=expected_classifier_spec,
        )
    )

    with mock_deployment(state) as mock_inference_run_deployment:
        inference_result = await inference(
            classifier_specs=[expected_classifier_spec],
            document_ids=input_doc_ids,
            config=test_config,
        )

        mock_inference_run_deployment.assert_called_once()

        assert type(inference_result) is set

        assert inference_result == set(input_doc_ids)


@pytest.mark.asyncio
async def test_get_latest_ingest_documents(
    test_config, mock_bucket_new_and_updated_documents_json
):
    _, latest_docs = mock_bucket_new_and_updated_documents_json
    doc_ids = await get_latest_ingest_documents(test_config)
    assert set(doc_ids) == latest_docs


@pytest.mark.asyncio
async def test_get_latest_ingest_documents_no_latest(
    test_config,
    # Setup the empty bucket
    mock_async_bucket,
):
    with pytest.raises(
        ValueError,
        match="failed to find",
    ):
        await get_latest_ingest_documents(test_config)


@pytest.mark.asyncio
async def test_run_classifier_inference_on_document(
    test_config,
    mock_classifiers_dir,
    mock_wandb,
    snapshot,
):
    # Setup
    _, mock_run, _ = mock_wandb
    test_config.local_classifier_dir = mock_classifiers_dir

    classifier_spec = ClassifierSpec(
        wikibase_id=WikibaseID("Q788"),
        classifier_id=ClassifierID.generate("Q788", "v5"),
        wandb_registry_version="v5",
    )

    # Load classifier
    classifier = await load_classifier_from_model_registry(
        mock_run,
        test_config,
        classifier_spec,
    )
    document_stem = DocumentStem("HTML.document.0.1")
    store_result = SingleDocumentInferenceResult(
        document_stem=document_stem,
        labelled_passages=[],
        document=BaseParserOutput(
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
            html_data=None,
            pdf_data=PDFData(
                page_metadata=[],
                md5sum="",
                text_blocks=[
                    PDFTextBlock(
                        text=[
                            "Ministry of fishing, overfishing, seafood harvest and fisheries."
                        ],
                        text_block_id="2",
                        page_number=1,
                        coords=[],
                        type=BlockType.TEXT,
                        type_confidence=0.5,
                    )
                ],
            ),
            pipeline_metadata={},
        ),
        classifier_spec=classifier_spec,
    )

    result = await run_classifier_inference_on_document(
        result=store_result,
        classifier=classifier,
    )
    assert result.labelled_passages == snapshot


@pytest.mark.asyncio
async def test_inference_batch_of_documents_cpu(
    test_config,
    mock_classifiers_dir,
    mock_wandb,
    mock_s3_async_client,
    mock_async_bucket_documents,
    mock_prefect_s3_block,
    snapshot,
):
    """Test successful batch processing of documents."""
    mock_wandb_init, mock_run, _ = mock_wandb
    test_config.local_classifier_dir = mock_classifiers_dir

    # Prepare test data - use only the PDF document which has languages field
    batch = [
        DocumentStem(Path(mock_async_bucket_documents[0]).stem)
    ]  # PDF.document.0.1 has languages
    classifier_spec = ClassifierSpec(
        wikibase_id=WikibaseID("Q788"),
        classifier_id="6vxrmcuf",
        wandb_registry_version="v7",
    )
    config_json = JsonDict(
        {
            "cache_bucket": test_config.cache_bucket,
            "wandb_model_registry": test_config.wandb_model_registry,
            "wandb_entity": test_config.wandb_entity,
            "wandb_api_key": str(test_config.wandb_api_key.get_secret_value()),
            "aws_env": test_config.aws_env.value,
            "local_classifier_dir": str(test_config.local_classifier_dir),
        }
    )

    # Should not raise any exceptions for successful processing
    inference_batch_of_documents_cpu.flow_run_name = (
        "test-inference-batch-of-documents-cpu"
    )
    result_state = await inference_batch_of_documents_cpu(
        batch=batch,
        config_json=config_json,
        classifier_spec_json=JsonDict(classifier_spec.model_dump()),
        return_state=True,
    )

    result = await result_state.result()
    assert isinstance(result, BatchInferenceResult)
    assert result.batch_document_stems == batch
    assert result.successful_document_stems == batch
    assert result.classifier_spec == classifier_spec
    assert not result.failed

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
        assert snapshot == (artifact.data, artifact.description)

    # Verify that inference outputs were stored in S3 using async s3 client
    expected_key = f"labelled_passages/{classifier_spec.wikibase_id}/{classifier_spec.classifier_id}/{batch[0]}.json"

    # Check that the S3 object exists
    response = await mock_s3_async_client.head_object(
        Bucket=test_config.cache_bucket, Key=expected_key
    )
    assert response["ContentLength"] > 0, (
        f"Expected S3 object {expected_key} to have content"
    )

    # Verify the content of the stored labels
    response = await mock_s3_async_client.get_object(
        Bucket=test_config.cache_bucket, Key=expected_key
    )
    jsonl_content = await response["Body"].read()
    jsonl_content_decoded = jsonl_content.decode("utf-8")

    # Parse JSONL format - each line is a JSON object
    lines = [
        line.strip()
        for line in jsonl_content_decoded.strip().split("\n")
        if line.strip()
    ]

    # Verify we have at least one label for successful processing
    assert len(lines) > 0, "Expected at least one labelled passage"

    # Verify each line is valid JSON
    for line in lines:
        json.loads(line)  # This will raise if invalid JSON


@pytest.mark.asyncio
async def test_inference_batch_of_documents_cpu_with_failures(
    test_config,
    mock_classifiers_dir,
    mock_wandb,
    mock_async_bucket,
    mock_s3_async_client,
    snapshot,
    mock_prefect_s3_block,
):
    """Test batch processing with some document failures."""
    mock_wandb_init, mock_run, _ = mock_wandb
    test_config.local_classifier_dir = mock_classifiers_dir

    # Use non-existent document IDs to trigger failures
    batch = [DocumentStem("NonExistent.doc.1"), DocumentStem("AnotherMissing.doc.2")]
    classifier_spec = ClassifierSpec(
        wikibase_id=WikibaseID("Q788"),
        classifier_id="aaaa2222",
        wandb_registry_version="v8",
    )
    config_json = JsonDict(
        {
            "cache_bucket": test_config.cache_bucket,
            "wandb_model_registry": test_config.wandb_model_registry,
            "wandb_entity": test_config.wandb_entity,
            "wandb_api_key": str(test_config.wandb_api_key.get_secret_value()),
            "aws_env": test_config.aws_env.value,
            "local_classifier_dir": str(test_config.local_classifier_dir),
        }
    )

    with pytest.raises(Fault) as exc_info:
        inference_batch_of_documents_cpu.flow_run_name = (
            "test-inference-batch-of-documents-cpu-with-failures"
        )
        _ = await inference_batch_of_documents_cpu(
            batch=batch,
            config_json=config_json,
            classifier_spec_json=JsonDict(classifier_spec.model_dump()),
        )

        assert exc_info.value.msg == "Failed to run inference on 2/2 documents."
        assert isinstance(exc_info.value.data, BatchInferenceResult)

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
        assert snapshot == (artifact.data, artifact.description)

    # For failed documents, no S3 files should be created since the documents don't exist
    # The failure happens before store_labels is called
    # using async s3 client to check

    # Check that no labels were stored for the non-existent documents
    for doc_stem in batch:
        expected_key = f"labelled_passages/{classifier_spec.wikibase_id}/{classifier_spec.wandb_registry_version}/{doc_stem}.json"
        with pytest.raises(ClientError) as exc_info:
            await mock_s3_async_client.head_object(
                Bucket=test_config.cache_bucket, Key=expected_key
            )
        assert exc_info.value.response["Error"]["Code"] == "404"


@pytest.mark.asyncio
async def test_inference_batch_of_documents_cpu_empty_batch(
    test_config,
    mock_classifiers_dir,
    mock_wandb,
    mock_async_bucket,
    snapshot,
    mock_prefect_s3_block,
):
    """Test batch processing with empty batch."""
    mock_wandb_init, mock_run, _ = mock_wandb
    test_config.local_classifier_dir = mock_classifiers_dir

    batch: list[DocumentStem] = []
    classifier_spec = ClassifierSpec(
        wikibase_id=WikibaseID("Q788"),
        classifier_id="aaaa2222",
        wandb_registry_version="v12",
    )
    config_json = JsonDict(
        {
            "cache_bucket": test_config.cache_bucket,
            "wandb_model_registry": test_config.wandb_model_registry,
            "wandb_entity": test_config.wandb_entity,
            "wandb_api_key": str(test_config.wandb_api_key.get_secret_value()),
            "aws_env": test_config.aws_env.value,
            "local_classifier_dir": str(test_config.local_classifier_dir),
        }
    )

    # Should complete successfully with empty batch
    inference_batch_of_documents_cpu.flow_run_name = (
        "test-inference-batch-of-documents-cpu-empty-batch"
    )
    _ = await inference_batch_of_documents_cpu(
        batch=batch,
        config_json=config_json,
        classifier_spec_json=JsonDict(classifier_spec.model_dump()),
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
        assert snapshot == (artifact.data, artifact.description)

    # For empty batch, no S3 files should be created since there are no documents to process
    # Since batch is empty, we don't need to check any specific files - there should be none created


@pytest.mark.asyncio
async def test__inference_batch_of_documents(
    test_config,
    mock_classifiers_dir,
    mock_wandb,
    mock_async_bucket_documents,
    mock_prefect_s3_block,
):
    """Test the inner _inference_batch_of_documents function with mocked flow context."""
    mock_wandb_init, mock_run, _ = mock_wandb
    test_config.local_classifier_dir = mock_classifiers_dir

    # Prepare test data - use only the PDF document which has languages field
    batch = [
        DocumentStem(Path(mock_async_bucket_documents[0]).stem)
    ]  # PDF.document.0.1 has languages
    classifier_spec = ClassifierSpec(
        wikibase_id=WikibaseID("Q788"),
        classifier_id="aaaa2222",
        wandb_registry_version="v7",
    )
    config_json = JsonDict(
        {
            "cache_bucket": test_config.cache_bucket,
            "wandb_model_registry": test_config.wandb_model_registry,
            "wandb_entity": test_config.wandb_entity,
            "wandb_api_key": str(test_config.wandb_api_key.get_secret_value()),
            "aws_env": test_config.aws_env.value,
            "local_classifier_dir": str(test_config.local_classifier_dir),
        }
    )

    # Still needed since there's an inner function that uses this.
    mock_flow_run = MagicMock()
    mock_flow_run.name = "test-flow-run"
    mock_context = MagicMock(spec=FlowRunContext)
    mock_context.flow_run = mock_flow_run

    with patch("flows.inference.get_run_context", return_value=mock_context):
        # Should not raise any exceptions for successful processing
        result = await _inference_batch_of_documents(
            batch=batch,
            config_json=config_json,
            classifier_spec_json=JsonDict(classifier_spec.model_dump()),
        )

    assert isinstance(result, BatchInferenceResult)
    assert result.batch_document_stems == batch
    assert result.successful_document_stems == batch
    assert result.classifier_spec == classifier_spec
    assert result.failed is False, "All documents succeeded, so failed should be False"
    assert result.failed_document_count == 0
    assert result.all_document_count == 1

    # Verify W&B was initialized
    mock_wandb_init.assert_called_once_with(
        entity=test_config.wandb_entity,
        job_type="concept_inference",
    )

    # Test partial success
    batch = [
        DocumentStem(Path(mock_async_bucket_documents[0]).stem),  # Real document
        DocumentStem("NonExistent.doc.1"),  # Will fail
    ]

    with patch("flows.inference.get_run_context", return_value=mock_context):
        result_partial = await _inference_batch_of_documents(
            batch=batch,
            config_json=config_json,
            classifier_spec_json=JsonDict(classifier_spec.model_dump()),
        )

        assert isinstance(result_partial, BatchInferenceResult)
        assert result_partial.batch_document_stems == batch
        assert len(result_partial.successful_document_stems) == 1
        assert result_partial.failed is False  # Not failed because some succeeded
        assert result_partial.failed_document_count > 0


def test_batch_inference_result_failed_property():
    """Test BatchInferenceResult.failed property."""
    classifier_spec = ClassifierSpec(
        wikibase_id=WikibaseID("Q788"),
        classifier_id="aaaa2222",
        wandb_registry_version="v7",
    )

    doc1 = DocumentStem("TEST.doc.1.1")
    doc2 = DocumentStem("TEST.doc.1.2")

    # Empty batch - not failed
    result_empty = BatchInferenceResult(
        batch_document_stems=[],
        successful_document_stems=[],
        classifier_spec=classifier_spec,
    )
    assert result_empty.failed is False

    # All successful - not failed
    result_all_success = BatchInferenceResult(
        batch_document_stems=[doc1, doc2],
        successful_document_stems=[doc1, doc2],
        classifier_spec=classifier_spec,
    )
    assert result_all_success.failed is False

    # Partial success - not failed (some succeeded)
    result_partial = BatchInferenceResult(
        batch_document_stems=[doc1, doc2],
        successful_document_stems=[doc1],
        classifier_spec=classifier_spec,
    )
    assert result_partial.failed is False

    # All failed - is failed
    result_all_failed = BatchInferenceResult(
        batch_document_stems=[doc1, doc2],
        successful_document_stems=[],
        classifier_spec=classifier_spec,
    )
    assert result_all_failed.failed is True


def test_batch_inference_result_failed_document_stems():
    """Test BatchInferenceResult.failed_document_stems property."""
    classifier_spec = ClassifierSpec(
        wikibase_id=WikibaseID("Q788"),
        classifier_id="aaaa2222",
        wandb_registry_version="v7",
    )

    doc1 = DocumentStem("TEST.doc.1.1")
    doc2 = DocumentStem("TEST.doc.1.2")
    doc3 = DocumentStem("TEST.doc.1.3")

    # All successful
    result = BatchInferenceResult(
        batch_document_stems=[doc1, doc2, doc3],
        successful_document_stems=[doc1, doc2, doc3],
        classifier_spec=classifier_spec,
    )
    assert result.failed_document_stems == []

    # Some failed
    result = BatchInferenceResult(
        batch_document_stems=[doc1, doc2, doc3],
        successful_document_stems=[doc1],
        classifier_spec=classifier_spec,
    )
    assert set(result.failed_document_stems) == {doc2, doc3}
    assert result.failed_document_count == 2
    assert not result.failed


def test_batch_inference_result_all_document_count():
    """Test BatchInferenceResult.all_document_count property."""
    classifier_spec = ClassifierSpec(
        wikibase_id=WikibaseID("Q788"),
        classifier_id="aaaa2222",
        wandb_registry_version="v7",
    )

    doc1 = DocumentStem("TEST.doc.1.1")
    doc2 = DocumentStem("TEST.doc.1.2")

    result = BatchInferenceResult(
        batch_document_stems=[doc1, doc2],
        successful_document_stems=[doc1],
        classifier_spec=classifier_spec,
    )
    assert result.all_document_count == 2
    assert len(result.successful_document_stems) == 1
    assert not result.failed


def test_jsonl_serialization_roundtrip():
    """Test that JSONL serialization and deserialization works correctly."""
    test_passages = [
        LabelledPassage(
            id="passage1",
            text="This is the first test passage",
            spans=[
                Span(
                    text="This is the first test passage",
                    start_index=17,
                    end_index=21,
                    concept_id="Q123",
                    labellers=["test_labeller"],
                    timestamps=["2023-01-01T00:00:00"],
                    id="span1",
                    labelled_text="test",
                )
            ],
            metadata={"source": "test"},
        ),
        LabelledPassage(
            id="passage2",
            text="This is the second test passage",
            spans=[],
            metadata={"source": "test"},
        ),
    ]

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".jsonl", delete=False) as f:
        try:
            jsonl_string = serialise_pydantic_list_as_jsonl(test_passages)
            serialized_data = BytesIO(jsonl_string.encode("utf-8"))
            f.write(serialized_data.read().decode("utf-8"))
            f.flush()

            with open(f.name, "r") as read_file:
                content = read_file.read()

            deserialized_passages = deserialise_pydantic_list_from_jsonl(
                content, LabelledPassage
            )

            assert deserialized_passages == test_passages
        finally:
            # Clean up
            os.unlink(f.name)


def test_original_format_fallback():
    """Test that the original serialization format can be deserialized correctly."""
    # Create test data
    test_passages = [
        LabelledPassage(
            id="passage1",
            text="This is the first test passage",
            spans=[
                Span(
                    text="This is the first test passage",
                    start_index=17,
                    end_index=21,
                    concept_id="Q123",
                    labellers=["test_labeller"],
                    timestamps=["2023-01-01T00:00:00"],
                    id="span1",
                    labelled_text="test",
                )
            ],
            metadata={"source": "test"},
        ),
        LabelledPassage(
            id="passage2",
            text="This is the second test passage",
            spans=[],
            metadata={"source": "test"},
        ),
    ]

    def original_serialise_labels(labels: list[LabelledPassage]) -> BytesIO:
        data = [label.model_dump_json() for label in labels]
        return BytesIO(json.dumps(data).encode("utf-8"))

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as f:
        try:
            serialized_data = original_serialise_labels(test_passages)
            f.write(serialized_data.read().decode("utf-8"))
            f.flush()

            with open(f.name, "r") as read_file:
                content = read_file.read()

            # This should trigger the fallback logic automatically
            deserialized_passages = deserialise_pydantic_list_with_fallback(
                content, LabelledPassage
            )

            assert deserialized_passages == test_passages
        finally:
            # Clean up
            os.unlink(f.name)


def test_document_passages(
    parser_output_pdf,
    parser_output_html,
    parser_output_html_converted_to_pdf,
) -> None:
    """Test that the document_passages function works correctly."""

    passages = list(document_passages(parser_output_pdf))
    assert parser_output_pdf.pdf_data is not None
    assert len(passages) == len(parser_output_pdf.pdf_data.text_blocks)

    passages = list(document_passages(parser_output_html))
    assert parser_output_html.html_data is not None
    assert len(passages) == len(parser_output_html.html_data.text_blocks)

    passages = list(document_passages(parser_output_html_converted_to_pdf))
    assert parser_output_html_converted_to_pdf.pdf_data is not None
    assert len(passages) == len(
        parser_output_html_converted_to_pdf.pdf_data.text_blocks
    )


@pytest.mark.parametrize(
    ("dont_run_on", "removed"),
    [
        (None, ["Sabin.document.2524.placeholder"]),
        ([], ["Sabin.document.2524.placeholder"]),
        (
            ["gef"],
            ["GEF.document.787.n0000.json", "Sabin.document.2524.placeholder"],
        ),
        (
            ["cpr", "sabin"],
            [
                "Sabin.document.9869.10352.json",
                "CPR.document.i00003835.n0000.json",
                "Sabin.document.2524.placeholder",
            ],
        ),
        (
            DontRunOnEnum.__members__.keys(),
            [
                "GCF.document.FP181_24530.13164.json",
                "CCLW.document.i00000300.n0000.json",
                "GEF.document.787.n0000.json",
                "AF.document.AFRDG00005.n0000.json",
                "CIF.document.XCTFMB030A.n0000_translated_en.json",
                "OEP.document.i00000231.n0000.json",
                "Sabin.document.9869.10352.json",
                "CPR.document.i00003835.n0000.json",
                "Sabin.document.2524.placeholder",
                "UNCDB.document.1.1",
                "UNCDB.document.2.2.json",
            ],
        ),
    ],
)
def test_filter_document_batch(dont_run_on, removed):
    file_stems = [
        "GCF.document.FP181_24530.13164.json",
        "CCLW.document.i00000300.n0000.json",
        "GEF.document.787.n0000.json",
        "AF.document.AFRDG00005.n0000.json",
        "CIF.document.XCTFMB030A.n0000_translated_en.json",
        "OEP.document.i00000231.n0000.json",
        "Sabin.document.9869.10352.json",
        "CPR.document.i00003835.n0000.json",
        "Sabin.document.2524.placeholder",
        "UNCDB.document.1.1",
        "UNCDB.document.2.2.json",
    ]
    accepted = [f for f in file_stems if f not in removed]

    filter_result = filter_document_batch(
        file_stems=file_stems,
        spec=ClassifierSpec(
            wikibase_id=WikibaseID("Q788"),
            classifier_id="bvaw9xxm",
            wandb_registry_version="v13",
            dont_run_on=dont_run_on,
        ),
    )
    assert filter_result.removed == removed
    assert filter_result.accepted == accepted


def test_log_client_error():
    error = ClientError(
        error_response={  # pyright: ignore
            "Error": {
                "Code": "RequestTimeTooSkewed",
                "Message": "The difference between the request time and the current time is too large.",
                "RequestTime": "20250922T154936Z",
                "ServerTime": "2025-09-22T16:09:37Z",
                "MaxAllowedSkewMilliseconds": "900000",
            },
        },
        operation_name="GetObject",
    )

    extra_context = parse_client_error_details(error)
    assert extra_context
    assert "Request-Server time discrepancy" in extra_context
    assert "skew.seconds=1201" in extra_context


def test_text_block_inference_span_validation_missing_timestamps():
    mock_classifier = MagicMock()

    spans_missing_timestamps = [
        Span(
            text="test text",
            start_index=0,
            end_index=4,
            labellers=["test_labeller"],
            timestamps=[],  # Missing timestamps
        )
    ]
    mock_classifier.predict.return_value = spans_missing_timestamps

    classifier_spec = ClassifierSpec(
        wikibase_id=WikibaseID("Q123"),
        classifier_id=ClassifierID("testabcd"),
        wandb_registry_version="v1",
    )

    with pytest.raises(ValueError, match="Found 1 span\\(s\\) with missing timestamps"):
        text_block_inference(
            classifier=mock_classifier,
            classifier_spec=classifier_spec,
            block_id="test_block",
            text="test text",
        )


def test_text_block_inference_span_validation_missing_labellers():
    mock_classifier = MagicMock()

    # Create a mock span to bypass Pydantic validation during construction
    mock_span_missing_labellers = MagicMock(spec=Span)
    mock_span_missing_labellers.timestamps = [datetime.now()]
    mock_span_missing_labellers.labellers = []  # Missing labellers
    spans_missing_labellers = [mock_span_missing_labellers]
    mock_classifier.predict.return_value = spans_missing_labellers

    classifier_spec = ClassifierSpec(
        wikibase_id=WikibaseID("Q123"),
        classifier_id=ClassifierID("testabcd"),
        wandb_registry_version="v1",
    )

    with pytest.raises(ValueError, match="Found 1 span\\(s\\) with missing labellers"):
        text_block_inference(
            classifier=mock_classifier,
            classifier_spec=classifier_spec,
            block_id="test_block",
            text="test text",
        )


def test_text_block_inference_span_validation_mismatched_lengths():
    mock_classifier = MagicMock()

    mock_span_mismatched = MagicMock(spec=Span)
    mock_span_mismatched.timestamps = [datetime.now()]  # 1 timestamp
    mock_span_mismatched.labellers = ["labeller1", "labeller2"]  # 2 labellers
    spans_mismatched_lengths = [mock_span_mismatched]
    mock_classifier.predict.return_value = spans_mismatched_lengths

    classifier_spec = ClassifierSpec(
        wikibase_id=WikibaseID("Q123"),
        classifier_id=ClassifierID("testabcd"),
        wandb_registry_version="v1",
    )

    with pytest.raises(
        ValueError,
        match="Found 1 span\\(s\\) with mismatched timestamp/labeller lengths",
    ):
        text_block_inference(
            classifier=mock_classifier,
            classifier_spec=classifier_spec,
            block_id="test_block",
            text="test text",
        )


def test_text_block_inference_span_validation_valid_spans():
    mock_classifier = MagicMock()

    valid_spans = [
        Span(
            text="test text",
            start_index=0,
            end_index=4,
            labellers=["labeller1"],
            timestamps=[datetime.now()],
        )
    ]
    mock_classifier.predict.return_value = valid_spans

    classifier_spec = ClassifierSpec(
        wikibase_id=WikibaseID("Q123"),
        classifier_id=ClassifierID("testabcd"),
        wandb_registry_version="v1",
    )

    # This should not raise any exceptions
    result = text_block_inference(
        classifier=mock_classifier,
        classifier_spec=classifier_spec,
        block_id="test_block",
        text="test text",
    )

    assert result.id == "test_block"
    assert result.text == "test text"
    assert len(result.spans) == 1


@pytest.mark.asyncio
async def test_store_metadata(
    test_config,
    mock_async_bucket,
    mock_s3_async_client,
    snapshot,
):
    """Test that store_metadata correctly builds S3 URI and stores metadata."""
    mock_tags = ["tag:value1", "sha:abc123", "branch:main"]
    mock_run_output_id = "2025-01-15T10:30-test-flow-run"

    # Create a real FlowRun object with proper data
    flow_run = FlowRun(
        id=uuid.UUID("0199bef8-7e41-7afc-9b4c-d3abd406be84"),
        flow_id=uuid.UUID("b213352f-3214-48e3-8f5d-ec19959cb28e"),
        name="test-flow-run",
        state=Running(),
        tags=mock_tags,
    )

    mock_context = MagicMock(spec=FlowRunContext)
    mock_context.flow_run = flow_run

    classifier_specs = [
        ClassifierSpec(
            concept_id=ConceptID("xyz78abc"),
            wikibase_id=WikibaseID("Q788"),
            classifier_id="abcd2345",
            wandb_registry_version="v1",
        )
    ]

    # Mock only the Prefect context, let moto handle S3
    with (
        patch("flows.inference.get_run_context", return_value=mock_context),
        patch(
            "flows.inference.build_run_output_identifier",
            return_value=mock_run_output_id,
        ),
    ):
        await store_metadata(
            config=test_config,
            classifier_specs=classifier_specs,
            run_output_identifier=mock_run_output_id,
        )

    expected_key = os.path.join(
        test_config.inference_document_target_prefix,
        mock_run_output_id,
        "metadata.json",
    )

    response = await mock_s3_async_client.head_object(
        Bucket=test_config.cache_bucket, Key=expected_key
    )
    assert response["ContentLength"] > 0, (
        f"Expected S3 object {expected_key} to have content"
    )

    response = await mock_s3_async_client.get_object(
        Bucket=test_config.cache_bucket, Key=expected_key
    )
    metadata_content = await response["Body"].read()
    metadata_dict = json.loads(metadata_content.decode("utf-8"))

    metadata = Metadata.model_validate(metadata_dict)
    assert metadata == snapshot


@pytest.mark.asyncio
async def test_store_inference_result(
    test_config,
    mock_async_bucket,
    mock_s3_async_client,
    snapshot,
):
    """Test that store_inference_result correctly builds S3 URI and stores results."""
    mock_run_output_id = "2025-01-15T10:30-test-flow-run"
    start_time = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)

    # Create a real FlowRun object with proper data
    flow_run = FlowRun(
        id=uuid.UUID("0199bef8-7e41-7afc-9b4c-d3abd406be84"),
        flow_id=uuid.UUID("b213352f-3214-48e3-8f5d-ec19959cb28e"),
        name="test-flow-run",
        start_time=start_time,
        state=Running(),
    )

    mock_context = MagicMock(spec=FlowRunContext)
    mock_context.flow_run = flow_run

    # Mock only the Prefect context, let moto handle S3
    with (
        patch("flows.inference.get_run_context", return_value=mock_context),
        patch(
            "flows.inference.build_run_output_identifier",
            return_value=mock_run_output_id,
        ),
    ):
        await store_inference_result(
            config=test_config,
            successful_document_stems=set([DocumentStem("TEST.DOC.1.1")]),
            run_output_identifier=mock_run_output_id,
        )

    expected_key = os.path.join(
        test_config.inference_document_target_prefix,
        mock_run_output_id,
        "results.json",
    )

    response = await mock_s3_async_client.head_object(
        Bucket=test_config.cache_bucket, Key=expected_key
    )
    assert response["ContentLength"] > 0, (
        f"Expected S3 object {expected_key} to have content"
    )

    response = await mock_s3_async_client.get_object(
        Bucket=test_config.cache_bucket, Key=expected_key
    )
    result_content = await response["Body"].read()
    result_dict = json.loads(result_content.decode("utf-8"))

    assert result_dict == snapshot


def test_did_inference_fail() -> None:
    """Test the did_inference_fail function."""

    q100_classifier_spec = ClassifierSpec(
        wikibase_id=WikibaseID("Q100"),
        classifier_id=ClassifierID("aaaa2222"),
        wandb_registry_version="v1",
    )
    q101_classifier_spec = ClassifierSpec(
        wikibase_id=WikibaseID("Q101"),
        classifier_id=ClassifierID("bbbb2222"),
        wandb_registry_version="v1",
    )

    # No batch results or requested documents
    batch_inference_results: list[BatchInferenceResult] = []
    requested_document_stems: set[DocumentStem] = set()
    successful_document_stems: set[DocumentStem] = set()

    inference_run_failed: bool = did_inference_fail(
        batch_inference_results=batch_inference_results,
        requested_document_stems=requested_document_stems,
        successful_document_stems=successful_document_stems,
    )
    assert inference_run_failed is True

    # No batch results but documents were requested
    batch_inference_results = []
    requested_document_stems = {DocumentStem("TEST.executive.1.1")}
    successful_document_stems = set()

    inference_run_failed = did_inference_fail(
        batch_inference_results=batch_inference_results,
        requested_document_stems=requested_document_stems,
        successful_document_stems=successful_document_stems,
    )
    assert inference_run_failed is True

    # No successes in any batches
    batch_inference_results = [
        BatchInferenceResult(
            batch_document_stems=[DocumentStem("TEST.executive.1.1")],
            successful_document_stems=[],
            classifier_spec=q100_classifier_spec,
        ),
        BatchInferenceResult(
            batch_document_stems=[DocumentStem("TEST.executive.1.1")],
            successful_document_stems=[],
            classifier_spec=q101_classifier_spec,
        ),
    ]
    requested_document_stems = {DocumentStem("TEST.executive.1.1")}
    successful_document_stems = set()

    inference_run_failed = did_inference_fail(
        batch_inference_results=batch_inference_results,
        requested_document_stems=requested_document_stems,
        successful_document_stems=successful_document_stems,
    )
    assert inference_run_failed is True

    # Success in only some batches
    batch_inference_results = [
        BatchInferenceResult(
            batch_document_stems=[DocumentStem("TEST.executive.1.1")],
            successful_document_stems=[DocumentStem("TEST.executive.1.1")],
            classifier_spec=q100_classifier_spec,
        ),
        BatchInferenceResult(
            batch_document_stems=[DocumentStem("TEST.executive.1.1")],
            successful_document_stems=[],  # No success
            classifier_spec=q100_classifier_spec,
        ),
    ]
    requested_document_stems = {DocumentStem("TEST.executive.1.1")}
    successful_document_stems = set()

    inference_run_failed = did_inference_fail(
        batch_inference_results=batch_inference_results,
        requested_document_stems=requested_document_stems,
        successful_document_stems=successful_document_stems,
    )
    assert inference_run_failed is True

    # Only some documents successful in all batches
    batch_inference_results = [
        BatchInferenceResult(
            batch_document_stems=[
                DocumentStem("TEST.executive.1.1"),
                DocumentStem("TEST.executive.1.2"),
            ],
            successful_document_stems=[
                DocumentStem("TEST.executive.1.1"),
                DocumentStem("TEST.executive.1.2"),
            ],
            classifier_spec=q100_classifier_spec,
        ),
        BatchInferenceResult(
            batch_document_stems=[
                DocumentStem("TEST.executive.1.1"),
                DocumentStem("TEST.executive.1.2"),
            ],
            successful_document_stems=[
                DocumentStem("TEST.executive.1.1")
            ],  # No success for TEST.executive.1.2
            classifier_spec=q101_classifier_spec,
        ),
    ]
    requested_document_stems = {
        DocumentStem("TEST.executive.1.1"),
        DocumentStem("TEST.executive.1.2"),
    }
    successful_document_stems = {DocumentStem("TEST.executive.1.1")}

    inference_run_failed = did_inference_fail(
        batch_inference_results=batch_inference_results,
        requested_document_stems=requested_document_stems,
        successful_document_stems=successful_document_stems,
    )
    assert inference_run_failed is True

    # Success across all batches
    batch_inference_results = [
        BatchInferenceResult(
            batch_document_stems=[DocumentStem("TEST.executive.1.1")],
            successful_document_stems=[DocumentStem("TEST.executive.1.1")],
            classifier_spec=q100_classifier_spec,
        ),
        BatchInferenceResult(
            batch_document_stems=[DocumentStem("TEST.executive.1.1")],
            successful_document_stems=[DocumentStem("TEST.executive.1.1")],
            classifier_spec=q100_classifier_spec,
        ),
    ]
    requested_document_stems = {DocumentStem("TEST.executive.1.1")}
    successful_document_stems = {DocumentStem("TEST.executive.1.1")}

    inference_run_failed = did_inference_fail(
        batch_inference_results=batch_inference_results,
        requested_document_stems=requested_document_stems,
        successful_document_stems=successful_document_stems,
    )
    assert inference_run_failed is False


def test_gather_successful_document_stems() -> None:
    """Test the gather_successful_document_stems function."""

    # Setup: 5 documents, 2 classifiers, 2 batches
    requested_document_stems = [
        DocumentStem("TEST.executive.1.1"),
        DocumentStem("TEST.executive.2.2"),
        DocumentStem("TEST.executive.3.3"),
        DocumentStem("TEST.executive.4.4"),
        DocumentStem("TEST.executive.5.5"),
    ]

    q100_classifier_spec = ClassifierSpec(
        wikibase_id=WikibaseID("Q100"),
        classifier_id="aaaa2222",
        wandb_registry_version="v1",
    )

    q101_classifier_spec = ClassifierSpec(
        wikibase_id=WikibaseID("Q101"),
        classifier_id="bbbb3333",
        wandb_registry_version="v1",
    )

    parameterised_batches = [
        ParameterisedFlow(
            fn=inference_batch_of_documents_cpu,
            params={
                "batch": requested_document_stems,
                "config_json": {},
                "classifier_spec_json": q100_classifier_spec.model_dump(),
            },
        ),
        ParameterisedFlow(
            fn=inference_batch_of_documents_cpu,
            params={
                "batch": requested_document_stems,
                "config_json": {},
                "classifier_spec_json": q101_classifier_spec.model_dump(),
            },
        ),
    ]

    # No results from any batches
    successful_document_stems: set[DocumentStem] = gather_successful_document_stems(
        parameterised_batches=parameterised_batches,
        requested_document_stems=set(requested_document_stems),
        batch_inference_results=[],  # No results
    )
    assert successful_document_stems == set(), "No results should return an empty set"

    # No documents successful for any batches
    all_failed_batch_1 = BatchInferenceResult(
        batch_document_stems=requested_document_stems,
        successful_document_stems=[],  # No success
        classifier_spec=q100_classifier_spec,
    )
    all_failed_batch_2 = BatchInferenceResult(
        batch_document_stems=requested_document_stems,
        successful_document_stems=[],  # No success
        classifier_spec=q101_classifier_spec,
    )
    successful_document_stems: set[DocumentStem] = gather_successful_document_stems(
        parameterised_batches=parameterised_batches,
        requested_document_stems=set(requested_document_stems),
        batch_inference_results=[all_failed_batch_1, all_failed_batch_2],
    )
    assert successful_document_stems == set(), (
        "All failures should return no successful documents"
    )

    # Only some batch results
    q101_batch_success = BatchInferenceResult(
        batch_document_stems=requested_document_stems,
        successful_document_stems=requested_document_stems,
        classifier_spec=q101_classifier_spec,
    )
    successful_document_stems: set[DocumentStem] = gather_successful_document_stems(
        parameterised_batches=parameterised_batches,
        requested_document_stems=set(requested_document_stems),
        batch_inference_results=[q101_batch_success],  # No results for q100
    )
    assert successful_document_stems == set(), (
        "Only documents that succeeded for all classifiers should be marked as successful"
    )

    # Not all documents successful for all batches
    q100_batch_success = BatchInferenceResult(
        batch_document_stems=requested_document_stems,
        successful_document_stems=requested_document_stems,
        classifier_spec=q100_classifier_spec,
    )
    q101_batch_partial_success = BatchInferenceResult(
        batch_document_stems=requested_document_stems,
        successful_document_stems=requested_document_stems[1:],  # Partial success
        classifier_spec=q101_classifier_spec,
    )
    successful_document_stems: set[DocumentStem] = gather_successful_document_stems(
        parameterised_batches=parameterised_batches,
        requested_document_stems=set(requested_document_stems),
        batch_inference_results=[q100_batch_success, q101_batch_partial_success],
    )
    assert successful_document_stems == set(requested_document_stems[1:]), (
        "Only documents that succeeded for all classifiers should be marked as successful"
    )

    # All documents successful for all batches
    q100_batch_success = BatchInferenceResult(
        batch_document_stems=requested_document_stems,
        successful_document_stems=requested_document_stems,
        classifier_spec=q100_classifier_spec,
    )
    successful_document_stems: set[DocumentStem] = gather_successful_document_stems(
        parameterised_batches=parameterised_batches,
        requested_document_stems=set(requested_document_stems),
        batch_inference_results=[q100_batch_success, q101_batch_success],
    )
    assert successful_document_stems == set(requested_document_stems), (
        "Only documents that succeeded for all classifiers should be marked as successful"
    )


def test_get_inference_fault_metadata() -> None:
    """Test the get_inference_fault_metadata function."""

    metadata_json: dict[str, Any] = get_inference_fault_metadata(
        all_successes=[
            BatchInferenceResult(
                batch_document_stems=[DocumentStem("TEST.executive.1.1")],
                successful_document_stems=[DocumentStem("TEST.executive.1.1")],
                classifier_spec=ClassifierSpec(
                    wikibase_id=WikibaseID("Q100"),
                    classifier_id=ClassifierID("aaaa2222"),
                    wandb_registry_version="v1",
                ),
            ),
        ],
        all_raw_failures=[
            FlowRun(
                id=uuid.UUID("0199bef8-7e41-7afc-9b4c-d3abd406be84"),
                flow_id=uuid.UUID("b213352f-3214-48e3-8f5d-ec19959cb28e"),
                name="test-flow-run",
                state=Completed(),
            ),
            BaseException(),
        ],
        requested_document_stems=set([DocumentStem("TEST.executive.1.1")]),
    )

    assert metadata_json.keys() == {
        "all_successes",
        "all_raw_failures",
        "requested_document_stems",
    }

    # Assert that we can dump the result to a string as this is a requirement of the
    # fault metadata.
    json.dumps(metadata_json)


def test_process_single_document_inference():
    doc_stem = DocumentStem("test_doc")
    classifier_spec = ClassifierSpec(
        wikibase_id=WikibaseID("Q123"),
        classifier_id=ClassifierID("testabcd"),
        wandb_registry_version="v1",
    )
    success_result = SingleDocumentInferenceResult(
        document=None,
        labelled_passages=[],
        document_stem=doc_stem,
        classifier_spec=classifier_spec,
    )

    results: list[
        tuple[DocumentStem, Exception | SingleDocumentInferenceResult] | BaseException
    ] = [
        (doc_stem, success_result),  # success
        (doc_stem, ValueError("test error")),  # failure
        RuntimeError("unknown error"),  # unknown failure
    ]

    successes, failures, unknown_failures = process_single_document_inference(results)
    assert (len(successes), len(failures), len(unknown_failures)) == (1, 1, 1)


@pytest.mark.asyncio
async def test_get_existing_inference_results_empty(
    test_config,
    mock_async_bucket,
    mock_s3_async_client,
):
    # A randomly generated, brand new classifier spec.
    classifier_id = Identifier.generate(
        "".join(
            random.choices(
                string.ascii_letters + string.digits,
                k=16,
            )
        )
    )

    classifier_spec = ClassifierSpec(
        wikibase_id=WikibaseID("Q999"),
        classifier_id=classifier_id,
        wandb_registry_version="v1",
    )

    existing = await get_existing_inference_results(
        config=test_config,
        classifier_spec=classifier_spec,
    )

    assert existing == set()


@pytest.mark.asyncio
async def test_get_existing_inference_results_with_results(
    test_config,
    mock_async_bucket,
    mock_s3_async_client,
):
    classifier_spec = ClassifierSpec(
        wikibase_id=WikibaseID("Q788"),
        classifier_id="testabcd",
        wandb_registry_version="v1",
    )

    # Create some fake result files in S3
    documents = [
        DocumentStem("TEST.doc.1.1"),
        DocumentStem("TEST.doc.2.2"),
        DocumentStem("TEST.doc.3.3"),
    ]

    for doc in documents:
        key = f"{test_config.inference_document_target_prefix}{classifier_spec.wikibase_id}/{classifier_spec.classifier_id}/{doc}.json"
        await mock_s3_async_client.put_object(
            Bucket=test_config.cache_bucket,
            Key=key,
            Body=b'{"test": "data"}',
        )

    existing = await get_existing_inference_results(
        config=test_config,
        classifier_spec=classifier_spec,
    )

    assert existing == set(documents)


@pytest.mark.asyncio
async def test_get_existing_inference_results_different_classifiers(
    test_config, mock_async_bucket, mock_s3_async_client
):
    classifier_spec_1 = ClassifierSpec(
        wikibase_id=WikibaseID("Q788"),
        classifier_id="aaaabbbb",
        wandb_registry_version="v1",
    )
    classifier_spec_2 = ClassifierSpec(
        wikibase_id=WikibaseID("Q789"),
        classifier_id="ccccdddd",
        wandb_registry_version="v1",
    )

    # Create results for classifier 1
    doc1 = DocumentStem("TEST.doc.1.1")
    key1 = f"{test_config.inference_document_target_prefix}{classifier_spec_1.wikibase_id}/{classifier_spec_1.classifier_id}/{doc1}.json"
    await mock_s3_async_client.put_object(
        Bucket=test_config.cache_bucket,
        Key=key1,
        Body=b'{"test": "data"}',
    )

    # Create results for classifier 2
    doc2 = DocumentStem("TEST.doc.2.2")
    key2 = f"{test_config.inference_document_target_prefix}{classifier_spec_2.wikibase_id}/{classifier_spec_2.classifier_id}/{doc2}.json"
    await mock_s3_async_client.put_object(
        Bucket=test_config.cache_bucket,
        Key=key2,
        Body=b'{"test": "data"}',
    )

    # Check that each classifier only sees its own results
    existing_1 = await get_existing_inference_results(
        config=test_config,
        classifier_spec=classifier_spec_1,
    )
    existing_2 = await get_existing_inference_results(
        config=test_config,
        classifier_spec=classifier_spec_2,
    )

    assert existing_1 == {doc1}
    assert existing_2 == {doc2}


@pytest.mark.asyncio
async def test_inference_with_caching_enabled(
    test_config,
    mock_classifiers_dir,
    mock_wandb,
    mock_async_bucket_documents,
    mock_s3_async_client,
    mock_deployment,
):
    input_doc_ids = [
        DocumentImportId(Path(doc_file).stem)
        for doc_file in mock_async_bucket_documents
    ]

    classifier_spec = ClassifierSpec(
        wikibase_id=WikibaseID("Q788"),
        classifier_id="abcd2345",
        wandb_registry_version="v1",
    )

    # First, pre-populate S3 with results for the first document only
    doc_with_cache = input_doc_ids[0]
    key = f"{test_config.inference_document_target_prefix}{classifier_spec.wikibase_id}/{classifier_spec.classifier_id}/{doc_with_cache}.json"
    await mock_s3_async_client.put_object(
        Bucket=test_config.cache_bucket,
        Key=key,
        Body=b'{"test": "cached_data"}',
    )

    # Mock deployment to only expect the document without cache
    doc_without_cache = input_doc_ids[1]
    state = Completed(
        data=BatchInferenceResult(
            batch_document_stems=[doc_without_cache],
            successful_document_stems=[doc_without_cache],
            classifier_spec=classifier_spec,
        )
    )

    test_config.skip_existing_inference_results = True

    with mock_deployment(state) as mock_inference_run_deployment:
        _ = await inference(
            classifier_specs=[classifier_spec],
            document_ids=input_doc_ids,
            config=test_config,
        )

        # Verify only the un-cached document was processed
        mock_inference_run_deployment.assert_called_once()
        call_params = mock_inference_run_deployment.call_args.kwargs["parameters"]
        assert call_params["batch"] == [doc_without_cache]

        # Verify the artifact shows 1 skipped document
        summary_artifact = await Artifact.get("removal-details-sandbox")
        assert summary_artifact and summary_artifact.description
        artifact_data = json.loads(summary_artifact.data)
        assert len(artifact_data) == 1
        assert artifact_data[0]["Cached (Skipped)"] == 1


@pytest.mark.asyncio
async def test_inference_with_caching_disabled(
    test_config,
    mock_classifiers_dir,
    mock_wandb,
    mock_async_bucket_documents,
    mock_s3_async_client,
    mock_deployment,
):
    input_doc_ids = [
        DocumentImportId(Path(doc_file).stem)
        for doc_file in mock_async_bucket_documents
    ]

    classifier_spec = ClassifierSpec(
        wikibase_id=WikibaseID("Q788"),
        classifier_id="efgh6789",
        wandb_registry_version="v1",
    )

    # Pre-populate S3 with results for all documents
    for doc_id in input_doc_ids:
        key = f"{test_config.inference_document_target_prefix}{classifier_spec.wikibase_id}/{classifier_spec.classifier_id}/{doc_id}.json"
        await mock_s3_async_client.put_object(
            Bucket=test_config.cache_bucket,
            Key=key,
            Body=b'{"test": "cached_data"}',
        )

    # Mock deployment to expect all documents
    state = Completed(
        data=BatchInferenceResult(
            batch_document_stems=list(input_doc_ids),
            successful_document_stems=list(input_doc_ids),
            classifier_spec=classifier_spec,
        )
    )

    test_config.skip_existing_inference_results = False

    with mock_deployment(state) as mock_inference_run_deployment:
        _ = await inference(
            classifier_specs=[classifier_spec],
            document_ids=input_doc_ids,
            config=test_config,
        )

        # Verify all documents were processed despite existing results
        mock_inference_run_deployment.assert_called_once()
        call_params = mock_inference_run_deployment.call_args.kwargs["parameters"]
        assert set(call_params["batch"]) == set(input_doc_ids)

        # Verify the artifact shows 0 skipped documents
        summary_artifact = await Artifact.get("removal-details-sandbox")
        assert summary_artifact and summary_artifact.description
        artifact_data = json.loads(summary_artifact.data)
        assert len(artifact_data) == 1
        assert artifact_data[0]["Cached (Skipped)"] == 0
