import asyncio
import json
import os
import tempfile
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pydantic
import pytest
from cpr_sdk.models.search import Concept as VespaConcept
from prefect import flow
from prefect.artifacts import Artifact
from prefect.client.schemas.objects import FlowRun
from prefect.context import FlowRunContext
from prefect.states import Running

from flows.aggregate import (
    AggregationFailure,
    Metadata,
    aggregate_batch_of_documents,
    build_run_output_identifier,
    collect_stems_by_specs,
    get_all_labelled_passages_for_one_document,
    process_document,
    store_metadata,
    validate_passages_are_same_except_concepts,
)
from flows.classifier_specs.spec_interface import ClassifierSpec
from flows.utils import DocumentStem
from knowledge_graph.identifiers import ConceptID, WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.span import Span
from scripts.update_classifier_spec import write_spec_file


@pytest.fixture
def mock_classifier_specs():
    with tempfile.TemporaryDirectory() as spec_dir:
        # Write the concept specs to a YAML file
        temp_spec_dir = Path(spec_dir)
        classifier_specs = [
            ClassifierSpec(
                wikibase_id="Q123",
                classifier_id="g29kcna9",
                wandb_registry_version="v4",
                dont_run_on=["sabin"],
            ),
            ClassifierSpec(
                wikibase_id="Q218",
                classifier_id="6z4pufsm",
                wandb_registry_version="v5",
                dont_run_on=["sabin"],
            ),
            ClassifierSpec(
                wikibase_id="Q223",
                classifier_id="36bhx4mu",
                wandb_registry_version="v3",
                dont_run_on=["sabin"],
            ),
            ClassifierSpec(
                wikibase_id="Q767",
                classifier_id="mgwutbqx",
                wandb_registry_version="v3",
                dont_run_on=["sabin"],
            ),
            ClassifierSpec(
                wikibase_id="Q1286",
                classifier_id="7bt99yeu",
                wandb_registry_version="v3",
                dont_run_on=["sabin", "cclw"],
            ),
        ]
        spec_file_path = temp_spec_dir / "sandbox.yaml"
        write_spec_file(spec_file_path, classifier_specs)
        with patch("flows.classifier_specs.spec_interface.SPEC_DIR", temp_spec_dir):
            yield spec_file_path, classifier_specs


@pytest.mark.asyncio
async def test_aggregate_batch_of_documents(
    mock_bucket_labelled_passages_large, test_config, mock_classifier_specs
):
    _, bucket, s3_async_client = mock_bucket_labelled_passages_large
    classifier_specs: list[ClassifierSpec] = mock_classifier_specs[1]

    document_stems = [
        DocumentStem("CCLW.executive.10061.4515"),
        DocumentStem("CPR.document.i00000549.n0000"),
        DocumentStem("UNFCCC.non-party.467.0"),
        DocumentStem("UNFCCC.party.492.0"),
    ]

    run_reference = await aggregate_batch_of_documents(
        document_stems=document_stems,
        config_json=test_config.model_dump(),
        classifier_specs=classifier_specs,
        run_output_identifier="test-run",
    )

    all_collected_ids = []
    for document_stem in document_stems:
        s3_path = os.path.join(
            test_config.aggregate_inference_results_prefix,
            run_reference,
            f"{document_stem}.json",
        )
        try:
            response = await s3_async_client.get_object(Bucket=bucket, Key=s3_path)
            data = await response["Body"].read()
            data = json.loads(data.decode("utf-8"))
        except s3_async_client.exceptions.NoSuchKey:
            pytest.fail(f"Unable to find output file: {s3_path}")
        except json.JSONDecodeError:
            pytest.fail(f"Unable to deserialise output for: {document_stem}")
        except Exception as e:
            pytest.fail(f"Unexpected error: {e}")

        wikibase_ids = [
            classifier_spec.wikibase_id for classifier_spec in classifier_specs
        ]

        document_inference_output = list(data.values())
        collected_ids_for_document = []
        for concepts in document_inference_output:
            for concept in concepts:
                try:
                    vespa_concept = VespaConcept.model_validate(concept)
                    collected_ids_for_document.append(vespa_concept.id)
                except pydantic.ValidationError as e:
                    pytest.fail(
                        f"Unable to deserialise concept: {concept} with error: {e}"
                    )

        # Q1286 should not be on CCLW.executive.10061.4515 because of dont_run_on
        if "CCLW" in document_stem:
            assert "Q1286" not in collected_ids_for_document
        assert len(collected_ids_for_document) > 0, (
            f"No concepts found for document: {document_stem}"
        )
        all_collected_ids.extend(collected_ids_for_document)

    assert set(all_collected_ids) == set(wikibase_ids), (
        f"Outputted: {set(all_collected_ids)} which doesnt match those in the specs: {set(wikibase_ids)}"
    )
    COUNT = 139
    assert len(all_collected_ids) == COUNT, (
        f"Expected {COUNT} concepts to be outputted, found: {len(all_collected_ids)}"
    )

    summary_artifact = await Artifact.get("aggregate-inference-sandbox")
    assert summary_artifact and summary_artifact.description
    assert summary_artifact.data == "[]"


@pytest.mark.asyncio
async def test_aggregate_batch_of_documents__with_failures(
    mock_bucket_labelled_passages_large, mock_classifier_specs, test_config
):
    _, classifier_specs = mock_classifier_specs
    expect_failure_stems = [
        DocumentStem("CCLW.Made.Up.Document.ID"),
        DocumentStem("OEP.One.That.Should.Fail"),
    ]
    document_stems = [DocumentStem("CCLW.executive.10061.4515")] + expect_failure_stems

    with pytest.raises(ValueError):
        await aggregate_batch_of_documents(
            document_stems=document_stems,
            config_json=test_config.model_dump(),
            classifier_specs=classifier_specs,
            run_output_identifier="test-run",
        )

    summary_artifact = await Artifact.get("aggregate-inference-sandbox")
    assert summary_artifact and summary_artifact.description
    artifact_data = json.loads(summary_artifact.data)
    failure_stems = [f["Failed document Stem"] for f in artifact_data]
    assert set(failure_stems) == set(expect_failure_stems)
    assert "NoSuchKey" in artifact_data[0]["Exception"]
    assert "NoSuchKey" in artifact_data[1]["Exception"]


def test_build_run_output_prefix():
    @flow()
    def fake_flow():
        return build_run_output_identifier()

    prefix = fake_flow()
    # From https://docs.aws.amazon.com/AmazonS3/latest/userguide/object-keys.html
    s3_unsupported_chars = r'\^`>{[<%#"}]~|'

    for char in s3_unsupported_chars:
        assert char not in prefix, f"Unsupported char found in prefix: {prefix}"

    assert "None" not in prefix


@pytest.mark.asyncio
async def test_get_all_labelled_passages_for_one_document(
    mock_bucket_labelled_passages_large, test_config
):
    _, _, s3_async_client = mock_bucket_labelled_passages_large
    document_stem = DocumentStem("CCLW.executive.10061.4515")
    classifier_specs = [
        ClassifierSpec(
            wikibase_id="Q218", classifier_id="6z4pufsm", wandb_registry_version="v5"
        ),
        ClassifierSpec(
            wikibase_id="Q767", classifier_id="mgwutbqx", wandb_registry_version="v3"
        ),
        ClassifierSpec(
            wikibase_id="Q1286", classifier_id="7bt99yeu", wandb_registry_version="v3"
        ),
        ClassifierSpec(
            wikibase_id="Q1400",
            classifier_id="7bt99yeu",
            wandb_registry_version="v3",
            dont_run_on=["cclw"],
        ),
    ]
    all_labelled_passages = []
    async for spec, labelled_passages in get_all_labelled_passages_for_one_document(
        s3_async_client, document_stem, classifier_specs, test_config
    ):
        all_labelled_passages.append(labelled_passages)
    assert len(all_labelled_passages) == 3


def test_validate_passages_are_same_except_concepts():
    passages: list[LabelledPassage] = []
    for i in range(10):
        span_one = Span(text=f"unique spans: {i}", start_index=0, end_index=4)
        span_two = Span(text=f"unique spans: {i} {i}", start_index=0, end_index=4)
        passage = LabelledPassage(
            id="1",
            text="id and text should match across identical labelled passages",
            spans=[span_one, span_two],
        )
        passages.append(passage)

    validate_passages_are_same_except_concepts(passages)

    with pytest.raises(ValueError):
        passages.append(
            LabelledPassage(
                id="2",
                text="Imagine if we messed up and drew in a different passage",
                spans=[Span(text="span", start_index=0, end_index=4)],
            )
        )
        validate_passages_are_same_except_concepts(passages)


@pytest.mark.asyncio
async def test_process_single_document__success(
    mock_bucket_labelled_passages_large, mock_classifier_specs, test_config
):
    _, bucket, s3_async_client = mock_bucket_labelled_passages_large
    _, classifier_specs = mock_classifier_specs
    document_stem = DocumentStem("CCLW.executive.10061.4515")

    assert document_stem == await process_document.fn(
        document_stem,
        classifier_specs,
        test_config,
        "run_output_identifier",
    )

    # Check that the file is in the run_output_identifier prefix with head object
    response = await s3_async_client.head_object(
        Bucket=bucket,
        Key=os.path.join(
            test_config.aggregate_inference_results_prefix,
            "run_output_identifier",
            f"{document_stem}.json",
        ),
    )
    assert response["ContentLength"] > 0

    # Check that the file is in the latest prefix with head object
    response = await s3_async_client.head_object(
        Bucket=bucket,
        Key=os.path.join(
            test_config.aggregate_inference_results_prefix,
            "latest",
            f"{document_stem}.json",
        ),
    )
    assert response["ContentLength"] > 0


@pytest.mark.asyncio
async def test_process_single_document__client_error(
    mock_bucket_labelled_passages_large, mock_classifier_specs, test_config
):
    document_stem = DocumentStem("CCLW.executive.10061.4515")
    spec_file_path, classifier_specs = mock_classifier_specs
    classifier_specs.append(
        ClassifierSpec(
            wikibase_id="Q9999999999",
            classifier_id="zzzzzzzz",
            wandb_registry_version="v99",
        )
    )
    write_spec_file(spec_file_path, classifier_specs)

    result = await asyncio.gather(
        process_document.fn(
            document_stem,
            classifier_specs,
            test_config,
            "run_output_identifier",
        ),
        return_exceptions=True,
    )
    assert len(result) == 1
    assert isinstance(result[0], AggregationFailure)
    code = result[0].exception.response["Error"]["Code"]
    assert code == "NoSuchKey"
    key = result[0].exception.response["Error"]["Key"]
    assert (
        key == "labelled_passages/Q9999999999/zzzzzzzz/CCLW.executive.10061.4515.json"
    )


@pytest.mark.asyncio
async def test_process_single_document__value_error(
    mock_bucket_labelled_passages_large,
    mock_classifier_specs,
    test_config,
    snapshot,
):
    keys, bucket, s3_async_client = mock_bucket_labelled_passages_large
    _, classifier_specs = mock_classifier_specs

    # Use Q223 and Q767 to create mismatch.
    #
    # Those classifiers are setup in the fixtures.
    classifier_specs = [
        spec for spec in classifier_specs if spec.wikibase_id in ["Q223", "Q767"]
    ]

    document_stem = DocumentStem("CCLW.executive.10061.4515")

    # Replace Q767 with fewer passages to create mismatch with Q223
    # (which has 27).
    new_data_short = [
        '{"id":"b1","text":"Some words","spans":[],"metadata":{}}',
        '{"id":"b2","text":"But not enough words","spans":[],"metadata":{}}',
    ]

    # Find and replace the Q767 file to have only 2 passages, while
    # Q223 keeps its 27.
    q767_key = [
        k for k in keys if "CCLW.executive.10061.4515" in k and "Q767/mgwutbqx" in k
    ][0]
    await s3_async_client.put_object(
        Bucket=bucket,
        Key=q767_key,
        Body=json.dumps(new_data_short),
    )

    async with asyncio.timeout(5):
        result = await asyncio.gather(
            process_document.fn(
                document_stem,
                classifier_specs,
                test_config,
                "run_output_identifier",
            ),
            return_exceptions=True,
        )
    assert result == snapshot


@pytest.mark.asyncio
async def test_collect_stems_by_specs(
    mock_classifier_specs, mock_bucket_labelled_passages_large, test_config
):
    stems = await collect_stems_by_specs(test_config)
    assert set(stems) == set(
        [
            "UNFCCC.non-party.467.0",
            "CCLW.executive.10061.4515",
            "AF.document.i00000021.n0000_translated_en",
            "UNFCCC.party.492.0",
            "CPR.document.i00000549.n0000",
        ]
    )


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
    with patch("flows.aggregate.get_run_context", return_value=mock_context):
        await store_metadata(
            config=test_config,
            classifier_specs=classifier_specs,
            run_output_identifier=mock_run_output_id,
        )

    expected_key = os.path.join(
        test_config.aggregate_inference_results_prefix,
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
