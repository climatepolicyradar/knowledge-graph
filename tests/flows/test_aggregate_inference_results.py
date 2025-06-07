import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import aioboto3
import pydantic
import pytest
from cpr_sdk.models.search import Concept as VespaConcept
from prefect import flow
from prefect.artifacts import Artifact

from flows.aggregate_inference_results import (
    AggregationFailure,
    aggregate_inference_results,
    build_run_output_identifier,
    collect_stems_by_specs,
    get_all_labelled_passages_for_one_document,
    process_single_document,
    validate_passages_are_same_except_concepts,
)
from scripts.cloud import ClassifierSpec
from scripts.update_classifier_spec import write_spec_file
from src.labelled_passage import LabelledPassage
from src.span import Span


@pytest.fixture
def mock_classifier_specs():
    with tempfile.TemporaryDirectory() as spec_dir:
        # Write the concept specs to a YAML file
        temp_spec_dir = Path(spec_dir)
        classifier_specs = [
            ClassifierSpec(name="Q123", alias="v4"),
            ClassifierSpec(name="Q223", alias="v3"),
            ClassifierSpec(name="Q218", alias="v5"),
            ClassifierSpec(name="Q767", alias="v3"),
            ClassifierSpec(name="Q1286", alias="v3"),
        ]
        spec_file_path = temp_spec_dir / "sandbox.yaml"
        write_spec_file(spec_file_path, classifier_specs)

        with patch("scripts.update_classifier_spec.SPEC_DIR", temp_spec_dir):
            yield spec_file_path, classifier_specs


@pytest.mark.asyncio
async def test_aggregate_inference_results(
    mock_bucket_labelled_passages_large, test_aggregate_config, mock_classifier_specs
):
    _, bucket, s3_async_client = mock_bucket_labelled_passages_large
    _, classifier_specs = mock_classifier_specs

    document_ids = [
        "CCLW.executive.10061.4515",
        "CPR.document.i00000549.n0000",
        "UNFCCC.non-party.467.0",
        "UNFCCC.party.492.0",
    ]

    run_reference = await aggregate_inference_results(
        document_ids, test_aggregate_config
    )

    all_collected_ids = []
    collected_ids_for_document = []

    for document_id in document_ids:
        s3_path = os.path.join(
            test_aggregate_config.aggregate_inference_results_prefix,
            run_reference,
            f"{document_id}.json",
        )
        try:
            response = await s3_async_client.get_object(Bucket=bucket, Key=s3_path)
            data = await response["Body"].read()
            data = json.loads(data.decode("utf-8"))
        except s3_async_client.exceptions.NoSuchKey:
            pytest.fail(f"Unable to find output file: {s3_path}")
        except json.JSONDecodeError:
            pytest.fail(f"Unable to deserialise output for: {document_id}")
        except Exception as e:
            pytest.fail(f"Unexpected error: {e}")

        wikibase_ids = [classifier_spec.name for classifier_spec in classifier_specs]

        document_inference_output = list(data.values())
        for concepts in document_inference_output:
            for concept in concepts:
                try:
                    vespa_concept = VespaConcept.model_validate(concept)
                    collected_ids_for_document.append(vespa_concept.id)
                except pydantic.ValidationError as e:
                    pytest.fail(
                        f"Unable to deserialise concept: {concept} with error: {e}"
                    )

        assert len(collected_ids_for_document) > 0, (
            f"No concepts found for document: {document_id}"
        )
        all_collected_ids.extend(collected_ids_for_document)

    assert set(all_collected_ids) == set(wikibase_ids), (
        f"Outputted: {set(all_collected_ids)} which doesnt match those in the specs: {set(wikibase_ids)}"
    )
    COUNT = 329
    assert len(all_collected_ids) == COUNT, (
        f"Expected {COUNT} concepts to be outputted, found: {len(all_collected_ids)}"
    )

    summary_artifact = await Artifact.get("aggregate-inference-sandbox")
    assert summary_artifact and summary_artifact.description
    assert summary_artifact.data == "[]"


@pytest.mark.asyncio
async def test_aggregate_inference_results__with_failures(
    mock_bucket_labelled_passages_large, mock_classifier_specs, test_aggregate_config
):
    expect_failure_ids = ["Some.Made.Up.Document.ID", "Another.One.That.Should.Fail"]
    document_ids = ["CCLW.executive.10061.4515"] + expect_failure_ids

    with pytest.raises(ValueError):
        await aggregate_inference_results(document_ids, test_aggregate_config)

    summary_artifact = await Artifact.get("aggregate-inference-sandbox")
    assert summary_artifact and summary_artifact.description
    artifact_data = json.loads(summary_artifact.data)
    failured_ids = [f["Failed document ID"] for f in artifact_data]
    assert set(failured_ids) == set(expect_failure_ids)
    assert artifact_data[0]["Context"]["Error"]["Code"] == "NoSuchKey"
    assert artifact_data[1]["Context"]["Error"]["Code"] == "NoSuchKey"


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
    mock_bucket_labelled_passages_large, test_aggregate_config
):
    _, _, s3_async_client = mock_bucket_labelled_passages_large
    document_id = "CCLW.executive.10061.4515"
    classifier_specs = [
        ClassifierSpec(name="Q218", alias="v5"),
        ClassifierSpec(name="Q767", alias="v3"),
        ClassifierSpec(name="Q1286", alias="v3"),
    ]
    all_labelled_passages = []
    async for spec, labelled_passages in get_all_labelled_passages_for_one_document(
        s3_async_client, document_id, classifier_specs, test_aggregate_config
    ):
        all_labelled_passages.append(labelled_passages)
    assert len(all_labelled_passages) == len(classifier_specs)


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
    mock_bucket_labelled_passages_large, mock_classifier_specs, test_aggregate_config
):
    _, bucket, s3_async_client = mock_bucket_labelled_passages_large
    _, classifier_specs = mock_classifier_specs
    document_id = "CCLW.executive.10061.4515"

    session = aioboto3.Session(region_name=test_aggregate_config.bucket_region)
    assert document_id == await process_single_document(
        session,
        document_id,
        classifier_specs,
        test_aggregate_config,
        "run_output_identifier",
    )

    # Check that the file is in the run_output_identifier prefix with head object
    response = await s3_async_client.head_object(
        Bucket=bucket,
        Key=os.path.join(
            test_aggregate_config.aggregate_inference_results_prefix,
            "run_output_identifier",
            f"{document_id}.json",
        ),
    )
    assert response["ContentLength"] > 0

    # Check that the file is in the latest prefix with head object
    response = await s3_async_client.head_object(
        Bucket=bucket,
        Key=os.path.join(
            test_aggregate_config.aggregate_inference_results_prefix,
            "latest",
            f"{document_id}.json",
        ),
    )
    assert response["ContentLength"] > 0


@pytest.mark.asyncio
async def test_process_single_document__client_error(
    mock_bucket_labelled_passages_large, mock_classifier_specs, test_aggregate_config
):
    document_id = "CCLW.executive.10061.4515"
    spec_file_path, classifier_specs = mock_classifier_specs
    classifier_specs.append(ClassifierSpec(name="Q9999999999", alias="v99"))
    write_spec_file(spec_file_path, classifier_specs)

    session = aioboto3.Session(region_name=test_aggregate_config.bucket_region)
    result = await asyncio.gather(
        process_single_document(
            session,
            document_id,
            classifier_specs,
            test_aggregate_config,
            "run_output_identifier",
        ),
        return_exceptions=True,
    )
    assert len(result) == 1
    assert isinstance(result[0], AggregationFailure)
    code = result[0].exception.response["Error"]["Code"]
    assert code == "NoSuchKey"
    key = result[0].exception.response["Error"]["Key"]
    assert key == "labelled_passages/Q9999999999/v99/CCLW.executive.10061.4515.json"


@pytest.mark.asyncio
async def test_process_single_document__value_error(
    mock_bucket_labelled_passages_large, mock_classifier_specs, test_aggregate_config
):
    keys, bucket, s3_async_client = mock_bucket_labelled_passages_large
    _, classifier_specs = mock_classifier_specs

    document_id = "CCLW.executive.10061.4515"

    # Replace the doc with a broken  one
    new_data = [
        '{"id":"b1","text":"Some words","spans":[],"metadata":{}}',
        '{"id":"b2","text":"But not enough words","spans":[],"metadata":{}}',
    ]
    key = [k for k in keys if "CCLW.executive.10061.4515" in k][0]
    await s3_async_client.put_object(
        Bucket=bucket,
        Key=key,
        Body=json.dumps(new_data),
    )

    session = aioboto3.Session(region_name=test_aggregate_config.bucket_region)
    result = await asyncio.gather(
        process_single_document(
            session,
            document_id,
            classifier_specs,
            test_aggregate_config,
            "run_output_identifier",
        ),
        return_exceptions=True,
    )
    assert len(result) == 1
    assert isinstance(result[0], AggregationFailure)
    assert isinstance(result[0].exception, ValueError)
    assert (
        result[0]
        .exception.args[0]
        .startswith("The number of passages diverge when appending")
    )
    assert (
        result[0]
        .exception.args[0]
        .endswith("len(labelled_passages)=2 != len(concepts_for_vespa)=27")
    )


def test_collect_stems_by_specs(
    mock_classifier_specs, mock_bucket_labelled_passages_large, test_aggregate_config
):
    stems = collect_stems_by_specs(test_aggregate_config)
    assert set(stems) == set(
        [
            "UNFCCC.non-party.467.0",
            "CCLW.executive.10061.4515",
            "AF.document.i00000021.n0000_translated_en",
            "UNFCCC.party.492.0",
            "CPR.document.i00000549.n0000",
        ]
    )
