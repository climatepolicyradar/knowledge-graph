import asyncio
import json
import os
import tempfile
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pydantic
import pytest
from cpr_sdk.models.search import Passage as VespaPassage
from prefect import flow
from prefect.artifacts import Artifact
from prefect.client.schemas.objects import FlowRun
from prefect.context import FlowRunContext
from prefect.states import Running

from flows.aggregate import (
    AggregateResult,
    AggregationFailure,
    Metadata,
    MiniClassifierSpec,
    _document_stems_from_parameters,
    aggregate,
    aggregate_batch_of_documents,
    build_run_output_identifier,
    collect_stems_by_specs,
    convert_labelled_passage_to_concepts,
    get_all_labelled_passages_for_one_document,
    get_model_from_span,
    get_parent_concepts_from_concept,
    parse_model_field,
    process_document,
    store_metadata,
    validate_passages_are_same_except_concepts,
)
from flows.classifier_specs.spec_interface import ClassifierSpec
from flows.config import Config
from flows.utils import DocumentStem, build_inference_result_s3_uri
from knowledge_graph.cloud import AwsEnv
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import ClassifierID, ConceptID, WikibaseID
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
async def test_aggregate_dont_allow_prod_classifier_specs(test_config):
    test_config.aws_env = AwsEnv.production

    with pytest.raises(
        ValueError, match="in production you must use the full classifier specs. list"
    ):
        _ = await aggregate(
            config=test_config,
            classifier_specs=[
                ClassifierSpec(
                    wikibase_id="Q1286",
                    classifier_id="7bt99yeu",
                    wandb_registry_version="v3",
                    dont_run_on=["sabin", "cclw"],
                ),
            ],
        )


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

    flow_run = FlowRun(
        id=uuid.UUID("0199bef8-7e41-7afc-9b4c-d3abd406be84"),
        flow_id=uuid.UUID("b213352f-3214-48e3-8f5d-ec19959cb28e"),
        name="test-flow-run",
        state=Running(),
    )

    mock_context = MagicMock(spec=FlowRunContext)
    mock_context.flow_run = flow_run

    with (
        patch("flows.aggregate.get_run_context", return_value=mock_context),
    ):
        aggregate_result = await aggregate_batch_of_documents(
            document_stems=document_stems,
            config_json=test_config.model_dump(),
            classifier_specs=classifier_specs,
            run_output_identifier="test-run",
        )

        if isinstance(aggregate_result, AggregateResult):
            run_reference = aggregate_result.RunOutputIdentifier

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
                    vespa_concept = VespaPassage.Concept.model_validate(concept)
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

    summary_artifact = await Artifact.get("batch-aggregate-sandbox-test-flow-run")
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

    flow_run = FlowRun(
        id=uuid.UUID("0199bef8-7e41-7afc-9b4c-d3abd406be84"),
        flow_id=uuid.UUID("b213352f-3214-48e3-8f5d-ec19959cb28e"),
        name="test-flow-run",
        state=Running(),
    )

    mock_context = MagicMock(spec=FlowRunContext)
    mock_context.flow_run = flow_run

    with (
        patch("flows.aggregate.get_run_context", return_value=mock_context),
    ):
        await aggregate_batch_of_documents(
            document_stems=document_stems,
            config_json=test_config.model_dump(),
            classifier_specs=classifier_specs,
            run_output_identifier="test-run",
        )

    summary_artifact = await Artifact.get("batch-aggregate-sandbox-test-flow-run")
    assert summary_artifact and summary_artifact.description
    artifact_data = json.loads(summary_artifact.data)
    failure_stems = [f["Failed document Stem"] for f in artifact_data]
    assert set(failure_stems) == set(expect_failure_stems)
    assert "NoSuchKey" in artifact_data[0]["Exception"]
    assert "NoSuchKey" in artifact_data[1]["Exception"]


@pytest.mark.asyncio
async def test_aggregate_batch_of_documents__returns_aggregate_result_containing_errors(
    mock_bucket_labelled_passages_large, mock_classifier_specs, test_config
):
    _, classifier_specs = mock_classifier_specs
    expect_failure_stems = [
        DocumentStem("CCLW.Made.Up.Document.ID"),
        DocumentStem("OEP.One.That.Should.Fail"),
    ]
    document_stems = [DocumentStem("CCLW.executive.10061.4515")] + expect_failure_stems

    flow_run = FlowRun(
        id=uuid.UUID("0199bef8-7e41-7afc-9b4c-d3abd406be84"),
        flow_id=uuid.UUID("b213352f-3214-48e3-8f5d-ec19959cb28e"),
        name="test-flow-run",
        state=Running(),
    )

    mock_context = MagicMock(spec=FlowRunContext)
    mock_context.flow_run = flow_run

    aggregate_result = None

    with (
        patch("flows.aggregate.get_run_context", return_value=mock_context),
    ):
        aggregate_result = await aggregate_batch_of_documents(
            document_stems=document_stems,
            config_json=test_config.model_dump(),
            classifier_specs=classifier_specs,
            run_output_identifier="test-run",
        )

    assert aggregate_result is not None


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


@pytest.mark.parametrize(
    ("model,expected"),
    [
        (
            "Q123:xabs2345:abcd2345",
            MiniClassifierSpec(
                wikibase_id=WikibaseID(
                    "Q123",
                ),
                concept_id=ConceptID("xabs2345"),
                classifier_id=ClassifierID("abcd2345"),
            ),
        ),
        (
            "Q123:None:abcd2345",
            None,
        ),
        (
            'KeywordClassifier("concept_38")',
            None,
        ),
    ],
)
def test_parse_model_field(model, expected):
    assert parse_model_field(model) == expected


def test_get_model_from_span():
    assert "KeywordClassifier" == get_model_from_span(
        span=Span(
            text="Test text.",
            start_index=0,
            end_index=8,
            concept_id=None,
            labellers=["KeywordClassifier"],
            timestamps=[datetime.now()],
        ),
        classifier_spec=None,
    )


def test_get_model_from_span_checks_length():
    with pytest.raises(ValueError, match="Span should have 1 labeller but has 0"):
        _ = get_model_from_span(
            span=Span(
                text="Test text.",
                start_index=0,
                end_index=8,
                concept_id=None,
                labellers=[],
            ),
            classifier_spec=None,
        )


def test_get_model_from_span_with_classifier_spec():
    """Test that get_model_from_span uses classifier_spec when provided."""
    classifier_spec = ClassifierSpec(
        concept_id=ConceptID("abcd2345"),
        wikibase_id=WikibaseID("Q123"),
        classifier_id=ClassifierID("xyz78abc"),
        wandb_registry_version="v1",
    )

    span = Span(
        text="Test text.",
        start_index=0,
        end_index=8,
        concept_id="Q123",
        labellers=["KeywordClassifier"],
        timestamps=[datetime.now()],
    )

    result = get_model_from_span(span=span, classifier_spec=classifier_spec)
    assert result == "Q123:abcd2345:xyz78abc"


def test_get_parent_concepts_from_concept() -> None:
    """Test that we can correctly retrieve the parent concepts from a concept."""
    assert get_parent_concepts_from_concept(
        concept=Concept(
            preferred_label="forestry sector",
            alternative_labels=[],
            negative_labels=[],
            wikibase_id=WikibaseID("Q10014"),
            subconcept_of=[WikibaseID("Q4470")],
            has_subconcept=[WikibaseID("Q4471")],
            labelled_passages=[],
        )
    ) == ([{"id": "Q4470", "name": ""}], "Q4470,")


def test_convert_labelled_passges_to_concepts(
    example_labelled_passages: list[LabelledPassage],
) -> None:
    """Test that we can correctly convert labelled passages to concepts."""
    concepts = convert_labelled_passage_to_concepts(example_labelled_passages[0])
    assert all([isinstance(concept, VespaPassage.Concept) for concept in concepts])


def test_convert_labelled_passges_to_concepts_skips_invalid_spans(
    example_labelled_passages: list[LabelledPassage],
) -> None:
    """Test that we ignore a Span has no concept ID or timestamps."""
    # Add a span without concept_id
    example_labelled_passages[0].spans.append(
        Span(
            text="Test text.",
            start_index=0,
            end_index=8,
            concept_id=None,
            labellers=[],
        )
    )

    concepts = convert_labelled_passage_to_concepts(example_labelled_passages[0])

    # Verify that the problematic spans were skipped but valid one was processed
    assert len(concepts) == 1


def test_convert_labelled_passage_to_concepts_with_classifier_spec_in_metadata(
    example_labelled_passages: list[LabelledPassage],
) -> None:
    """Test that convert_labelled_passage_to_concepts uses classifier_spec from metadata."""
    # Create a ClassifierSpec and serialize it like the inference code does
    classifier_spec = ClassifierSpec(
        concept_id=ConceptID("xyz78abc"),
        wikibase_id=WikibaseID("Q1363"),
        classifier_id=ClassifierID("vax7e3n7"),
        wandb_registry_version="v13",
    )

    # Add classifier_spec to the metadata using model_dump() like inference does
    example_labelled_passages[0].metadata["classifier_spec"] = (
        classifier_spec.model_dump()
    )

    concepts = convert_labelled_passage_to_concepts(example_labelled_passages[0])

    assert len(concepts) == 1
    # Should use classifier_spec format instead of span.labellers
    assert concepts[0].model == "Q1363:xyz78abc:vax7e3n7"


@pytest.mark.asyncio
async def test__document_stems_from_parameters_neither(
    test_config: Config,
    mock_classifier_specs,
    mock_bucket_labelled_passages_large,
):
    actual = await _document_stems_from_parameters(
        config=test_config,
        document_stems=None,
        run_output_identifier=None,
    )

    assert set(actual) == set(
        [
            "UNFCCC.non-party.467.0",
            "CCLW.executive.10061.4515",
            "AF.document.i00000021.n0000_translated_en",
            "UNFCCC.party.492.0",
            "CPR.document.i00000549.n0000",
        ]
    )


@pytest.mark.asyncio
async def test__document_stems_from_parameters_both(test_config: Config):
    run_output_identifier = "2025-01-15T10:30-test-flow-run"

    with pytest.raises(
        ValueError,
        match="only one of document_stems and run_output_identifier can be used",
    ):
        _ = await _document_stems_from_parameters(
            config=test_config,
            document_stems=[
                DocumentStem("UNFCCC.non-party.467.0"),
                DocumentStem("CCLW.executive.10061.4515"),
                DocumentStem("AF.document.i00000021.n0000_translated_en"),
                DocumentStem("UNFCCC.party.492.0"),
                DocumentStem("CPR.document.i00000549.n0000"),
            ],
            run_output_identifier=run_output_identifier,
        )


@pytest.mark.asyncio
async def test__document_stems_from_parameters_document_stems(
    test_config: Config,
    mock_classifier_specs,
    mock_bucket_labelled_passages_large,
):
    actual = await _document_stems_from_parameters(
        config=test_config,
        document_stems=[
            DocumentStem("UNFCCC.non-party.467.0"),
            DocumentStem("CCLW.executive.10061.4515"),
            DocumentStem("AF.document.i00000021.n0000_translated_en"),
            DocumentStem("UNFCCC.party.492.0"),
            DocumentStem("CPR.document.i00000549.n0000"),
        ],
        run_output_identifier=None,
    )

    assert set(actual) == set(
        [
            "UNFCCC.non-party.467.0",
            "CCLW.executive.10061.4515",
            "AF.document.i00000021.n0000_translated_en",
            "UNFCCC.party.492.0",
            "CPR.document.i00000549.n0000",
        ]
    )


@pytest.mark.asyncio
async def test__document_stems_from_parameters_pointer(
    test_config: Config,
    mock_classifier_specs,
    mock_bucket_labelled_passages_large,
    mock_s3_async_client,
):
    run_output_identifier = "2025-01-15T10:30-test-flow-run"

    inference_result_s3_uri = build_inference_result_s3_uri(
        cache_bucket_str=test_config.cache_bucket_str,
        inference_document_target_prefix=test_config.inference_document_target_prefix,
        run_output_identifier=run_output_identifier,
    )

    document_stems = [
        "UNFCCC.non-party.467.0",
        "CCLW.executive.10061.4515",
        "AF.document.i00000021.n0000_translated_en",
        "UNFCCC.party.492.0",
        "CPR.document.i00000549.n0000",
    ]

    result_data = {
        "successful_document_stems": list(document_stems),
    }

    result_json = json.dumps(result_data)

    _ = await mock_s3_async_client.put_object(
        Bucket=inference_result_s3_uri.bucket,
        Key=inference_result_s3_uri.key,
        Body=result_json,
        ContentType="application/json",
    )

    actual = await _document_stems_from_parameters(
        config=test_config,
        document_stems=None,
        run_output_identifier=run_output_identifier,
    )

    assert set(actual) == set(
        [
            "UNFCCC.non-party.467.0",
            "CCLW.executive.10061.4515",
            "AF.document.i00000021.n0000_translated_en",
            "UNFCCC.party.492.0",
            "CPR.document.i00000549.n0000",
        ]
    )
