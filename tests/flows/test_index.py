import json
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from cpr_sdk.models.search import Concept as VespaConcept
from cpr_sdk.models.search import Passage as VespaPassage
from cpr_sdk.search_adaptors import VespaSearchAdapter
from prefect.logging import disable_run_logger

from flows.index import (
    CONCEPTS_COUNTS_PREFIX_DEFAULT,
    ClassifierSpec,
    ConceptModel,
    Config,
    DocumentImporter,
    convert_labelled_passage_to_concepts,
    get_data_id_from_vespa_hit_id,
    get_document_passage_from_vespa,
    get_document_passages_from_vespa,
    get_parent_concepts_from_concept,
    get_updated_passage_concepts,
    get_vespa_search_adapter_from_aws_secrets,
    index_by_s3,
    index_labelled_passages_from_s3_to_vespa,
    iterate_batch,
    partial_update_text_block,
    run_partial_updates_of_concepts_for_document_passages,
    s3_obj_generator,
    s3_obj_generator_from_s3_paths,
    s3_obj_generator_from_s3_prefixes,
    s3_paths_or_s3_prefixes,
)
from scripts.cloud import AwsEnv
from src.concept import Concept
from src.identifiers import WikibaseID
from src.labelled_passage import LabelledPassage
from src.span import Span

DOCUMENT_PASSAGE_ID_PATTERN = re.compile(
    r"id:doc_search:document_passage::[a-zA-Z]+.[a-zA-Z]+.\d+.\d+.\d+"
)
DATA_ID_PATTERN = re.compile(r"[a-zA-Z]+.[a-zA-Z]+.\d+.\d+.\d+")


def test_get_data_id_from_vespa_hit_id() -> None:
    """Test that we can extract the data ID from a vespa hit id."""
    assert (
        DATA_ID_PATTERN.match(
            get_data_id_from_vespa_hit_id(
                "id:doc_search:document_passage::CCLW.executive.00000.0000.001"
            )
        )
        is not None
    )


def test_vespa_search_adapter_from_aws_secrets(
    create_vespa_params,
    mock_vespa_credentials,
    tmp_path,
) -> None:
    """Test that we can successfully instantiate the VespaSearchAdpater from ssm params."""
    vespa_search_adapter = get_vespa_search_adapter_from_aws_secrets(
        cert_dir=str(tmp_path),
        vespa_public_cert_param_name="VESPA_PUBLIC_CERT_FULL_ACCESS",
        vespa_private_key_param_name="VESPA_PRIVATE_KEY_FULL_ACCESS",
    )

    cert_path = tmp_path / "cert.pem"
    assert cert_path.exists()

    key_path = tmp_path / "key.pem"
    assert key_path.exists()

    with open(cert_path, "r", encoding="utf-8") as f:
        assert f.read() == "Public cert content\n"
    with open(key_path, "r", encoding="utf-8") as f:
        assert f.read() == "Private key content\n"
    assert (
        vespa_search_adapter.instance_url
        == mock_vespa_credentials["VESPA_INSTANCE_URL"]
    )
    assert vespa_search_adapter.client.cert == str(cert_path)
    assert vespa_search_adapter.client.key == str(key_path)


def test_s3_obj_generator_from_s3_prefixes(
    mock_bucket,
    mock_bucket_b,
    mock_bucket_labelled_passages,
    mock_bucket_labelled_passages_b,
    s3_prefix_labelled_passages,
    labelled_passage_fixture_files,
) -> None:
    """Test the s3 object generator."""
    gen = s3_obj_generator_from_s3_prefixes(
        [
            os.path.join("s3://", mock_bucket, s3_prefix_labelled_passages),
            os.path.join("s3://", mock_bucket_b, s3_prefix_labelled_passages),
        ],
    )
    result = list(gen)
    expected: list[DocumentImporter] = [
        (
            Path(f).stem,
            f"s3://{b}/{s3_prefix_labelled_passages}/{Path(f).stem}.json",
        )
        for b in [mock_bucket, mock_bucket_b]
        for f in labelled_passage_fixture_files
    ]
    assert expected == result


def test_s3_obj_generator_from_s3_paths(
    mock_bucket,
    mock_bucket_labelled_passages,
    s3_prefix_labelled_passages,
    labelled_passage_fixture_files,
) -> None:
    """Test the s3 object generator."""
    s3_paths = [
        os.path.join("s3://", mock_bucket, s3_prefix_labelled_passages, f)
        for f in labelled_passage_fixture_files
    ]
    gen = s3_obj_generator_from_s3_paths(s3_paths=s3_paths)
    result = list(gen)
    expected: list[DocumentImporter] = [
        (
            Path(f).stem,
            f"s3://{mock_bucket}/{s3_prefix_labelled_passages}/{Path(f).stem}.json",
        )
        for f in labelled_passage_fixture_files
    ]
    assert expected == result


@pytest.mark.vespa
def test_get_document_passages_from_vespa(
    local_vespa_search_adapter: VespaSearchAdapter,
    document_passages_test_data_file_path: str,
    vespa_app,
) -> None:
    """Test that we can retrieve all the passages for a document in vespa."""

    # Test that we retrieve no passages for a document that doesn't exist
    document_passages = get_document_passages_from_vespa(
        document_import_id="test.executive.1.1",
        vespa_search_adapter=local_vespa_search_adapter,
    )

    assert document_passages == []

    # Test that we can retrieve all the passages for a document that does exist
    document_import_id = "CCLW.executive.10014.4470"

    with open(document_passages_test_data_file_path) as f:
        document_passage_test_data = json.load(f)

    family_document_passages_count_expected = sum(
        1
        for doc in document_passage_test_data
        if doc["fields"]["family_document_ref"]
        == f"id:doc_search:family_document::{document_import_id}"
    )

    document_passages = get_document_passages_from_vespa(
        document_import_id=document_import_id,
        vespa_search_adapter=local_vespa_search_adapter,
    )

    assert len(document_passages) > 0
    assert len(document_passages) == family_document_passages_count_expected
    assert all(
        [
            (
                type(passage) is VespaPassage
                and type(passage_id) is str
                and bool(DOCUMENT_PASSAGE_ID_PATTERN.fullmatch(passage_id))
            )
            for passage_id, passage in document_passages
        ]
    )


@pytest.mark.vespa
def test_get_document_passage_from_vespa(
    local_vespa_search_adapter: VespaSearchAdapter,
    vespa_app,
) -> None:
    """Test that we can retrieve a passage for a document in vespa."""

    # Test that we retrieve no passages for a document that doesn't exist
    with pytest.raises(
        ValueError, match="Expected 1 document passage for text block `00001`, got 0"
    ):
        get_document_passage_from_vespa(
            text_block_id="00001",  # This text block doesn't exist
            document_import_id="test.executive.1.1",  # This document doesn't exist
            vespa_search_adapter=local_vespa_search_adapter,
        )

    # Test that we can retrieve all the passages for a document that does exist
    document_passage_id, document_passage = get_document_passage_from_vespa(
        text_block_id="1457",
        document_import_id="CCLW.executive.10014.4470",
        vespa_search_adapter=local_vespa_search_adapter,
    )

    assert isinstance(document_passage, VespaPassage)
    assert isinstance(document_passage_id, str)
    assert DOCUMENT_PASSAGE_ID_PATTERN.fullmatch(document_passage_id)


@pytest.mark.asyncio
@pytest.mark.vespa
async def test_run_partial_updates_of_concepts_for_document_passages(
    local_vespa_search_adapter: VespaSearchAdapter,
    vespa_app,
    mock_bucket,
    s3_prefix_labelled_passages,
    labelled_passage_fixture_files,
    mock_bucket_labelled_passages,
    mock_s3_client,
) -> None:
    """Test that we can run partial updates of concepts for document passages."""
    document_fixture = labelled_passage_fixture_files[0]
    document_import_id = Path(document_fixture).stem
    document_object_uri = (
        f"s3://{mock_bucket}/{s3_prefix_labelled_passages}/{document_fixture}"
    )

    # Confirm that the example concepts are not in the document passages
    initial_passages = get_document_passages_from_vespa(
        document_import_id=document_import_id,
        vespa_search_adapter=local_vespa_search_adapter,
    )
    initial_concepts = [
        concept
        for _, passage in initial_passages
        if passage.concepts
        for concept in passage.concepts
    ]
    assert len(initial_concepts) == 3660

    # Confirm that we can add the example concepts to the document passages

    test_counts = Counter(
        {
            ConceptModel(
                wikibase_id=WikibaseID("Q10015"),
                model_name='KeywordClassifier("professional services sector")',
            ): 2,
            ConceptModel(
                wikibase_id=WikibaseID("Q10014"),
                model_name='KeywordClassifier("professional services sector")',
            ): 1,
        }
    )

    assert (
        test_counts
        == await run_partial_updates_of_concepts_for_document_passages.fn(
            document_importer=(document_import_id, document_object_uri),
            vespa_search_adapter=local_vespa_search_adapter,
            cache_bucket=mock_bucket,
            concepts_counts_prefix=CONCEPTS_COUNTS_PREFIX_DEFAULT,
        )
    )
    updated_passages = get_document_passages_from_vespa(
        document_import_id=document_import_id,
        vespa_search_adapter=local_vespa_search_adapter,
    )
    updated_concepts = [
        concept
        for _, passage in updated_passages
        if passage.concepts
        for concept in passage.concepts
    ]

    # Original + fixture (.tests/flows/fixtures/CCLW.executive.10014.4470.json)
    assert len(updated_concepts) == 3663

    test_counts_serialised = {str(k): v for k, v in test_counts.items()}

    result = mock_s3_client.get_object(
        Bucket=mock_bucket,
        Key="concepts_counts/Q788/latest/CCLW.executive.10014.4470.json",
    )
    assert test_counts_serialised == json.loads(result["Body"].read().decode("utf-8"))


@pytest.mark.asyncio
@pytest.mark.vespa
async def test_run_partial_updates_of_concepts_for_document_passages_task_failure(
    local_vespa_search_adapter: VespaSearchAdapter,
    example_vespa_concepts: list[VespaConcept],
    vespa_app,
    mock_bucket,
    s3_prefix_labelled_passages,
    labelled_passage_fixture_files,
    mock_bucket_labelled_passages,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that we can continue on errors."""

    def mock_update_data(schema, namespace, data_id, fields):
        raise Exception("Forced update failure")

    with patch.object(
        local_vespa_search_adapter.client, "update_data", side_effect=mock_update_data
    ):
        document_fixture = labelled_passage_fixture_files[0]
        document_import_id = Path(document_fixture).stem
        document_object_uri = (
            f"s3://{mock_bucket}/{s3_prefix_labelled_passages}/{document_fixture}"
        )

        # Run the update
        assert (
            Counter()
            == await run_partial_updates_of_concepts_for_document_passages.fn(
                document_importer=(document_import_id, document_object_uri),
                vespa_search_adapter=local_vespa_search_adapter,
                cache_bucket=mock_bucket,
                concepts_counts_prefix=CONCEPTS_COUNTS_PREFIX_DEFAULT,
            )
        )

        # Verify error was logged for the failed update
        error_logs = [r for r in caplog.records if r.levelname == "ERROR"]
        assert any(
            "failed to do partial update for text block `1273`: Forced update failure"
            == str(r.message)
            for r in error_logs
        )
        assert any(
            "failed to do partial update for text block `1052`: Forced update failure"
            == str(r.message)
            for r in error_logs
        )


@pytest.mark.asyncio
@pytest.mark.vespa
# @pytest.mark.flaky_on_ci  # Disabled for now, to see if it's still flaky
async def test_index_by_s3_with_s3_prefixes(
    mock_bucket,
    mock_bucket_labelled_passages,
    s3_prefix_labelled_passages,
    labelled_passage_fixture_files,
    local_vespa_search_adapter: VespaSearchAdapter,
    vespa_app,
) -> None:
    """We can successfully index labelled passages from S3 into Vespa."""
    initial_passages_response = local_vespa_search_adapter.client.query(
        yql="select * from document_passage where true"
    )
    initial_concepts_count = sum(
        len(hit["fields"]["concepts"]) for hit in initial_passages_response.hits
    )

    await index_by_s3(
        aws_env=AwsEnv.sandbox,
        vespa_search_adapter=local_vespa_search_adapter,
        s3_prefixes=[os.path.join("s3://", mock_bucket, s3_prefix_labelled_passages)],
        s3_paths=None,
        as_deployment=False,
        cache_bucket=mock_bucket,
        concepts_counts_prefix=CONCEPTS_COUNTS_PREFIX_DEFAULT,
    )

    final_passages_response = local_vespa_search_adapter.client.query(
        yql="select * from document_passage where true"
    )
    final_concepts_count = sum(
        len(hit["fields"]["concepts"]) for hit in final_passages_response.hits
    )

    assert initial_concepts_count < final_concepts_count
    # Original + fixture (.tests/flows/fixtures/*.json)
    assert final_concepts_count == 3933


@pytest.mark.asyncio
@pytest.mark.vespa
# @pytest.mark.flaky_on_ci  # Disabled for now, to see if it's still flaky
async def test_index_by_s3_with_s3_paths(
    mock_bucket,
    mock_bucket_labelled_passages,
    s3_prefix_labelled_passages,
    labelled_passage_fixture_files,
    local_vespa_search_adapter: VespaSearchAdapter,
    vespa_app,
) -> None:
    """We can successfully index labelled passages from S3 into Vespa."""
    initial_passages_response = local_vespa_search_adapter.client.query(
        yql="select * from document_passage where true"
    )
    initial_concepts_count = sum(
        len(hit["fields"]["concepts"]) for hit in initial_passages_response.hits
    )

    s3_paths = [
        f"s3://{mock_bucket}/{s3_prefix_labelled_passages}/{doc_file}"
        for doc_file in labelled_passage_fixture_files
    ]

    await index_by_s3(
        aws_env=AwsEnv.sandbox,
        vespa_search_adapter=local_vespa_search_adapter,
        s3_prefixes=None,
        s3_paths=s3_paths,
        as_deployment=False,
        cache_bucket=mock_bucket,
        concepts_counts_prefix=CONCEPTS_COUNTS_PREFIX_DEFAULT,
    )

    final_passages_response = local_vespa_search_adapter.client.query(
        yql="select * from document_passage where true"
    )
    final_concepts_count = sum(
        len(hit["fields"]["concepts"]) for hit in final_passages_response.hits
    )

    assert initial_concepts_count < final_concepts_count
    # Original + fixture (.tests/flows/fixtures/*.json)
    assert final_concepts_count == 3933


@pytest.mark.asyncio
@pytest.mark.vespa
async def test_index_by_s3_task_failure(
    mock_bucket,
    mock_bucket_labelled_passages,
    s3_prefix_labelled_passages,
    labelled_passage_fixture_files,
    local_vespa_search_adapter: VespaSearchAdapter,
    vespa_app,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that index_by_s3 handles task failures gracefully."""

    async def mock_run_partial_updates_of_concepts_for_batch_flow_or_deployment(
        *args, **kwargs
    ):
        raise Exception("Forced update failure")

    with patch(
        "flows.index.run_partial_updates_of_concepts_for_batch_flow_or_deployment",
        side_effect=mock_run_partial_updates_of_concepts_for_batch_flow_or_deployment,
    ):
        await index_by_s3(
            aws_env=AwsEnv.sandbox,
            vespa_search_adapter=local_vespa_search_adapter,
            s3_prefixes=[
                os.path.join("s3://", mock_bucket, s3_prefix_labelled_passages)
            ],
            s3_paths=None,
            as_deployment=False,
            cache_bucket=mock_bucket,
            concepts_counts_prefix=CONCEPTS_COUNTS_PREFIX_DEFAULT,
        )

        # Verify error was logged for the failed update
        error_logs = [r for r in caplog.records if r.levelname == "ERROR"]
        assert any(
            "failed to process document" in r.message
            and "Forced update failure" in r.message
            for r in error_logs
        ), "Expected error log for failed document update not found"


@pytest.mark.asyncio
@pytest.mark.vespa
async def test_partial_update_text_block(
    local_vespa_search_adapter: VespaSearchAdapter,
    vespa_app,
    example_vespa_concepts: list[VespaConcept],
):
    document_import_id = "CCLW.executive.10014.4470"

    concept_a, concept_b = example_vespa_concepts

    # Confirm that the example concepts are not in the document passages
    initial_passages = get_document_passages_from_vespa(
        document_import_id=document_import_id,
        vespa_search_adapter=local_vespa_search_adapter,
    )
    initial_concepts = [
        concept
        for _, passage in initial_passages
        if passage.concepts
        for concept in passage.concepts
    ]

    assert len(initial_passages) > 0, "fixture data wasn't loaded in"
    assert all(concept not in initial_concepts for concept in example_vespa_concepts)

    text_block_id = "1401"

    assert concept_a not in initial_concepts

    result_a = await partial_update_text_block(
        text_block_id,
        [concept_a],
        document_import_id,
        local_vespa_search_adapter,
    )

    assert result_a is None

    passages_a = get_document_passages_from_vespa(
        document_import_id=document_import_id,
        vespa_search_adapter=local_vespa_search_adapter,
    )
    concepts_a = [
        concept
        for _, passage in passages_a
        if passage.concepts
        for concept in passage.concepts
    ]

    assert concept_a in concepts_a

    assert concept_b not in concepts_a

    result_b = await partial_update_text_block(
        text_block_id,
        [concept_b],
        document_import_id,
        local_vespa_search_adapter,
    )

    assert result_b is None

    passages_b = get_document_passages_from_vespa(
        document_import_id=document_import_id,
        vespa_search_adapter=local_vespa_search_adapter,
    )
    concepts_b = [
        concept
        for _, passage in passages_b
        if passage.concepts
        for concept in passage.concepts
    ]

    assert concept_a in concepts_a
    assert concept_b in concepts_b


@pytest.mark.asyncio
@pytest.mark.vespa
async def test_index_labelled_passages_from_s3_to_vespa_with_document_ids_with_config(
    mock_bucket,
    mock_bucket_labelled_passages,
    s3_prefix_labelled_passages,
    labelled_passage_fixture_files,
    local_vespa_search_adapter: VespaSearchAdapter,
    vespa_app,
) -> None:
    initial_passages_response = local_vespa_search_adapter.client.query(
        yql="select * from document_passage where true"
    )
    initial_concepts_count = sum(
        len(hit["fields"]["concepts"]) for hit in initial_passages_response.hits
    )

    classifier_spec = ClassifierSpec(name="Q788", alias="latest")
    document_ids = [
        Path(labelled_passage_fixture_file).stem
        for labelled_passage_fixture_file in labelled_passage_fixture_files
    ]
    config = Config(
        cache_bucket=mock_bucket,
        vespa_search_adapter=local_vespa_search_adapter,
        as_deployment=False,
    )

    await index_labelled_passages_from_s3_to_vespa(
        classifier_specs=[classifier_spec],
        document_ids=document_ids,
        config=config,
    )

    final_passages_response = local_vespa_search_adapter.client.query(
        yql="select * from document_passage where true"
    )
    final_concepts_count = sum(
        len(hit["fields"]["concepts"]) for hit in final_passages_response.hits
    )

    assert initial_concepts_count < final_concepts_count
    # Original + fixture (.tests/flows/fixtures/*.json)
    assert final_concepts_count == 3933


@pytest.mark.asyncio
@pytest.mark.vespa
@pytest.mark.skip(reason="cannot test due to run_deployment usage")
async def test_index_labelled_passages_from_s3_to_vespa_with_document_ids_with_default_config(
    mock_bucket,
    mock_bucket_labelled_passages,
    s3_prefix_labelled_passages,
    labelled_passage_fixture_files,
    local_vespa_search_adapter: VespaSearchAdapter,
    vespa_app,
) -> None:
    with patch("flows.index.get_prefect_job_variable", return_value=mock_bucket), patch(
        "flows.index.get_vespa_search_adapter_from_aws_secrets",
        return_value=local_vespa_search_adapter,
    ), disable_run_logger():
        initial_passages_response = local_vespa_search_adapter.client.query(
            yql="select * from document_passage where true"
        )
        initial_concepts_count = sum(
            len(hit["fields"]["concepts"]) for hit in initial_passages_response.hits
        )

        classifier_spec = ClassifierSpec(name="Q788", alias="latest")
        document_ids = [
            Path(labelled_passage_fixture_file).stem
            for labelled_passage_fixture_file in labelled_passage_fixture_files
        ]

        await index_labelled_passages_from_s3_to_vespa(
            classifier_specs=[classifier_spec],
            document_ids=document_ids,
        )

        final_passages_response = local_vespa_search_adapter.client.query(
            yql="select * from document_passage where true"
        )
        final_concepts_count = sum(
            len(hit["fields"]["concepts"]) for hit in final_passages_response.hits
        )

    assert initial_concepts_count < final_concepts_count
    # Original + fixture (.tests/flows/fixtures/*.json)
    assert final_concepts_count == 3933


def test_get_updated_passage_concepts(
    example_vespa_concepts: list[VespaConcept],
) -> None:
    """Test that we can retrieve the updated passage concepts dict."""
    for concept in example_vespa_concepts:
        # Test we can add a concept to the passage concepts that doesn't already
        # exist.
        updated_passage_concepts = get_updated_passage_concepts(
            passage=VespaPassage(
                text_block="Test text.",
                text_block_id="1",
                text_block_type="Text",
                concepts=[],
            ),
            concepts=[concept],
        )
        assert len(updated_passage_concepts) == 1
        assert updated_passage_concepts[0] == concept.model_dump(mode="json")

        # Test that we can remove old model concepts from the passage concepts and
        # add the new one.
        updated_passage_concepts = get_updated_passage_concepts(
            passage=VespaPassage(
                text_block="Test text.",
                text_block_id="1",
                text_block_type="Text",
                concepts=[
                    VespaConcept(
                        id="1",
                        name="extreme weather",
                        parent_concepts=[{"name": "weather", "id": "Q123"}],
                        parent_concept_ids_flat="Q123,",
                        model=concept.model,  # Ensure the models are the same
                        end=100,
                        start=0,
                        timestamp=datetime.now(),
                    )
                ],
            ),
            concepts=[concept],
        )
        assert len(updated_passage_concepts) == 1
        assert updated_passage_concepts[0] == concept.model_dump(mode="json")

        # Test that we can add new concepts and retain concepts from other models
        updated_passage_concepts = get_updated_passage_concepts(
            passage=VespaPassage(
                text_block="Test text.",
                text_block_id="1",
                text_block_type="Text",
                concepts=[
                    VespaConcept(
                        id="1",
                        name="extreme weather",
                        parent_concepts=[{"name": "weather", "id": "Q123"}],
                        parent_concept_ids_flat="Q123,",
                        model="non-existent model",  # Ensure the models are NOT the same
                        end=100,
                        start=0,
                        timestamp=datetime.now(),
                    )
                ],
            ),
            concepts=[concept],
        )
        assert len(updated_passage_concepts) == 2
        assert concept.model_dump(mode="json") in updated_passage_concepts


def test_convert_labelled_passges_to_concepts(
    example_labelled_passages: list[LabelledPassage],
) -> None:
    """Test that we can correctly convert labelled passages to concepts."""
    concepts = convert_labelled_passage_to_concepts(example_labelled_passages[0])
    assert all([isinstance(concept, VespaConcept) for concept in concepts])


def test_convert_labelled_passges_to_concepts_raises_error(
    example_labelled_passages: list[LabelledPassage],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that we correctly log errors when a Span has no concept ID or timestamps."""
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
    # Add a span without timestamps
    example_labelled_passages[0].spans.append(
        Span(
            text="Test text.",
            start_index=0,
            end_index=8,
            concept_id="Q123",
            labellers=[],
            timestamps=[],  # Empty timestamps
        )
    )

    concepts = convert_labelled_passage_to_concepts(example_labelled_passages[0])

    # Check that appropriate error messages were logged
    error_messages = [r.message for r in caplog.records if r.levelname == "ERROR"]
    assert any("span concept ID is missing" in msg for msg in error_messages)
    assert any("span timestamps are missing" in msg for msg in error_messages)

    # Verify that the problematic spans were skipped but valid one was processed
    assert len(concepts) == 1


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


def test_s3_paths_or_s3_prefixes_no_classifier(
    mock_bucket,
    mock_bucket_labelled_passages,
):
    """Test s3_paths_or_s3_prefixes returns base prefix when no classifier spec provided."""
    config = Config(cache_bucket=mock_bucket)

    s3_accessor = s3_paths_or_s3_prefixes(
        classifier_specs=None,
        document_ids=None,
        cache_bucket=config.cache_bucket,  # pyright: ignore[reportArgumentType]
        prefix=config.document_source_prefix,
    )

    assert s3_accessor.prefixes == [f"s3://{mock_bucket}/labelled_passages"]
    assert s3_accessor.paths is None


def test_s3_paths_or_s3_prefixes_no_classifier_and_docs(
    mock_bucket,
    mock_bucket_labelled_passages,
    labelled_passage_fixture_ids,
    labelled_passage_fixture_files,
    s3_prefix_mock_bucket_labelled_passages,
):
    config = Config(cache_bucket=mock_bucket)

    with pytest.raises(
        ValueError,
        match="if document IDs are specified, a classifier "
        "specifcation must also be specified, since they're "
        "namespaced by classifiers \\(e\\.g\\. "
        "`s3://cpr-sandbox-data-pipeline-cache/labelled_passages/Q787/"
        "v4/CCLW\\.legislative\\.10695\\.6015\\.json`\\)",
    ), disable_run_logger():
        s3_paths_or_s3_prefixes(
            classifier_specs=None,
            document_ids=labelled_passage_fixture_ids,
            cache_bucket=config.cache_bucket,  # pyright: ignore[reportArgumentType]
            prefix=config.document_source_prefix,
        )


def test_s3_paths_or_s3_prefixes_with_classifier_no_docs(
    mock_bucket,
):
    """Test s3_paths_or_s3_prefixes returns classifier-specific prefix when no document IDs provided."""
    config = Config(cache_bucket=mock_bucket)
    classifier_spec_q788 = ClassifierSpec(name="Q788", alias="latest")
    classifier_spec_q699 = ClassifierSpec(name="Q699", alias="latest")

    s3_accessor = s3_paths_or_s3_prefixes(
        classifier_specs=[classifier_spec_q788, classifier_spec_q699],
        document_ids=None,
        cache_bucket=config.cache_bucket,  # pyright: ignore[reportArgumentType]
        prefix=config.document_source_prefix,
    )

    assert s3_accessor.prefixes == [
        f"s3://{mock_bucket}/labelled_passages/Q788/latest",
        f"s3://{mock_bucket}/labelled_passages/Q699/latest",
    ]
    assert s3_accessor.paths is None


def test_s3_paths_or_s3_prefixes_with_classifier_and_docs(
    mock_bucket,
    mock_bucket_labelled_passages,
    labelled_passage_fixture_ids,
    labelled_passage_fixture_files,
    s3_prefix_mock_bucket_labelled_passages,
):
    """Test s3_paths_or_s3_prefixes returns specific paths when both classifier and document IDs provided."""
    config = Config(cache_bucket=mock_bucket)
    classifier_spec = ClassifierSpec(name="Q788", alias="latest")

    expected_paths = [
        f"{s3_prefix_mock_bucket_labelled_passages}/{labelled_passage_fixture_file}"
        for labelled_passage_fixture_file in labelled_passage_fixture_files
    ]

    s3_accessor = s3_paths_or_s3_prefixes(
        classifier_specs=[classifier_spec],
        document_ids=labelled_passage_fixture_ids,
        cache_bucket=config.cache_bucket,  # pyright: ignore[reportArgumentType]
        prefix=config.document_source_prefix,
    )

    assert s3_accessor.prefixes is None
    assert sorted(s3_accessor.paths) == sorted(  # pyright: ignore[reportArgumentType]
        expected_paths
    )


@pytest.mark.parametrize(
    "s3_prefixes,s3_paths,expected_error,error_match",
    [
        # Both provided - should raise error
        (
            ["s3://bucket/prefix"],
            ["s3://bucket/prefix/file.json"],
            ValueError,
            "Either s3_prefixes or s3_paths must be provided, not both.",
        ),
        # Neither provided - should raise error
        (
            None,
            None,
            ValueError,
            "Either s3_prefix or s3_paths must be provided.",
        ),
        # Invalid types - should raise error
        (
            None,
            None,
            ValueError,
            "Either s3_prefix or s3_paths must be provided.",
        ),
    ],
)
def test_s3_obj_generator_errors(
    s3_prefixes: list[str] | None,
    s3_paths: list[str] | None,
    expected_error: type[Exception],
    error_match: str,
) -> None:
    """Test s3_obj_generator error cases."""
    with pytest.raises(expected_error, match=error_match), disable_run_logger():
        _ = s3_obj_generator(s3_prefixes=s3_prefixes, s3_paths=s3_paths)


@pytest.mark.parametrize(
    "use_prefixes",
    [True, False],
    ids=["using_prefixes", "using_paths"],
)
def test_s3_obj_generator_valid_cases(
    mock_bucket: str,
    mock_bucket_labelled_passages: None,
    s3_prefix_labelled_passages: str,
    labelled_passage_fixture_files: list[str],
    use_prefixes: bool,
) -> None:
    """Test s3_obj_generator with valid inputs using either prefixes or paths."""
    if use_prefixes:
        s3_prefixes = [os.path.join("s3://", mock_bucket, s3_prefix_labelled_passages)]
        s3_paths = None
    else:
        s3_prefixes = None
        s3_paths = [
            os.path.join("s3://", mock_bucket, s3_prefix_labelled_passages, f)
            for f in labelled_passage_fixture_files
        ]

    gen = s3_obj_generator(s3_prefixes=s3_prefixes, s3_paths=s3_paths)
    result = list(gen)
    expected: list[DocumentImporter] = [
        (
            Path(f).stem,
            f"s3://{mock_bucket}/{s3_prefix_labelled_passages}/{Path(f).stem}.json",
        )
        for f in labelled_passage_fixture_files
    ]
    assert expected == result


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
