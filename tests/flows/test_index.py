import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Type
from unittest.mock import patch

import pytest
from cpr_sdk.models.search import Concept as VespaConcept
from cpr_sdk.models.search import Passage as VespaPassage
from cpr_sdk.search_adaptors import VespaSearchAdapter
from prefect.logging import disable_run_logger

from flows.index import (
    ClassifierSpec,
    Config,
    convert_labelled_passages_to_concepts,
    convert_labelled_passages_to_document_concepts,
    dump_document_concepts,
    get_document_passages_from_vespa,
    get_parent_concepts_from_concept,
    get_passage_for_text_block,
    get_updated_passage_concepts,
    get_vespa_search_adapter_from_aws_secrets,
    group_concepts_on_text_block,
    index_by_s3,
    index_labelled_passages_from_s3_to_vespa,
    iterate_batch,
    labelled_passages_generator,
    load_document_concepts,
    partial_update_text_block,
    run_partial_updates_of_concepts_for_document_passages,
    s3_obj_generator,
    s3_obj_generator_from_s3_paths,
    s3_obj_generator_from_s3_prefixes,
    s3_paths_or_s3_prefixes,
    split_large_concepts_updates,
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


def test_vespa_search_adapter_from_aws_secrets(
    create_vespa_params, mock_vespa_credentials, tmp_path
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
    with disable_run_logger():
        s3_gen = s3_obj_generator_from_s3_prefixes(
            [
                os.path.join("s3://", mock_bucket, s3_prefix_labelled_passages),
                os.path.join("s3://", mock_bucket_b, s3_prefix_labelled_passages),
            ],
        )
        s3_files = list(s3_gen)
        assert len(s3_files) == len(labelled_passage_fixture_files * 2)

        expected_keys = [
            f"{s3_prefix_labelled_passages}/{Path(f).stem}"
            for f in labelled_passage_fixture_files
        ] + [
            f"{s3_prefix_labelled_passages}/{Path(f).stem}"
            for f in labelled_passage_fixture_files
        ]
        s3_files_keys = [file[0].replace(".json", "") for file in s3_files]

        assert sorted(s3_files_keys) == sorted(expected_keys)


def test_s3_obj_generator_from_s3_paths(
    mock_bucket,
    mock_bucket_labelled_passages,
    s3_prefix_labelled_passages,
    labelled_passage_fixture_files,
) -> None:
    """Test the s3 object generator."""
    with disable_run_logger():
        s3_paths = {
            os.path.join("s3://", mock_bucket, s3_prefix_labelled_passages, f)
            for f in labelled_passage_fixture_files
        } | {"gibberish"}
        s3_gen = s3_obj_generator_from_s3_paths(s3_paths=s3_paths)
        s3_files = list(s3_gen)
        assert len(s3_files) == len(labelled_passage_fixture_files)

        expected_keys = [
            f"{s3_prefix_labelled_passages}/{Path(f).stem}"
            for f in labelled_passage_fixture_files
        ]
        s3_files_keys = [file[0].replace(".json", "") for file in s3_files]

        assert sorted(s3_files_keys) == sorted(expected_keys)


def test_labelled_passages_generator(
    mock_bucket,
    mock_bucket_labelled_passages,
    s3_prefix_labelled_passages,
    labelled_passage_fixture_files,
) -> None:
    """Test that the document concepts generator yields the correct objects."""
    s3_gen = s3_obj_generator_from_s3_prefixes(
        [os.path.join("s3://", mock_bucket, s3_prefix_labelled_passages)]
        + ["gibberish"]
    )
    labelled_passages_gen = labelled_passages_generator(generator_func=s3_gen)
    labelled_passages_files = list(labelled_passages_gen)
    expected_keys = [
        f"{s3_prefix_labelled_passages}/{Path(f).stem}.json"
        for f in labelled_passage_fixture_files
    ]

    assert len(labelled_passages_files) == len(labelled_passage_fixture_files)
    for s3_key, labelled_passages in labelled_passages_files:
        assert all([type(i) is LabelledPassage for i in labelled_passages])
        assert s3_key in expected_keys


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


@pytest.mark.asyncio
@pytest.mark.vespa
async def test_run_partial_updates_of_concepts_for_document_passages(
    local_vespa_search_adapter: VespaSearchAdapter,
    example_vespa_concepts: list[VespaConcept],
    vespa_app,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that we can run partial updates of concepts for document passages."""
    with disable_run_logger():
        document_import_id = "CCLW.executive.10014.4470"

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

        assert len(initial_passages) > 0
        assert all(
            concept not in initial_concepts for concept in example_vespa_concepts
        )

        document_concepts = [(c.id, c) for c in example_vespa_concepts]

        # Add a clear failure
        fake_id = "99999999"
        document_concepts.append(
            (
                fake_id,
                Concept(
                    id=fake_id,
                    name="wood industry",
                    parent_concepts=[
                        {"name": "forestry sector", "id": "Q788"},
                        {"name": "lumber", "id": "Q789"},
                    ],
                    parent_concept_ids_flat="Q788,Q789",
                    model='KeywordClassifier("wood industry")',
                    end=100,
                    start=0,
                    timestamp=datetime(2025, 1, 15, 16, 17, 21, 362785),
                    preferred_label="x",
                ),
            )
        )

        # Confirm that we can add the example concepts to the document passages
        await run_partial_updates_of_concepts_for_document_passages.fn(
            document_import_id=document_import_id,
            document_concepts=document_concepts,
            vespa_search_adapter=local_vespa_search_adapter,
        )

        # Verify error was logged for the failed update
        error_logs = [r for r in caplog.records if r.levelname == "ERROR"]
        assert any(
            f"No passages found for text block: {fake_id}" in r.message
            for r in error_logs
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

        assert len(updated_passages) > 0
        assert len(updated_concepts) != len(initial_concepts)
        assert all(
            [
                any([new_vespa_concept == c for c in updated_concepts])
                for new_vespa_concept in example_vespa_concepts
            ]
        )

        # Confirm we remove existing concepts and add new ones based on the model field
        modified_example_vespa_concepts = [
            (concept.id, concept.model_copy()) for concept in example_vespa_concepts * 2
        ]
        for idx, concept in enumerate(modified_example_vespa_concepts):
            # Make a change to the concept but keep the same model, this triggers removal
            # of the existing concepts with the same model
            concept[1].end = idx

        await run_partial_updates_of_concepts_for_document_passages.fn(
            document_import_id=document_import_id,
            document_concepts=modified_example_vespa_concepts,
            vespa_search_adapter=local_vespa_search_adapter,
        )

        second_updated_passages = get_document_passages_from_vespa(
            document_import_id=document_import_id,
            vespa_search_adapter=local_vespa_search_adapter,
        )
        second_updated_concepts = [
            concept
            for _, passage in second_updated_passages
            if passage.concepts
            for concept in passage.concepts
        ]

        assert len(second_updated_passages) > 0
        assert len(second_updated_concepts) != len(updated_concepts)
        # Assert that the number of concepts after a second update in vespa is correct.
        # This is equal to:
        #   (all existing concepts in vespa)
        #   - (minus concepts that have the same model as the new updates)
        #   + (new updates)
        #   This is as we remove old concepts for a model and replace them with the new ones.
        assert len(second_updated_concepts) == (
            len(updated_concepts)
            + len(modified_example_vespa_concepts)
            - len(example_vespa_concepts)
        )
        for _, new_vespa_concept in modified_example_vespa_concepts:
            assert new_vespa_concept in second_updated_concepts
        for example_vespa_concept in example_vespa_concepts:
            assert example_vespa_concept not in second_updated_concepts

        # Check that concept counts are indexed.
        response = local_vespa_search_adapter.client.query(
            yql=(
                "select * from family_document "
                "where concept_counts contains sameElement(value > 1)"
            )
        )
        assert len(response.hits) > 0


@pytest.mark.asyncio
@pytest.mark.vespa
async def test_run_partial_updates_of_concepts_for_document_passages_task_failure(
    local_vespa_search_adapter: VespaSearchAdapter,
    example_vespa_concepts: list[VespaConcept],
    vespa_app,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that we can continue on errors."""

    def mock_update_data(schema, namespace, data_id, fields):
        raise Exception("Forced update failure")

    with disable_run_logger(), patch.object(
        local_vespa_search_adapter.client, "update_data", side_effect=mock_update_data
    ):
        document_import_id = "CCLW.executive.10014.4470"

        document_concepts = [(c.id, c) for c in example_vespa_concepts]

        # Run the update
        await run_partial_updates_of_concepts_for_document_passages.fn(
            document_import_id=document_import_id,
            document_concepts=document_concepts,
            vespa_search_adapter=local_vespa_search_adapter,
        )

        # Verify error was logged for the failed update
        error_logs = [r for r in caplog.records if r.levelname == "ERROR"]
        assert any(
            "failed to do partial update for concept `1273`: Forced update failure"
            == str(r.message)
            for r in error_logs
        )


@pytest.mark.asyncio
@pytest.mark.vespa
@pytest.mark.flaky_on_ci
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
        as_subflow=False,
    )

    final_passages_response = local_vespa_search_adapter.client.query(
        yql="select * from document_passage where true"
    )
    final_concepts_count = sum(
        len(hit["fields"]["concepts"]) for hit in final_passages_response.hits
    )

    assert initial_concepts_count < final_concepts_count
    assert initial_concepts_count + len(labelled_passage_fixture_files) == (
        final_concepts_count
    )


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

    async def mock_run_partial_updates_of_concepts_for_document_passages_as(
        *args, **kwargs
    ):
        raise Exception("Forced update failure")

    with disable_run_logger(), patch(
        "flows.index.run_partial_updates_of_concepts_for_document_passages_as",
        side_effect=mock_run_partial_updates_of_concepts_for_document_passages_as,
    ):
        await index_by_s3(
            aws_env=AwsEnv.sandbox,
            vespa_search_adapter=local_vespa_search_adapter,
            s3_prefixes=[
                os.path.join("s3://", mock_bucket, s3_prefix_labelled_passages)
            ],
            s3_paths=None,
            as_subflow=False,
        )

        # Verify error was logged for the failed update
        error_logs = [r for r in caplog.records if r.levelname == "ERROR"]
        assert any(
            "failed to process document for S3 key" in r.message
            and "Forced update failure" in r.message
            for r in error_logs
        ), "Expected error log for failed document update not found"


@pytest.mark.asyncio
@pytest.mark.vespa
@pytest.mark.flaky_on_ci
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
        as_subflow=False,
    )

    final_passages_response = local_vespa_search_adapter.client.query(
        yql="select * from document_passage where true"
    )
    final_concepts_count = sum(
        len(hit["fields"]["concepts"]) for hit in final_passages_response.hits
    )

    assert initial_concepts_count < final_concepts_count
    assert initial_concepts_count + len(labelled_passage_fixture_files) == (
        final_concepts_count
    )


@pytest.mark.asyncio
@pytest.mark.vespa
async def test_partial_update_text_block(
    local_vespa_search_adapter: VespaSearchAdapter,
    vespa_app,
    example_vespa_concepts: list[VespaConcept],
):
    with disable_run_logger():
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
        assert all(
            concept not in initial_concepts for concept in example_vespa_concepts
        )

        text_block_id = "1401"

        assert concept_a not in initial_concepts

        result_a = await partial_update_text_block(
            text_block_id,
            initial_passages,
            local_vespa_search_adapter,
            [concept_a],
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
            initial_passages,
            local_vespa_search_adapter,
            [concept_b],
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


@pytest.mark.parametrize("text_block_id", ["1457", "p_2_b_120"])
def test_get_passage_for_text_block(
    example_vespa_concepts: list[VespaConcept], text_block_id: str
) -> None:
    """Test that we can retrieve the relevant passage for a text block."""
    relevant_passage = (
        "doc_id_1",
        VespaPassage(
            text_block="test text",
            text_block_id=text_block_id,
            text_block_type="test_type",
        ),
    )
    irrelevant_passage = (
        "doc_id_2",
        VespaPassage(
            text_block="test text",
            text_block_id="wrong_id",
            text_block_type="test_type",
        ),
    )

    for concept in example_vespa_concepts:
        concept.id = text_block_id

        data_id, passage_id, passage = get_passage_for_text_block(
            text_block_id=text_block_id,
            document_passages=[relevant_passage, irrelevant_passage],
        )
        assert data_id is not None
        data_id_pattern_match = DATA_ID_PATTERN.match(data_id) is not None
        assert data_id_pattern_match is not None
        assert passage_id == relevant_passage[0]
        assert passage == relevant_passage[1]


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
    with disable_run_logger():
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
            as_subflow=False,
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
        assert initial_concepts_count + len(labelled_passage_fixture_files) == (
            final_concepts_count
        )


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
        assert initial_concepts_count + len(labelled_passage_fixture_files) == (
            final_concepts_count
        )


def test_get_updated_passage_concepts(
    example_vespa_concepts: list[VespaConcept],
) -> None:
    """Test that we can retrieve the updated passage concepts dict."""
    for concept in example_vespa_concepts:
        # Test we can add a concept to the passage concepts that doesn't already
        # exist.
        updated_passage_concepts = get_updated_passage_concepts(
            concepts=[concept],
            passage=VespaPassage(
                text_block="Test text.",
                text_block_id="1",
                text_block_type="Text",
                concepts=[],
            ),
        )
        assert len(updated_passage_concepts) == 1
        assert updated_passage_concepts[0] == concept.model_dump(mode="json")

        # Test that we can remove old model concepts from the passage concepts and
        # add the new one.
        updated_passage_concepts = get_updated_passage_concepts(
            concepts=[concept],
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
        )
        assert len(updated_passage_concepts) == 1
        assert updated_passage_concepts[0] == concept.model_dump(mode="json")

        # Test that we can add new concepts and retain concepts from other models
        updated_passage_concepts = get_updated_passage_concepts(
            concepts=[concept],
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
        )
        assert len(updated_passage_concepts) == 2
        assert concept.model_dump(mode="json") in updated_passage_concepts


def test_convert_labelled_passges_to_concepts(
    example_labelled_passages: list[LabelledPassage],
) -> None:
    """Test that we can correctly convert labelled passages to concepts."""
    with disable_run_logger():
        concepts = convert_labelled_passages_to_concepts(example_labelled_passages)
        assert all(
            [
                (isinstance(text_block_id, str) and isinstance(concept, VespaConcept))
                for text_block_id, concept in concepts
            ]
        )


def test_convert_labelled_passges_to_concepts_raises_error(
    example_labelled_passages: list[LabelledPassage],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test that we correctly log errors when a Span has no concept ID or timestamps."""
    with disable_run_logger():
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

        concepts = convert_labelled_passages_to_concepts(example_labelled_passages)

        # Check that appropriate error messages were logged
        error_messages = [r.message for r in caplog.records if r.levelname == "ERROR"]
        assert any("span concept ID is missing" in msg for msg in error_messages)
        assert any("span timestamps are missing" in msg for msg in error_messages)

        # Verify that the problematic spans were skipped but valid ones were processed
        assert len(concepts) == len(
            [
                span
                for passage in example_labelled_passages
                for span in passage.spans
                if span.concept_id is not None and span.timestamps
            ]
        )


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


def test_group_concepts_on_text_block(
    example_vespa_concepts: list[VespaConcept],
) -> None:
    """
    Test that we can successfully group concepts on the relevant text block.

    Args:
        example_vespa_concepts (List[VespaConcept]): List of example Vespa concepts.
    """
    text_block_one_concept_count = 2
    text_block_one_concepts = [
        ("text_block_1", example_vespa_concepts[0])
    ] * text_block_one_concept_count

    text_block_two_concept_count = 11
    text_block_two_concepts = [
        ("text_block_2", example_vespa_concepts[0])
    ] * text_block_two_concept_count

    all_concepts = text_block_one_concepts + text_block_two_concepts
    grouped_concepts = group_concepts_on_text_block(all_concepts)

    assert isinstance(grouped_concepts, dict)
    for text_block_id, concepts in grouped_concepts.items():
        assert isinstance(text_block_id, str)
        assert all(isinstance(concept, VespaConcept) for concept in concepts)

    assert len(grouped_concepts) == 2
    assert set(grouped_concepts.keys()) == {"text_block_1", "text_block_2"}
    assert len(grouped_concepts["text_block_1"]) == text_block_one_concept_count
    assert len(grouped_concepts["text_block_2"]) == text_block_two_concept_count


@pytest.mark.asyncio
async def test_convert_labelled_passages_to_document_concepts(
    example_labelled_passages: list[LabelledPassage],
) -> None:
    """Test conversion of labelled passages to document concepts."""
    with disable_run_logger():

        def mock_generator():
            yield "s3://bucket/path1.json", example_labelled_passages[:2]
            yield "s3://bucket/path2.json", example_labelled_passages[2:]

        result = await convert_labelled_passages_to_document_concepts(mock_generator())

        assert len(result) == 2
        assert isinstance(result[0][0], str)
        assert isinstance(result[0][1], list)
        assert all(isinstance(concept, tuple) for concept in result[0][1])
        assert all(isinstance(concept[1], VespaConcept) for concept in result[0][1])


def test_dump_document_concepts(example_vespa_concepts: list[VespaConcept]):
    """Test dumping document concepts to JSON strings."""
    # Create test input with text block IDs and VespaConcepts
    document_concepts = [
        ("block_1", example_vespa_concepts[0]),
        ("block_2", example_vespa_concepts[1]),
    ]

    # Dump to JSON strings
    dumped = dump_document_concepts(document_concepts)

    # Verify structure and types
    assert len(dumped) == len(document_concepts)
    for (orig_id, orig_concept), (dumped_id, dumped_json) in zip(
        document_concepts, dumped
    ):
        assert dumped_id == orig_id
        assert isinstance(dumped_json, str)
        # Verify the JSON string can be loaded back into an equivalent object
        loaded_concept = VespaConcept.model_validate_json(dumped_json)
        assert loaded_concept == orig_concept


def test_load_document_concepts(example_vespa_concepts: list[VespaConcept]):
    """Test loading document concepts from JSON strings."""
    # Create test input with text block IDs and JSON strings
    document_concepts = [
        ("block_1", example_vespa_concepts[0].model_dump_json()),
        ("block_2", example_vespa_concepts[1].model_dump_json()),
    ]

    # Load from JSON strings
    loaded = load_document_concepts(document_concepts)

    # Verify structure and types
    assert len(loaded) == len(document_concepts)
    for (orig_id, orig_json), (loaded_id, loaded_concept) in zip(
        document_concepts, loaded
    ):
        assert loaded_id == orig_id
        assert isinstance(loaded_concept, VespaConcept)
        # Verify the loaded object matches the original
        original_concept = VespaConcept.model_validate_json(orig_json)
        assert loaded_concept == original_concept


def test_dump_load_document_concepts_roundtrip(
    example_vespa_concepts: list[VespaConcept],
):
    """Test that dumping and loading document concepts preserves data."""
    # Create test input
    original = [
        ("block_1", example_vespa_concepts[0]),
        ("block_2", example_vespa_concepts[1]),
    ]

    # Round trip through dump and load
    dumped = dump_document_concepts(original)
    loaded = load_document_concepts(dumped)

    # Verify nothing was lost
    assert len(loaded) == len(original)
    for (orig_id, orig_concept), (loaded_id, loaded_concept) in zip(original, loaded):
        assert loaded_id == orig_id
        assert loaded_concept == orig_concept


@pytest.mark.parametrize(
    "document_concepts, batch_size, expected",
    [
        # Test case 1: No splitting needed
        (
            [
                (
                    "block1",
                    [
                        VespaConcept(
                            id="1",
                            name="test",
                            parent_concepts=[],
                            parent_concept_ids_flat="",
                            model="model",
                            start=0,
                            end=1,
                            timestamp=datetime(2025, 1, 15, 16, 17, 21, 362785),
                        )
                    ]
                    * 10,
                )
            ],
            400,
            [
                (
                    "block1",
                    [
                        VespaConcept(
                            id="1",
                            name="test",
                            parent_concepts=[],
                            parent_concept_ids_flat="",
                            model="model",
                            start=0,
                            end=1,
                            timestamp=datetime(2025, 1, 15, 16, 17, 21, 362785),
                        )
                    ]
                    * 10,
                )
            ],
        ),
        # Test case 2: Single batch needs splitting
        (
            [
                (
                    "block1",
                    [
                        VespaConcept(
                            id="1",
                            name="test",
                            parent_concepts=[],
                            parent_concept_ids_flat="",
                            model="model",
                            start=0,
                            end=1,
                            timestamp=datetime(2025, 1, 15, 16, 17, 21, 362785),
                        )
                    ]
                    * 5,
                )
            ],
            4,
            [
                (
                    "block1",
                    [
                        VespaConcept(
                            id="1",
                            name="test",
                            parent_concepts=[],
                            parent_concept_ids_flat="",
                            model="model",
                            start=0,
                            end=1,
                            timestamp=datetime(2025, 1, 15, 16, 17, 21, 362785),
                        )
                    ]
                    * 4,
                ),
                (
                    "block1",
                    [
                        VespaConcept(
                            id="1",
                            name="test",
                            parent_concepts=[],
                            parent_concept_ids_flat="",
                            model="model",
                            start=0,
                            end=1,
                            timestamp=datetime(2025, 1, 15, 16, 17, 21, 362785),
                        )
                    ]
                    * 1,
                ),
            ],
        ),
        # Test case 3: Multiple batches, some need splitting
        (
            [
                (
                    "block1",
                    [
                        VespaConcept(
                            id="1",
                            name="test",
                            parent_concepts=[],
                            parent_concept_ids_flat="",
                            model="model",
                            start=0,
                            end=1,
                            timestamp=datetime(2025, 1, 15, 16, 17, 21, 362785),
                        )
                    ]
                    * 300,
                ),
                (
                    "block2",
                    [
                        VespaConcept(
                            id="2",
                            name="test",
                            parent_concepts=[],
                            parent_concept_ids_flat="",
                            model="model",
                            start=0,
                            end=1,
                            timestamp=datetime(2025, 1, 15, 16, 17, 21, 362785),
                        )
                    ]
                    * 500,
                ),
                (
                    "block3",
                    [
                        VespaConcept(
                            id="3",
                            name="test",
                            parent_concepts=[],
                            parent_concept_ids_flat="",
                            model="model",
                            start=0,
                            end=1,
                            timestamp=datetime(2025, 1, 15, 16, 17, 21, 362785),
                        )
                    ]
                    * 100,
                ),
            ],
            400,
            [
                (
                    "block1",
                    [
                        VespaConcept(
                            id="1",
                            name="test",
                            parent_concepts=[],
                            parent_concept_ids_flat="",
                            model="model",
                            start=0,
                            end=1,
                            timestamp=datetime(2025, 1, 15, 16, 17, 21, 362785),
                        )
                    ]
                    * 300,
                ),
                (
                    "block2",
                    [
                        VespaConcept(
                            id="2",
                            name="test",
                            parent_concepts=[],
                            parent_concept_ids_flat="",
                            model="model",
                            start=0,
                            end=1,
                            timestamp=datetime(2025, 1, 15, 16, 17, 21, 362785),
                        )
                    ]
                    * 400,
                ),
                (
                    "block2",
                    [
                        VespaConcept(
                            id="2",
                            name="test",
                            parent_concepts=[],
                            parent_concept_ids_flat="",
                            model="model",
                            start=0,
                            end=1,
                            timestamp=datetime(2025, 1, 15, 16, 17, 21, 362785),
                        )
                    ]
                    * 100,
                ),
                (
                    "block3",
                    [
                        VespaConcept(
                            id="3",
                            name="test",
                            parent_concepts=[],
                            parent_concept_ids_flat="",
                            model="model",
                            start=0,
                            end=1,
                            timestamp=datetime(2025, 1, 15, 16, 17, 21, 362785),
                        )
                    ]
                    * 100,
                ),
            ],
        ),
        # Test case 4: Empty input
        ([], 400, []),
        # Test case 5: Single concept that doesn't need splitting
        (
            [
                (
                    "block1",
                    [
                        VespaConcept(
                            id="1",
                            name="test",
                            parent_concepts=[],
                            parent_concept_ids_flat="",
                            model="model",
                            start=0,
                            end=1,
                            timestamp=datetime(2025, 1, 15, 16, 17, 21, 362785),
                        )
                    ],
                )
            ],
            400,
            [
                (
                    "block1",
                    [
                        VespaConcept(
                            id="1",
                            name="test",
                            parent_concepts=[],
                            parent_concept_ids_flat="",
                            model="model",
                            start=0,
                            end=1,
                            timestamp=datetime(2025, 1, 15, 16, 17, 21, 362785),
                        )
                    ],
                )
            ],
        ),
        # Test case 6: Exact batch size
        (
            [
                (
                    "block1",
                    [
                        VespaConcept(
                            id="1",
                            name="test",
                            parent_concepts=[],
                            parent_concept_ids_flat="",
                            model="model",
                            start=0,
                            end=1,
                            timestamp=datetime(2025, 1, 15, 16, 17, 21, 362785),
                        )
                    ]
                    * 400,
                )
            ],
            400,
            [
                (
                    "block1",
                    [
                        VespaConcept(
                            id="1",
                            name="test",
                            parent_concepts=[],
                            parent_concept_ids_flat="",
                            model="model",
                            start=0,
                            end=1,
                            timestamp=datetime(2025, 1, 15, 16, 17, 21, 362785),
                        )
                    ]
                    * 400,
                )
            ],
        ),
        # Test case 7: Multiple batches with exact sizes
        (
            [
                (
                    "block1",
                    [
                        VespaConcept(
                            id="1",
                            name="test",
                            parent_concepts=[],
                            parent_concept_ids_flat="",
                            model="model",
                            start=0,
                            end=1,
                            timestamp=datetime(2025, 1, 15, 16, 17, 21, 362785),
                        )
                    ]
                    * 400,
                ),
                (
                    "block2",
                    [
                        VespaConcept(
                            id="2",
                            name="test",
                            parent_concepts=[],
                            parent_concept_ids_flat="",
                            model="model",
                            start=0,
                            end=1,
                            timestamp=datetime(2025, 1, 15, 16, 17, 21, 362785),
                        )
                    ]
                    * 400,
                ),
            ],
            400,
            [
                (
                    "block1",
                    [
                        VespaConcept(
                            id="1",
                            name="test",
                            parent_concepts=[],
                            parent_concept_ids_flat="",
                            model="model",
                            start=0,
                            end=1,
                            timestamp=datetime(2025, 1, 15, 16, 17, 21, 362785),
                        )
                    ]
                    * 400,
                ),
                (
                    "block2",
                    [
                        VespaConcept(
                            id="2",
                            name="test",
                            parent_concepts=[],
                            parent_concept_ids_flat="",
                            model="model",
                            start=0,
                            end=1,
                            timestamp=datetime(2025, 1, 15, 16, 17, 21, 362785),
                        )
                    ]
                    * 400,
                ),
            ],
        ),
    ],
)
def test_split_large_concepts_updates(
    document_concepts,
    batch_size,
    expected,
):
    """Test that large concept updates are split correctly."""
    # Create test input in the format the function expects
    input_data = [(block_id, concepts) for block_id, concepts in document_concepts]

    result = split_large_concepts_updates(input_data, batch_size)

    # Verify the output
    assert len(result) == len(expected)

    for (result_block_id, result_concepts), (
        expected_block_id,
        expected_concepts,
    ) in zip(result, expected):
        assert result_block_id == expected_block_id
        assert len(result_concepts) == len(expected_concepts)
        assert all(
            actual == expected
            for actual, expected in zip(result_concepts, expected_concepts)
        )


def test_s3_paths_or_s3_prefixes_no_classifier(
    mock_bucket,
    mock_bucket_labelled_passages,
):
    """Test s3_paths_or_s3_prefixes returns base prefix when no classifier spec provided."""
    with disable_run_logger():
        config = Config(cache_bucket=mock_bucket)

        paths, prefixes = s3_paths_or_s3_prefixes(
            classifier_specs=None,
            document_ids=None,
            config=config,
        )

        assert prefixes == [f"s3://{mock_bucket}/labelled_passages"]
        assert paths is None


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
            config=config,
        )


def test_s3_paths_or_s3_prefixes_with_classifier_no_docs(
    mock_bucket,
):
    """Test s3_paths_or_s3_prefixes returns classifier-specific prefix when no document IDs provided."""
    with disable_run_logger():
        config = Config(cache_bucket=mock_bucket)
        classifier_spec_q788 = ClassifierSpec(name="Q788", alias="latest")
        classifier_spec_q699 = ClassifierSpec(name="Q699", alias="latest")

        paths, prefixes = s3_paths_or_s3_prefixes(
            classifier_specs=[classifier_spec_q788, classifier_spec_q699],
            document_ids=None,
            config=config,
        )

        assert prefixes == [
            f"s3://{mock_bucket}/labelled_passages/Q788/latest",
            f"s3://{mock_bucket}/labelled_passages/Q699/latest",
        ]
        assert paths is None


def test_s3_paths_or_s3_prefixes_with_classifier_and_docs(
    mock_bucket,
    mock_bucket_labelled_passages,
    labelled_passage_fixture_ids,
    labelled_passage_fixture_files,
    s3_prefix_mock_bucket_labelled_passages,
):
    """Test s3_paths_or_s3_prefixes returns specific paths when both classifier and document IDs provided."""
    with disable_run_logger():
        config = Config(cache_bucket=mock_bucket)
        classifier_spec = ClassifierSpec(name="Q788", alias="latest")

        expected_paths = [
            f"{s3_prefix_mock_bucket_labelled_passages}/{labelled_passage_fixture_file}"
            for labelled_passage_fixture_file in labelled_passage_fixture_files
        ]

        paths, prefixes = s3_paths_or_s3_prefixes(
            classifier_specs=[classifier_spec],
            document_ids=labelled_passage_fixture_ids,
            config=config,
        )

        assert prefixes is None
        assert sorted(paths) == sorted(expected_paths)


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
            "invalid",
            "invalid",
            ValueError,
            r"Unexpected types: `s3_prefixes=<class 'str'>`, `s3_paths=<class 'str'>`",
        ),
    ],
)
def test_s3_obj_generator_errors(
    s3_prefixes: Optional[List[str]],
    s3_paths: Optional[List[str]],
    expected_error: Type[Exception],
    error_match: str,
) -> None:
    """Test s3_obj_generator error cases."""
    with pytest.raises(expected_error, match=error_match), disable_run_logger():
        s3_obj_generator(s3_prefixes=s3_prefixes, s3_paths=s3_paths)


@pytest.mark.parametrize(
    "use_prefixes",
    [True, False],
    ids=["using_prefixes", "using_paths"],
)
def test_s3_obj_generator_valid_cases(
    mock_bucket: str,
    mock_bucket_labelled_passages: None,
    s3_prefix_labelled_passages: str,
    labelled_passage_fixture_files: List[str],
    use_prefixes: bool,
) -> None:
    """Test s3_obj_generator with valid inputs using either prefixes or paths."""
    with disable_run_logger():
        if use_prefixes:
            s3_prefixes = [
                os.path.join("s3://", mock_bucket, s3_prefix_labelled_passages)
            ]
            s3_paths = None
        else:
            s3_prefixes = None
            s3_paths = [
                os.path.join("s3://", mock_bucket, s3_prefix_labelled_passages, f)
                for f in labelled_passage_fixture_files
            ]

        gen = s3_obj_generator(s3_prefixes=s3_prefixes, s3_paths=s3_paths)
        s3_files = list(gen)

        assert len(s3_files) == len(labelled_passage_fixture_files)
        expected_keys = [
            f"{s3_prefix_labelled_passages}/{Path(f).stem}"
            for f in labelled_passage_fixture_files
        ]
        s3_files_keys = [file[0].replace(".json", "") for file in s3_files]
        assert sorted(s3_files_keys) == sorted(expected_keys)


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
