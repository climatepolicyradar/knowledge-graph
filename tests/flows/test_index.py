import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
from cpr_sdk.models.search import Concept as VespaConcept
from cpr_sdk.models.search import Passage as VespaPassage
from cpr_sdk.search_adaptors import VespaSearchAdapter
from prefect.logging import disable_run_logger

from flows.boundary import (
    ConceptModel,
    get_document_passage_from_vespa,
    get_document_passages_from_vespa,
    index_by_s3,
    partial_update_text_block,
)
from flows.index import (
    CONCEPTS_COUNTS_PREFIX_DEFAULT,
    Config,
    index_labelled_passages_from_s3_to_vespa,
    run_partial_updates_of_concepts_for_document_passages__update,
    update_concepts_on_existing_vespa_concepts,
)
from scripts.cloud import (
    AwsEnv,
    ClassifierSpec,
)
from src.identifiers import WikibaseID
from tests.flows.test_boundary import DOCUMENT_PASSAGE_ID_PATTERN


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

    update_result = (
        await run_partial_updates_of_concepts_for_document_passages__update.fn(
            document_importer=(document_import_id, document_object_uri),
            vespa_search_adapter=local_vespa_search_adapter,
            cache_bucket=mock_bucket,
            concepts_counts_prefix=CONCEPTS_COUNTS_PREFIX_DEFAULT,
        )
    )

    assert test_counts == update_result
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
            == await run_partial_updates_of_concepts_for_document_passages__update.fn(
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
        partial_update_flow=run_partial_updates_of_concepts_for_document_passages__update,
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
        partial_update_flow=run_partial_updates_of_concepts_for_document_passages__update,
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
        "flows.boundary.run_partial_updates_of_concepts_for_batch_flow_or_deployment",
        side_effect=mock_run_partial_updates_of_concepts_for_batch_flow_or_deployment,
    ):
        await index_by_s3(
            partial_update_flow=run_partial_updates_of_concepts_for_document_passages__update,
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
        update_concepts_on_existing_vespa_concepts,
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
        update_concepts_on_existing_vespa_concepts,
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
    with (
        patch("flows.index.get_prefect_job_variable", return_value=mock_bucket),
        patch(
            "flows.boundary.get_vespa_search_adapter_from_aws_secrets",
            return_value=local_vespa_search_adapter,
        ),
        disable_run_logger(),
    ):
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


def test_update_concepts_on_existing_vespa_concepts(
    example_vespa_concepts: list[VespaConcept],
) -> None:
    """Test that we can retrieve the updated passage concepts dict."""
    for concept in example_vespa_concepts:
        # Test we can add a concept to the passage concepts that doesn't already
        # exist.
        updated_passage_concepts = update_concepts_on_existing_vespa_concepts(
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
        updated_passage_concepts = update_concepts_on_existing_vespa_concepts(
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
        updated_passage_concepts = update_concepts_on_existing_vespa_concepts(
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
