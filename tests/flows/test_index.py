import json
import os
import re
from pathlib import Path

import pytest
from cpr_sdk.models.search import Concept as VespaConcept
from cpr_sdk.models.search import Passage
from cpr_sdk.models.search import Passage as VespaPassage
from cpr_sdk.search_adaptors import VespaSearchAdapter

from flows.index import (
    document_concepts_generator,
    get_document_passages_from_vespa,
    get_passage_for_concept,
    get_vespa_search_adapter_from_aws_secrets,
    index_concepts_from_s3_to_vespa,
    run_partial_updates_of_concepts_for_document_passages,
    s3_obj_generator,
)

DOCUMENT_PASSAGE_ID_PATTERN = re.compile(
    r"id:doc_search:document_passage::[a-zA-Z]+.[a-zA-Z]+.\d+.\d+.\d+"
)


def test_vespa_search_adapter_from_aws_secrets(
    create_vespa_params, mock_vespa_credentials, tmpdir
) -> None:
    """Test that we can successfully instantiate the VespaSearchAdpater from ssm params."""
    vespa_search_adapter = get_vespa_search_adapter_from_aws_secrets(
        cert_dir=tmpdir,
        vespa_public_cert_param_name="VESPA_PUBLIC_CERT_FULL_ACCESS",
        vespa_private_key_param_name="VESPA_PRIVATE_KEY_FULL_ACCESS",
    )

    assert os.path.exists(f"{tmpdir}/cert.pem")
    assert os.path.exists(f"{tmpdir}/key.pem")
    assert (
        vespa_search_adapter.instance_url
        == mock_vespa_credentials["VESPA_INSTANCE_URL"]
    )
    assert vespa_search_adapter.client.cert == f"{tmpdir}/cert.pem"
    assert vespa_search_adapter.client.key == f"{tmpdir}/key.pem"


def test_s3_obj_generator(
    mock_bucket,
    mock_bucket_concepts,
    s3_prefix_concepts,
    concept_fixture_files,
) -> None:
    """Test the s3 object generator."""
    s3_gen = s3_obj_generator(os.path.join("s3://", mock_bucket, s3_prefix_concepts))
    s3_files = list(s3_gen)
    assert len(s3_files) == len(concept_fixture_files)

    expected_keys = [
        f"{s3_prefix_concepts}/{Path(f).stem}" for f in concept_fixture_files
    ]
    s3_files_keys = [file[0].replace(".json", "") for file in s3_files]

    assert sorted(s3_files_keys) == sorted(expected_keys)


def test_document_concepts_generator(
    mock_bucket,
    mock_bucket_concepts,
    s3_prefix_concepts,
    concept_fixture_files,
) -> None:
    """Test that the document concepts generator yields the correct objects."""
    s3_gen = s3_obj_generator(os.path.join("s3://", mock_bucket, s3_prefix_concepts))
    document_concepts_gen = document_concepts_generator(generator_func=s3_gen)
    document_concepts_files = list(document_concepts_gen)
    expected_keys = [
        f"{s3_prefix_concepts}/{Path(f).stem}.json" for f in concept_fixture_files
    ]

    assert len(document_concepts_files) == len(concept_fixture_files)
    for s3_key, document_concepts in document_concepts_files:
        assert all([type(i) is VespaConcept for i in document_concepts])
        assert s3_key in expected_keys


def test_get_document_passages_from_vespa(
    mock_vespa_search_adapter: VespaSearchAdapter,
    document_passages_test_data_file_path: str,
) -> None:
    """Test that we can retrieve all the passages for a document in vespa."""

    # Test that we retrieve no passages for a document that doesn't exist
    document_passages = get_document_passages_from_vespa(
        document_import_id="test.executive.1.1",
        vespa_search_adapter=mock_vespa_search_adapter,
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
        vespa_search_adapter=mock_vespa_search_adapter,
    )

    assert len(document_passages) > 0
    assert len(document_passages) == family_document_passages_count_expected
    assert all(
        [
            (
                type(passage) is Passage
                and type(passage_id) is str
                and bool(DOCUMENT_PASSAGE_ID_PATTERN.fullmatch(passage_id))
            )
            for passage_id, passage in document_passages
        ]
    )


@pytest.mark.asyncio
async def test_run_partial_updates_of_concepts_for_document_passages(
    mock_vespa_search_adapter: VespaSearchAdapter,
    example_vespa_concepts: list[VespaConcept],
) -> None:
    """Test that we can run partial updates of concepts for document passages."""
    document_import_id = "CCLW.executive.10014.4470"
    initial_passages = get_document_passages_from_vespa(
        document_import_id=document_import_id,
        vespa_search_adapter=mock_vespa_search_adapter,
    )
    initial_concepts = [
        concept
        for _, passage in initial_passages
        if passage.concepts
        for concept in passage.concepts
    ]

    assert len(initial_passages) > 0
    assert all(concept not in initial_concepts for concept in example_vespa_concepts)

    await run_partial_updates_of_concepts_for_document_passages(
        document_import_id=document_import_id,
        document_concepts=example_vespa_concepts,
        vespa_search_adapter=mock_vespa_search_adapter,
    )

    updated_passages = get_document_passages_from_vespa(
        document_import_id=document_import_id,
        vespa_search_adapter=mock_vespa_search_adapter,
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


@pytest.mark.asyncio
async def test_index_concepts_from_s3_to_vespa(
    mock_bucket,
    mock_bucket_concepts,
    s3_prefix_concepts,
    concept_fixture_files,
    mock_vespa_search_adapter: VespaSearchAdapter,
) -> None:
    """Test that we can successfully index concepts from s3 into vespa."""
    initial_passages_response = mock_vespa_search_adapter.client.query(
        yql="select * from document_passage where true"
    )
    initial_concepts_count = sum(
        len(hit["fields"]["concepts"]) for hit in initial_passages_response.hits
    )

    await index_concepts_from_s3_to_vespa(
        s3_path=os.path.join("s3://", mock_bucket, s3_prefix_concepts),
        vespa_search_adapter=mock_vespa_search_adapter,
    )

    final_passages_response = mock_vespa_search_adapter.client.query(
        yql="select * from document_passage where true"
    )
    final_concepts_count = sum(
        len(hit["fields"]["concepts"]) for hit in final_passages_response.hits
    )

    assert initial_concepts_count < final_concepts_count
    assert initial_concepts_count + len(concept_fixture_files) == (final_concepts_count)


@pytest.mark.parametrize("text_block_id", ["1457", "p_2_b_120"])
def test_get_passage_for_concept(
    example_vespa_concepts: list[VespaConcept], text_block_id: str
) -> None:
    """Test that we can retrieve the relevant passage for a concept."""
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
        concept.id = f"Q788-RuleBasedClassifier.{text_block_id}"

        passage_id, passage = get_passage_for_concept(
            concept=concept, document_passages=[relevant_passage, irrelevant_passage]
        )

        assert passage_id == relevant_passage[0]
        assert passage == relevant_passage[1]
