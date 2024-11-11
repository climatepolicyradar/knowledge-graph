import json
import os
import re
from datetime import datetime
from pathlib import Path

import pytest
from cpr_sdk.models.search import Concept as VespaConcept
from cpr_sdk.models.search import Passage
from cpr_sdk.models.search import Passage as VespaPassage
from cpr_sdk.search_adaptors import VespaSearchAdapter

from flows.index import (
    convert_labelled_passages_to_concepts,
    get_document_passages_from_vespa,
    get_parent_concepts_from_concept,
    get_passage_for_concept,
    get_updated_passage_concepts,
    get_vespa_search_adapter_from_aws_secrets,
    index_labelled_passages_from_s3_to_vespa,
    labelled_passages_generator,
    run_partial_updates_of_concepts_for_document_passages,
    s3_obj_generator_from_s3_paths,
    s3_obj_generator_from_s3_prefix,
)
from src.concept import Concept
from src.identifiers import WikibaseID
from src.labelled_passage import LabelledPassage
from src.span import Span

DOCUMENT_PASSAGE_ID_PATTERN = re.compile(
    r"id:doc_search:document_passage::[a-zA-Z]+.[a-zA-Z]+.\d+.\d+.\d+"
)
DATA_ID_PATTERN = re.compile(r"[a-zA-Z]+.[a-zA-Z]+.\d+.\d+.\d+")


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
    with open(f"{tmpdir}/cert.pem") as f:
        assert f.read() == "Public cert content\n"
    with open(f"{tmpdir}/key.pem") as f:
        assert f.read() == "Private key content\n"
    assert (
        vespa_search_adapter.instance_url
        == mock_vespa_credentials["VESPA_INSTANCE_URL"]
    )
    assert vespa_search_adapter.client.cert == f"{tmpdir}/cert.pem"
    assert vespa_search_adapter.client.key == f"{tmpdir}/key.pem"


def test_s3_obj_generator_from_s3_prefix(
    mock_bucket,
    mock_bucket_labelled_passages,
    s3_prefix_labelled_passages,
    labelled_passage_fixture_files,
) -> None:
    """Test the s3 object generator."""
    s3_gen = s3_obj_generator_from_s3_prefix(
        os.path.join("s3://", mock_bucket, s3_prefix_labelled_passages)
    )
    s3_files = list(s3_gen)
    assert len(s3_files) == len(labelled_passage_fixture_files)

    expected_keys = [
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
    s3_paths = {
        os.path.join("s3://", mock_bucket, s3_prefix_labelled_passages, f)
        for f in labelled_passage_fixture_files
    }
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
    s3_gen = s3_obj_generator_from_s3_prefix(
        os.path.join("s3://", mock_bucket, s3_prefix_labelled_passages)
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
async def test_index_labelled_passages_from_s3_to_vespa(
    mock_bucket,
    mock_bucket_labelled_passages,
    s3_prefix_labelled_passages,
    labelled_passage_fixture_files,
    mock_vespa_search_adapter: VespaSearchAdapter,
) -> None:
    """Test that we can successfully index labelled passages from s3 into vespa."""
    initial_passages_response = mock_vespa_search_adapter.client.query(
        yql="select * from document_passage where true"
    )
    initial_concepts_count = sum(
        len(hit["fields"]["concepts"]) for hit in initial_passages_response.hits
    )

    await index_labelled_passages_from_s3_to_vespa(
        s3_prefix=os.path.join("s3://", mock_bucket, s3_prefix_labelled_passages),
        vespa_search_adapter=mock_vespa_search_adapter,
    )

    final_passages_response = mock_vespa_search_adapter.client.query(
        yql="select * from document_passage where true"
    )
    final_concepts_count = sum(
        len(hit["fields"]["concepts"]) for hit in final_passages_response.hits
    )

    assert initial_concepts_count < final_concepts_count
    assert initial_concepts_count + len(labelled_passage_fixture_files) == (
        final_concepts_count
    )


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
        concept.id = text_block_id

        data_id, passage_id, passage = get_passage_for_concept(
            concept=concept, document_passages=[relevant_passage, irrelevant_passage]
        )
        assert data_id is not None
        data_id_pattern_match = DATA_ID_PATTERN.match(data_id) is not None
        assert data_id_pattern_match is not None
        assert passage_id == relevant_passage[0]
        assert passage == relevant_passage[1]


def test_get_updated_passage_concepts(
    example_vespa_concepts: list[VespaConcept],
) -> None:
    """Test that we can retrieve the updated passage concepts dict."""
    for concept in example_vespa_concepts:
        # Test we can add a concept to the passage concepts that doesn't already
        # exist.
        updated_passage_concepts = get_updated_passage_concepts(
            concept=concept,
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
            concept=concept,
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
            concept=concept,
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
    concepts = convert_labelled_passages_to_concepts(example_labelled_passages)
    assert all([isinstance(concept, VespaConcept) for concept in concepts])

    example_labelled_passage = example_labelled_passages[0].model_copy()
    example_labelled_passage.spans.append(
        Span(
            text="Test text.",
            start_index=0,
            end_index=8,
            concept_id=None,
            labellers=[],
        )
    )
    assert example_labelled_passage.spans[-1].concept_id is None
    with pytest.raises(ValueError, match="Concept ID is None."):
        convert_labelled_passages_to_concepts([example_labelled_passage])


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
