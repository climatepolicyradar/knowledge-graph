import json
import os
from datetime import datetime
from pathlib import Path

import pytest
from cpr_sdk.models.search import Concept as VespaConcept
from cpr_sdk.models.search import Passage
from cpr_sdk.search_adaptors import VespaSearchAdapter

from flows.index import (
    document_concepts_generator,
    get_document_passages_from_vespa,
    get_vespa_search_adapter_from_aws_secrets,
    run_partial_updates_of_concepts_for_document_passages,
    s3_obj_generator,
)


def test_vespa_search_adapter_from_aws_secrets(
    create_vespa_params, mock_vespa_credentials, tmpdir
) -> None:
    """Test that we can successfully instantiate the VespaSearchAdpater from ssm params."""
    vespa_search_adapter = get_vespa_search_adapter_from_aws_secrets(cert_dir=tmpdir)

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
    with open(document_passages_test_data_file_path) as f:
        document_passage_test_data = json.load(f)

    family_document_passages_count_expected = sum(
        1
        for doc in document_passage_test_data
        if doc["fields"]["family_document_ref"]
        == "id:doc_search:family_document::CCLW.executive.10014.4470"
    )

    document_passages = get_document_passages_from_vespa(
        document_import_id="test.executive.1.1",
        vespa_search_adapter=mock_vespa_search_adapter,
    )

    assert document_passages == []

    document_passages = get_document_passages_from_vespa(
        document_import_id="CCLW.executive.10014.4470",
        vespa_search_adapter=mock_vespa_search_adapter,
    )

    assert len(document_passages) > 0
    assert len(document_passages) == family_document_passages_count_expected
    assert all([type(i) is Passage for i in document_passages])


@pytest.mark.asyncio
async def test_run_partial_updates_of_concepts_for_document_passages(
    mock_vespa_search_adapter: VespaSearchAdapter,
) -> None:
    """Test that we can run partial updates of concepts for document passages."""
    new_vespa_concepts = [
        VespaConcept(
            id="Q788-RuleBasedClassifier.1457",
            name="Q788-RuleBasedClassifier",
            parent_concepts=[
                {"name": "RuleBasedClassifier", "id": "Q788"},
                {"name": "RuleBasedClassifier", "id": "Q789"},
            ],
            parent_concept_ids_flat="Q788,Q789",
            model="RuleBasedClassifier",
            end=100,
            start=0,
            timestamp=datetime.now(),
        ),
        VespaConcept(
            id="Q788-RuleBasedClassifier.1273",
            name="Q788-RuleBasedClassifier",
            parent_concepts=[
                {"name": "Q1-RuleBasedClassifier", "id": "Q2"},
                {"name": "Q2-RuleBasedClassifier", "id": "Q3"},
            ],
            parent_concept_ids_flat="Q2,Q3",
            model="RuleBasedClassifier-2.0.12",
            end=100,
            start=0,
            timestamp=datetime.now(),
        ),
    ]

    document_passages_initial = get_document_passages_from_vespa(
        document_import_id="CCLW.executive.10014.4470",
        vespa_search_adapter=mock_vespa_search_adapter,
    )

    document_passages_initial__concepts = [
        concept
        for passage in document_passages_initial
        if passage.concepts
        for concept in passage.concepts
    ]

    assert len(document_passages_initial) > 0
    assert all(
        concept not in document_passages_initial__concepts
        for concept in new_vespa_concepts
    )

    await run_partial_updates_of_concepts_for_document_passages(
        document_import_id="CCLW.executive.10014.4470",
        document_concepts=new_vespa_concepts,
        vespa_search_adapter=mock_vespa_search_adapter,
    )

    document_passages_updated = get_document_passages_from_vespa(
        document_import_id="CCLW.executive.10014.4470",
        vespa_search_adapter=mock_vespa_search_adapter,
    )

    document_passages_updated__concepts = [
        concept
        for passage in document_passages_updated
        if passage.concepts
        for concept in passage.concepts
    ]

    assert len(document_passages_updated) > 0
    assert len(document_passages_updated__concepts) != len(
        document_passages_initial__concepts
    )
    assert all(
        [
            any([new_vespa_concept == c for c in document_passages_updated__concepts])
            for new_vespa_concept in new_vespa_concepts
        ]
    )


# TODO: Test index_concepts_from_s3_to_vespa
