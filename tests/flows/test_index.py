import os
from pathlib import Path

from cpr_sdk.models.search import Concept as VespaConcept
from cpr_sdk.models.search import Passage
from cpr_sdk.search_adaptors import VespaSearchAdapter

from flows.index import (
    document_concepts_generator,
    get_document_passages_from_vespa,
    get_vespa_search_adapter_from_aws_secrets,
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
) -> None:
    """Test that we can retrieve all the passages for a document in vespa."""
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
    # TODO: Why 9?
    assert len(document_passages) == 9
    assert all([type(i) is Passage for i in document_passages])


# TODO: Test run partial udpates


# TODO: Test index_concepts_from_s3_to_vespa
