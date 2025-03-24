import json
import os
import re
from pathlib import Path

import pytest
from cpr_sdk.models.search import Concept as VespaConcept
from cpr_sdk.models.search import Passage as VespaPassage
from cpr_sdk.search_adaptors import VespaSearchAdapter
from prefect.logging import disable_run_logger

from flows.boundary import (
    DocumentImporter,
    convert_labelled_passage_to_concepts,
    get_data_id_from_vespa_hit_id,
    get_document_passages_from_vespa,
    get_parent_concepts_from_concept,
    get_vespa_search_adapter_from_aws_secrets,
    s3_obj_generator,
    s3_obj_generator_from_s3_paths,
    s3_obj_generator_from_s3_prefixes,
    s3_paths_or_s3_prefixes,
)
from flows.index import Config
from scripts.cloud import ClassifierSpec
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

    with (
        pytest.raises(
            ValueError,
            match="if document IDs are specified, a classifier "
            "specifcation must also be specified, since they're "
            "namespaced by classifiers \\(e\\.g\\. "
            "`s3://cpr-sandbox-data-pipeline-cache/labelled_passages/Q787/"
            "v4/CCLW\\.legislative\\.10695\\.6015\\.json`\\)",
        ),
        disable_run_logger(),
    ):
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
    classifier_spec_q788 = ClassifierSpec(name="Q788", alias="v4")
    classifier_spec_q699 = ClassifierSpec(name="Q699", alias="v4")

    s3_accessor = s3_paths_or_s3_prefixes(
        classifier_specs=[classifier_spec_q788, classifier_spec_q699],
        document_ids=None,
        cache_bucket=config.cache_bucket,  # pyright: ignore[reportArgumentType]
        prefix=config.document_source_prefix,
    )

    assert s3_accessor.prefixes == [
        f"s3://{mock_bucket}/labelled_passages/Q788/v4",
        f"s3://{mock_bucket}/labelled_passages/Q699/v4",
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
    classifier_spec = ClassifierSpec(name="Q788", alias="v4")

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
