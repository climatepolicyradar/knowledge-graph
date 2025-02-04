import asyncio
import json
from io import BytesIO

import pytest
from cpr_sdk.search_adaptors import VespaSearchAdapter

from flows.count_family_document_concepts import (
    load_parse_concepts_counts,
    load_update_document_concepts_counts,
)
from src.concept import Concept
from src.identifiers import WikibaseID


@pytest.mark.asyncio
async def test_load_parse_concepts_counts(
    mock_bucket,
    mock_bucket_concepts_counts,
    mock_concepts_counts_document_keys,
) -> None:
    """Test that we can load and parse concept counts from S3."""
    mock_concepts_counts_document_uri = mock_concepts_counts_document_keys[0]
    document_object_uri = f"s3://{mock_bucket}/{ mock_concepts_counts_document_uri }"

    counter = await load_parse_concepts_counts(document_object_uri)

    # Verify the counter contains expected data
    assert len(counter) == 1

    # Get the single concept-count pair
    concept, count = next(iter(counter.items()))

    # Verify concept details
    assert isinstance(concept, Concept)
    assert concept.wikibase_id == WikibaseID("Q761")
    assert concept.preferred_label == "manufacturing sector"
    assert count == 1


@pytest.mark.asyncio
async def test_load_parse_concepts_counts_invalid_json(
    mock_bucket,
    mock_s3_client,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test handling of invalid JSON in concept counts file."""
    # Create invalid JSON file
    invalid_json = "{"  # Invalid JSON
    body = BytesIO(invalid_json.encode("utf-8"))
    key = "concepts_counts/invalid.json"
    mock_s3_client.put_object(
        Bucket=mock_bucket, Key=key, Body=body, ContentType="application/json"
    )

    document_object_uri = f"s3://{mock_bucket}/{key}"

    with pytest.raises(json.JSONDecodeError):
        await load_parse_concepts_counts(document_object_uri)


@pytest.mark.asyncio
async def test_load_parse_concepts_counts_invalid_concept_key(
    mock_bucket,
    mock_s3_client,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Test handling of invalid concept key format in JSON."""
    # Create JSON with invalid concept key format
    invalid_data = {"invalid_key": 1}
    body = BytesIO(json.dumps(invalid_data).encode("utf-8"))
    key = "concepts_counts/invalid_format.json"
    mock_s3_client.put_object(
        Bucket=mock_bucket, Key=key, Body=body, ContentType="application/json"
    )

    document_object_uri = f"s3://{mock_bucket}/{key}"

    with pytest.raises(ValueError, match="not enough values to unpack"):
        await load_parse_concepts_counts(document_object_uri)


@pytest.mark.vespa
@pytest.mark.asyncio
async def test_load_update_document_concepts_counts(
    local_vespa_search_adapter: VespaSearchAdapter,
    document_passages_test_data_file_path: str,
    vespa_app,
    mock_bucket,
    mock_bucket_concepts_counts,
) -> None:
    # Get specific id
    BATCH_SIZE = 1
    for doc_fixture_id in ["CCLW.executive.10014.4470", "CCLW.executive.4934.1571"]:
        vespa_doc_id = f"id:doc_search:family_document::{doc_fixture_id}"
        key = f"concepts_counts/Q761/v4/{doc_fixture_id}.json"
        document_object_uris = [f"s3://{mock_bucket}/{key}"]

        counts_before = local_vespa_search_adapter.get_by_id(
            vespa_doc_id
        ).concept_counts
        if counts_before is not None:
            pytest.fail(f"Fixture should not start with concept counts: {vespa_doc_id}")

        task = load_update_document_concepts_counts(
            document_import_id=doc_fixture_id,
            document_object_uris=document_object_uris,
            batch_size=1,
            vespa_search_adapter=local_vespa_search_adapter,
        )
        loaded_counts = await asyncio.gather(task, return_exceptions=True)
        assert len(loaded_counts) == BATCH_SIZE

        counts_after = local_vespa_search_adapter.get_by_id(vespa_doc_id).concept_counts
        assert counts_after == loaded_counts[0]
        assert counts_after != counts_before
