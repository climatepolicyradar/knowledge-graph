import json
from io import BytesIO

import pytest

from flows.count_family_document_concepts import load_parse_concepts_counts
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
