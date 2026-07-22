from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import polars as pl
import pytest
from botocore.exceptions import ClientError
from vespa.io import VespaResponse

from flows.sync_concepts import (
    concepts_to_dataframe,
    get_new_versions,
    load_concepts,
    s3_prefix_has_objects,
)
from flows.utils import S3Uri
from knowledge_graph.cloud import AwsEnv
from knowledge_graph.concept import Concept
from knowledge_graph.wikibase import WikibaseAuth


@pytest.fixture
def wikibase_auth_fixture(mock_wikibase_url: str) -> WikibaseAuth:
    return WikibaseAuth(
        username="test_user",
        password="test_password",
        url=mock_wikibase_url,
    )


@pytest.fixture
def concept_with_vespa_fields(concept) -> Concept:
    concept_dict = concept.model_dump()
    concept_dict["wikibase_revision"] = 12345
    concept_dict["wikibase_url"] = "https://test.wikibase.org/wiki/Item:Q787"
    return Concept.model_validate(concept_dict)


@pytest.fixture
def mock_vespa_response_success() -> VespaResponse:
    return VespaResponse(
        status_code=200,
        operation_type="update",
        json={
            "pathId": "/document/v1/family-document-passage/concept/docid/Q10.test_id"
        },
        url="http://localhost:8080",
    )


@pytest.fixture
def mock_vespa_response_failure() -> VespaResponse:
    return VespaResponse(
        status_code=500,
        operation_type="update",
        json={"error": "Internal server error", "message": "Failed to update document"},
        url="http://localhost:8080",
    )


def test_concepts_to_dataframe(mock_concepts):
    df = concepts_to_dataframe(mock_concepts)

    # Verify schema
    assert "id" in df.columns
    assert "wikibase_id" in df.columns
    assert "preferred_label" in df.columns
    assert "synced_at" in df.columns
    # Verify data
    assert len(df) == len(mock_concepts)

    # Verify types for Delta Lake compatibility
    assert df["description"].dtype == pl.Utf8 or all(df["description"].is_null())
    assert df["definition"].dtype == pl.Utf8 or all(df["definition"].is_null())


def test_concepts_to_dataframe__empty_concepts():
    df = concepts_to_dataframe([])

    assert df.height == 1
    assert df.columns == ["synced_at"]
    assert df.schema["synced_at"] == pl.Datetime(time_unit="us", time_zone="UTC")


def test_concepts_to_dataframe_fully_populated(fully_populated_concept):
    """
    A concept with every field populated is correctly mapped to a dataframe.

    classifier_ids is deliberately excluded from what is written to S3 (see
    concepts_to_dataframe).
    """
    df = concepts_to_dataframe([fully_populated_concept])

    assert df.height == 1

    # Values
    assert df["wikibase_id"].to_list() == [fully_populated_concept.wikibase_id]
    assert df["preferred_label"].to_list() == [fully_populated_concept.preferred_label]
    assert df["alternative_labels"].to_list() == [
        fully_populated_concept.alternative_labels
    ]
    assert df["negative_labels"].to_list() == [fully_populated_concept.negative_labels]
    assert df["subconcept_of"].to_list() == [fully_populated_concept.subconcept_of]
    assert df["has_subconcept"].to_list() == [fully_populated_concept.has_subconcept]
    assert df["related_concepts"].to_list() == [
        fully_populated_concept.related_concepts
    ]
    assert df["negative_concepts"].to_list() == [
        fully_populated_concept.negative_concepts
    ]
    assert df["recursive_subconcept_of"].to_list() == [
        fully_populated_concept.recursive_subconcept_of
    ]
    assert df["recursive_has_subconcept"].to_list() == [
        fully_populated_concept.recursive_has_subconcept
    ]
    assert df.schema["synced_at"] == pl.Datetime(time_unit="us", time_zone="UTC")


def test_concepts_to_dataframe__output_columns_are_pinned(fully_populated_concept):
    """
    Guard against unhandled Concept schema changes (ENRI-1505).

    The columns written to S3 are pinned here. If a field is added to Concept
    it will show up as a new column (or, if it can't be serialised, break the
    sync), and this test will fail - prompting a decision about whether the new
    field should be persisted or added to the exclude set in
    concepts_to_dataframe to keep the output schema consistent.
    """
    df = concepts_to_dataframe([fully_populated_concept])

    # classifier_ids and labelled_passages are deliberately excluded; id is a
    # computed field and synced_at is added by concepts_to_dataframe.
    expected_columns = {
        "id",
        "preferred_label",
        "alternative_labels",
        "negative_labels",
        "description",
        "definition",
        "wikibase_id",
        "wikibase_revision",
        "subconcept_of",
        "has_subconcept",
        "related_concepts",
        "negative_concepts",
        "recursive_subconcept_of",
        "recursive_has_subconcept",
        "synced_at",
    }
    assert set(df.columns) == expected_columns


def test_concepts_to_dataframe__schema_stable_across_files(
    mock_concepts, fully_populated_concept, tmp_path
):
    """
    Empty and populated concepts must produce the same Parquet schema (ENRI-1505).

    The flow scans the archive as a glob of Parquet files written by separate
    runs. Without explicit dtypes an empty list field infers List(Null) while a
    populated one infers List(String), so files from different runs cannot be
    read together. Write both and read them back as one to prove they union.
    """
    concepts_to_dataframe(mock_concepts).write_parquet(tmp_path / "empty.parquet")
    concepts_to_dataframe([fully_populated_concept]).write_parquet(
        tmp_path / "populated.parquet"
    )

    combined = pl.read_parquet(tmp_path / "*.parquet")

    assert len(combined) == len(mock_concepts) + 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "existing_count,expected_new",
    [
        (3, 0),  # No new when all exist
        (1, 2),  # Partial changes
    ],
)
async def test_get_new_versions__parameterized(
    mock_concepts, existing_count, expected_new, tmp_path
):
    current_df = concepts_to_dataframe(mock_concepts).lazy()

    # Create existing IDs based on existing_count
    existing_concepts = mock_concepts[:existing_count]
    existing_ids = concepts_to_dataframe(existing_concepts).select("id").lazy()

    new_versions = await get_new_versions(current_df, existing_ids)

    assert len(new_versions) == expected_new


@pytest.mark.asyncio
async def test_get_new_versions__all_new(mock_concepts, tmp_path):
    """Test when all concepts are new (empty existing_ids LazyFrame)."""
    current_df = concepts_to_dataframe(mock_concepts).lazy()
    # Empty existing_ids means all concepts are new
    empty_df = pl.DataFrame({"id": []}, schema={"id": pl.String})
    existing_ids = empty_df.lazy()

    new_versions = await get_new_versions(current_df, existing_ids)

    assert len(new_versions) == len(mock_concepts)


@pytest.mark.asyncio
async def test_get_new_versions__no_archive(mock_concepts, tmp_path):
    """Test when no archive exists (e.g., in a new environment)."""
    current_df = concepts_to_dataframe(mock_concepts).lazy()

    # Pass None for existing_ids to indicate no archive
    new_versions = await get_new_versions(current_df, None)

    # Should return all concepts as new when no archive exists
    # Exclude synced_at column as it varies between calls
    cols_to_compare = [c for c in new_versions.columns if c != "synced_at"]
    expected_df = concepts_to_dataframe(mock_concepts)
    assert (
        new_versions.select(cols_to_compare)
        .sort("id")
        .equals(expected_df.select(cols_to_compare).sort("id"))
    )


@pytest.mark.asyncio
async def test_s3_prefix_has_objects__with_objects(mock_s3_async_client):
    """Test s3_prefix_has_objects returns True when objects exist."""
    bucket_name = "test-bucket"
    prefix = "test-prefix"

    # Create bucket and add object
    await mock_s3_async_client.create_bucket(
        Bucket=bucket_name,
        CreateBucketConfiguration={"LocationConstraint": "eu-west-1"},
    )
    await mock_s3_async_client.put_object(
        Bucket=bucket_name, Key=f"{prefix}/test.parquet", Body=b"test"
    )

    # Mock get_async_session to return a session that yields our mock client
    @asynccontextmanager
    async def mock_client(*args, **kwargs):
        yield mock_s3_async_client

    mock_session = AsyncMock()
    mock_session.client = mock_client

    with patch("flows.sync_concepts.get_async_session", return_value=mock_session):
        s3_uri = S3Uri(bucket=bucket_name, key=prefix)
        result = await s3_prefix_has_objects(s3_uri, "eu-west-1", AwsEnv.sandbox)

    assert result is True


@pytest.mark.asyncio
async def test_s3_prefix_has_objects__no_objects(mock_s3_async_client):
    """Test s3_prefix_has_objects returns False when no objects exist at prefix."""
    bucket_name = "test-bucket-empty"
    prefix = "empty-prefix"

    # Create bucket but don't add any objects
    await mock_s3_async_client.create_bucket(
        Bucket=bucket_name,
        CreateBucketConfiguration={"LocationConstraint": "eu-west-1"},
    )

    # Mock get_async_session to return a session that yields our mock client
    @asynccontextmanager
    async def mock_client(*args, **kwargs):
        yield mock_s3_async_client

    mock_session = AsyncMock()
    mock_session.client = mock_client

    with patch("flows.sync_concepts.get_async_session", return_value=mock_session):
        s3_uri = S3Uri(bucket=bucket_name, key=prefix)
        result = await s3_prefix_has_objects(s3_uri, "eu-west-1", AwsEnv.sandbox)

    assert result is False


@pytest.mark.asyncio
async def test_s3_prefix_has_objects__error_propagates():
    """Test s3_prefix_has_objects lets genuine errors propagate."""
    # Create a mock client that raises AccessDenied
    mock_s3 = AsyncMock()
    mock_s3.list_objects_v2.side_effect = ClientError(
        {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}},
        "ListObjectsV2",
    )

    @asynccontextmanager
    async def mock_client(*args, **kwargs):
        yield mock_s3

    mock_session = AsyncMock()
    mock_session.client = mock_client

    with patch("flows.sync_concepts.get_async_session", return_value=mock_session):
        s3_uri = S3Uri(bucket="test-bucket", key="test-prefix")

        # Should raise the ClientError, not catch it
        with pytest.raises(ClientError) as exc_info:
            await s3_prefix_has_objects(s3_uri, "eu-west-1", AwsEnv.sandbox)

        assert exc_info.value.response["Error"]["Code"] == "AccessDenied"


@pytest.mark.asyncio
async def test_load_concepts__from_cache(
    tmp_path, mock_concepts, wikibase_auth_fixture
):
    cache_path = tmp_path / "cache.jsonl"

    # Write cache
    with open(cache_path, "w") as f:
        for concept in mock_concepts:
            f.write(concept.model_dump_json() + "\n")

    concepts = await load_concepts(
        wikibase_auth=wikibase_auth_fixture,
        wikibase_cache_path=cache_path,
        wikibase_cache_save_if_missing=False,
    )

    assert len(concepts) == len(mock_concepts)

    # Sort both lists by ID for stable comparison
    concepts_sorted = sorted(concepts, key=lambda c: c.id)
    mock_concepts_sorted = sorted(mock_concepts, key=lambda c: c.id)

    assert concepts_sorted[0].id == mock_concepts_sorted[0].id
    # Verify all IDs match
    assert [c.id for c in concepts_sorted] == [c.id for c in mock_concepts_sorted]
