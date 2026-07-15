from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import polars as pl
import pytest
from botocore.exceptions import ClientError
from pydantic import ValidationError as PydanticValidationError
from vespa.io import VespaResponse

from flows.sync_concepts import (
    concepts_to_dataframe,
    create_vespa_sync_summary_artifact,
    dataframe_to_concepts,
    get_new_versions,
    load_concepts,
    s3_prefix_has_objects,
    send_concept_validation_alert,
    update_concept_in_vespa,
    update_concepts_in_vespa,
)
from flows.utils import S3Uri
from knowledge_graph.cloud import AwsEnv
from knowledge_graph.concept import Concept
from knowledge_graph.result import Err, Error, Ok, is_err, is_ok, unwrap_err, unwrap_ok
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


def test_dataframe_to_concepts__roundtrip(mock_concepts):
    """Test roundtrip: concepts -> DataFrame -> concepts."""
    df = concepts_to_dataframe(mock_concepts)
    recovered = dataframe_to_concepts(df)

    assert len(recovered) == len(mock_concepts)
    for original, recovered_concept in zip(mock_concepts, recovered):
        assert original.id == recovered_concept.id
        assert original.wikibase_id == recovered_concept.wikibase_id
        assert original.preferred_label == recovered_concept.preferred_label


def test_dataframe_to_concepts__empty_dataframe():
    df = concepts_to_dataframe([])

    assert [] == dataframe_to_concepts(df)


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


@pytest.mark.asyncio
async def test_create_vespa_sync_summary_artifact__all_success(mock_concepts):
    results = [Ok(concept) for concept in mock_concepts]

    with patch(
        "flows.sync_concepts.acreate_table_artifact", new_callable=AsyncMock
    ) as mock_create:
        await create_vespa_sync_summary_artifact(
            results=results,
            parquet_path="/path/to/concepts_20250101_120000.parquet",
            aws_env=AwsEnv.staging,
        )

        # Verify artifact was created
        mock_create.assert_called_once()
        call_args = mock_create.call_args

        # Verify the table data
        table = call_args.kwargs["table"]
        assert len(table) == len(mock_concepts)
        assert all(row["Status"] == "✓" for row in table)
        assert all(row["Error"] == "N/A" for row in table)

        # Verify Parquet path in description
        description = call_args.kwargs["description"]
        assert "concepts_20250101_120000.parquet" in description


@pytest.mark.asyncio
async def test_create_vespa_sync_summary_artifact__all_failures(mock_concepts):
    results = [
        Err(
            Error(
                msg="Vespa update failed",
                metadata={"concept": concept.model_dump(), "response": None},
            )
        )
        for concept in mock_concepts
    ]

    with patch(
        "flows.sync_concepts.acreate_table_artifact", new_callable=AsyncMock
    ) as mock_create:
        await create_vespa_sync_summary_artifact(
            results=results,
            parquet_path="/path/to/file.parquet",
            aws_env=AwsEnv.staging,
        )

        # Verify artifact was created
        mock_create.assert_called_once()
        call_args = mock_create.call_args

        # Verify the table data
        table = call_args.kwargs["table"]
        assert len(table) == len(mock_concepts)
        assert all(row["Status"] == "✗" for row in table)
        assert all("Vespa update failed" in row["Error"] for row in table)


@pytest.mark.asyncio
async def test_create_vespa_sync_summary_artifact__mixed(mock_concepts):
    results = [
        Ok(mock_concepts[0]),
        Err(
            Error(
                msg="Vespa update failed",
                metadata={"concept": mock_concepts[1].model_dump(), "response": None},
            )
        ),
        Ok(mock_concepts[2]),
    ]

    with patch(
        "flows.sync_concepts.acreate_table_artifact", new_callable=AsyncMock
    ) as mock_create:
        await create_vespa_sync_summary_artifact(
            results=results,
            parquet_path="/path/to/file.parquet",
            aws_env=AwsEnv.staging,
        )

        # Verify artifact was created
        mock_create.assert_called_once()
        call_args = mock_create.call_args

        # Verify the table data
        table = call_args.kwargs["table"]
        assert len(table) == 3
        # Table is ordered: successes first, then failures
        assert table[0]["Status"] == "✓"
        assert table[1]["Status"] == "✓"
        assert table[2]["Status"] == "✗"


@pytest.mark.asyncio
async def test_create_vespa_sync_summary_artifact__error_with_vespa_response(
    mock_concepts, mock_vespa_response_failure
):
    results = [
        Err(
            Error(
                msg="Vespa update failed",
                metadata={
                    "concept": mock_concepts[0].model_dump(),
                    "response": mock_vespa_response_failure,
                },
            )
        )
    ]

    with patch(
        "flows.sync_concepts.acreate_table_artifact", new_callable=AsyncMock
    ) as mock_create:
        await create_vespa_sync_summary_artifact(
            results=results,
            parquet_path="/path/to/file.parquet",
            aws_env=AwsEnv.staging,
        )

        # Verify artifact was created
        call_args = mock_create.call_args
        table = call_args.kwargs["table"]

        # Verify error includes JSON response
        error_msg = table[0]["Error"]
        assert "Vespa update failed" in error_msg
        assert "Internal server error" in error_msg


@pytest.mark.asyncio
async def test_create_vespa_sync_summary_artifact__error_without_metadata():
    results = [Err(Error(msg="Some error", metadata=None))]

    with patch(
        "flows.sync_concepts.acreate_table_artifact", new_callable=AsyncMock
    ) as mock_create:
        await create_vespa_sync_summary_artifact(
            results=results,
            parquet_path="/path/to/file.parquet",
            aws_env=AwsEnv.staging,
        )

        # Should not raise an error
        mock_create.assert_called_once()
        call_args = mock_create.call_args
        table = call_args.kwargs["table"]

        assert table[0]["Concept ID"] == "Unknown"
        assert table[0]["Error"] == "Some error"


@pytest.mark.asyncio
async def test_create_vespa_sync_summary_artifact__no_parquet_path():
    results = [Err(Error(msg="Vespa update failed", metadata=None))]

    with patch(
        "flows.sync_concepts.acreate_table_artifact", new_callable=AsyncMock
    ) as mock_create:
        await create_vespa_sync_summary_artifact(
            results=results, parquet_path=None, aws_env=AwsEnv.staging
        )

        # Verify artifact was created
        mock_create.assert_called_once()
        call_args = mock_create.call_args

        # Verify description shows no successful syncs
        description = call_args.kwargs["description"]
        assert "None (no successful syncs)" in description
        assert "concepts_" not in description  # No Parquet filename


@pytest.mark.asyncio
async def test_update_concept_in_vespa__success(
    concept_with_vespa_fields, mock_vespa_response_success
):
    """Test successful Vespa update returns Ok."""
    mock_pool = AsyncMock()
    mock_pool.update_data = AsyncMock(return_value=mock_vespa_response_success)
    mock_pool.app.get_document_v1_path = AsyncMock(return_value="/document/v1/...")

    result = await update_concept_in_vespa(concept_with_vespa_fields, mock_pool)

    assert is_ok(result), f"Expected Ok result, got {result}"
    returned_concept = unwrap_ok(result)
    assert returned_concept.id == concept_with_vespa_fields.id


@pytest.mark.asyncio
async def test_update_concept_in_vespa__vespa_failure(
    concept_with_vespa_fields, mock_vespa_response_failure
):
    mock_pool = AsyncMock()
    mock_pool.update_data = AsyncMock(return_value=mock_vespa_response_failure)
    mock_pool.app.get_document_v1_path = AsyncMock(return_value="/document/v1/...")

    result = await update_concept_in_vespa(concept_with_vespa_fields, mock_pool)

    assert is_err(result), f"Expected Err result, got {result}"
    error = unwrap_err(result)
    assert error.msg == "Vespa update failed"
    assert "response" in error.metadata


@pytest.mark.vespa
@pytest.mark.asyncio
async def test_update_concept_in_vespa__real_vespa_success(
    concept_with_vespa_fields, local_vespa_search_adapter
):
    async with local_vespa_search_adapter.client.asyncio(
        connections=1
    ) as vespa_connection_pool:
        result = await update_concept_in_vespa(
            concept_with_vespa_fields, vespa_connection_pool
        )

        assert is_ok(result), f"Expected Ok result, got {result}"
        returned_concept = unwrap_ok(result)
        assert returned_concept.id == concept_with_vespa_fields.id


@pytest.mark.vespa
@pytest.mark.asyncio
async def test_update_concept_in_vespa__real_vespa_idempotent(
    concept_with_vespa_fields, local_vespa_search_adapter
):
    async with local_vespa_search_adapter.client.asyncio(
        connections=1
    ) as vespa_connection_pool:
        # First update
        result1 = await update_concept_in_vespa(
            concept_with_vespa_fields, vespa_connection_pool
        )
        assert is_ok(result1)

        # Second update (should also succeed)
        result2 = await update_concept_in_vespa(
            concept_with_vespa_fields, vespa_connection_pool
        )
        assert is_ok(result2)


@pytest.mark.asyncio
async def test_update_concept_in_vespa__empty_description_handled(
    concept_with_vespa_fields, mock_vespa_response_success
):
    """Test that empty string description is converted to None and doesn't cause validation error."""
    # Create concept with empty description
    concept_dict = concept_with_vespa_fields.model_dump()
    concept_dict["description"] = ""
    concept_dict["definition"] = ""
    concept_with_empty = Concept.model_validate(concept_dict)

    mock_pool = AsyncMock()
    mock_pool.update_data = AsyncMock(return_value=mock_vespa_response_success)
    mock_pool.app.get_document_v1_path = AsyncMock(return_value="/document/v1/...")

    result = await update_concept_in_vespa(concept_with_empty, mock_pool)

    assert is_ok(result), f"Expected Ok result, got {result}"
    returned_concept = unwrap_ok(result)
    assert returned_concept.id == concept_with_empty.id


@pytest.mark.asyncio
async def test_update_concept_in_vespa__validation_error_caught(concept):
    """Test that validation errors are caught and returned as Err."""
    # Create concept with empty preferred_label which violates min_length=1
    concept_dict = concept.model_dump()
    concept_dict["wikibase_revision"] = 1
    concept_dict["wikibase_url"] = "https://example.com"
    concept_dict["preferred_label"] = ""  # Empty string violates min_length=1

    # We need to bypass Concept validation to create an invalid concept
    # by using model_construct which skips validation
    invalid_concept = Concept.model_construct(**concept_dict)

    mock_pool = AsyncMock()

    result = await update_concept_in_vespa(invalid_concept, mock_pool)

    assert is_err(result), f"Expected Err result, got {result}"
    error = unwrap_err(result)
    assert "Failed to create VespaConcept" in error.msg
    assert "concept" in error.metadata


@pytest.mark.asyncio
async def test_update_concept_in_vespa__unexpected_error_caught(
    concept_with_vespa_fields,
):
    """Test that unexpected errors during Vespa update are caught."""
    mock_pool = AsyncMock()
    mock_pool.update_data = AsyncMock(side_effect=RuntimeError("Unexpected error"))
    mock_pool.app.get_document_v1_path = AsyncMock(return_value="/document/v1/...")

    result = await update_concept_in_vespa(concept_with_vespa_fields, mock_pool)

    assert is_err(result), f"Expected Err result, got {result}"
    error = unwrap_err(result)
    assert "Unexpected error during Vespa update" in error.msg
    assert "Unexpected error" in error.msg


@pytest.mark.asyncio
async def test_update_concepts_in_vespa__all_success(
    mock_concepts, mock_vespa_response_success
):
    """Test update_concepts_in_vespa subflow with all successful updates."""
    # Add required Vespa fields to all concepts
    concepts_with_vespa = []
    for concept in mock_concepts:
        concept_dict = concept.model_dump()
        concept_dict["wikibase_revision"] = 12345
        concept_dict["wikibase_url"] = "https://test.wikibase.org/wiki/Item:Q787"
        concepts_with_vespa.append(Concept.model_validate(concept_dict))

    mock_pool = AsyncMock()
    mock_pool.update_data = AsyncMock(return_value=mock_vespa_response_success)
    mock_pool.app.get_document_v1_path = AsyncMock(return_value="/document/v1/...")

    # Call the underlying function directly to avoid Prefect serialization issues
    results = await update_concepts_in_vespa(concepts_with_vespa, mock_pool)

    assert len(results) == len(mock_concepts)
    assert all(is_ok(r) for r in results)


@pytest.mark.asyncio
async def test_update_concepts_in_vespa__mixed_results(
    mock_concepts, mock_vespa_response_success, mock_vespa_response_failure
):
    """Test update_concepts_in_vespa subflow with mixed success/failure."""
    # Add required Vespa fields to all concepts
    concepts_with_vespa = []
    for i, concept in enumerate(mock_concepts):
        concept_dict = concept.model_dump()
        concept_dict["wikibase_revision"] = 12345
        concept_dict["wikibase_url"] = "https://test.wikibase.org/wiki/Item:Q787"
        concepts_with_vespa.append(Concept.model_validate(concept_dict))

    mock_pool = AsyncMock()
    # First succeeds, second fails, third succeeds
    mock_pool.update_data = AsyncMock(
        side_effect=[
            mock_vespa_response_success,
            mock_vespa_response_failure,
            mock_vespa_response_success,
        ]
    )
    mock_pool.app.get_document_v1_path = AsyncMock(return_value="/document/v1/...")

    results = await update_concepts_in_vespa(concepts_with_vespa, mock_pool)

    assert len(results) == 3
    assert is_ok(results[0])
    assert is_err(results[1])
    assert is_ok(results[2])


@pytest.mark.asyncio
async def test_update_concepts_in_vespa__missing_required_fields(mock_concepts):
    """Test update_concepts_in_vespa with concepts missing required fields."""
    # First concept missing wikibase_revision, second missing wikibase_url
    concepts = [
        mock_concepts[0],  # No Vespa fields
        mock_concepts[1],  # No Vespa fields
    ]

    mock_pool = AsyncMock()

    results = await update_concepts_in_vespa(concepts, mock_pool)

    assert len(results) == 2
    assert all(is_err(r) for r in results)

    # Check error messages
    error1 = unwrap_err(results[0])
    assert "missing Wikibase revision" in error1.msg

    error2 = unwrap_err(results[1])
    assert "missing Wikibase revision" in error2.msg


@pytest.mark.asyncio
async def test_update_concepts_in_vespa__continues_after_failure(
    mock_concepts, mock_vespa_response_success
):
    """Test that update_concepts_in_vespa continues processing after a failure."""
    # Create 3 concepts: valid, missing field, valid
    valid_concept_dict = mock_concepts[0].model_dump()
    valid_concept_dict["wikibase_revision"] = 12345
    valid_concept_dict["wikibase_url"] = "https://test.wikibase.org/wiki/Item:Q787"
    valid_concept = Concept.model_validate(valid_concept_dict)

    concepts = [
        valid_concept,
        mock_concepts[1],  # Missing Vespa fields - will fail validation
        valid_concept,
    ]

    mock_pool = AsyncMock()
    mock_pool.update_data = AsyncMock(return_value=mock_vespa_response_success)
    mock_pool.app.get_document_v1_path = AsyncMock(return_value="/document/v1/...")

    results = await update_concepts_in_vespa(concepts, mock_pool)

    assert len(results) == 3
    assert is_ok(results[0])
    assert is_err(results[1])
    assert is_ok(results[2])

    # Verify mock was called only for valid concepts (not for the invalid one)
    assert mock_pool.update_data.call_count == 2


@pytest.mark.asyncio
async def test_update_concepts_in_vespa__empty_list():
    """Test update_concepts_in_vespa with empty concept list."""
    mock_pool = AsyncMock()

    results = await update_concepts_in_vespa([], mock_pool)

    assert results == []
    mock_pool.update_data.assert_not_called()


@pytest.mark.vespa
@pytest.mark.asyncio
async def test_update_concepts_in_vespa__real_vespa(
    mock_concepts, local_vespa_search_adapter
):
    """Test update_concepts_in_vespa subflow with real Vespa."""
    # Add required Vespa fields to all concepts
    concepts_with_vespa = []
    for concept in mock_concepts:
        concept_dict = concept.model_dump()
        concept_dict["wikibase_revision"] = 12345
        concept_dict["wikibase_url"] = "https://test.wikibase.org/wiki/Item:Q787"
        concepts_with_vespa.append(Concept.model_validate(concept_dict))

    async with local_vespa_search_adapter.client.asyncio(
        connections=1
    ) as vespa_connection_pool:
        results = await update_concepts_in_vespa(
            concepts_with_vespa, vespa_connection_pool
        )

        assert len(results) == len(mock_concepts)
        assert all(is_ok(r) for r in results)


# Slack notification tests


@pytest.mark.asyncio
async def test_send_concept_validation_alert__validation_errors_only():
    """Test Slack alert with only validation errors."""
    # Create a validation error
    try:
        Concept(preferred_label="")
    except PydanticValidationError as e:
        validation_error = e

    failures = [
        Error(
            msg="validation error",
            metadata={
                "concept": {"wikibase_id": "Q123", "preferred_label": ""},
                "validations": validation_error,
            },
        )
    ]

    mock_slack_client = AsyncMock()
    mock_slack_client.chat_postMessage = AsyncMock(
        return_value={"ok": True, "ts": "1234567890.123456"}
    )

    # Mock the Prefect run context
    mock_flow_run = AsyncMock()
    mock_flow_run.name = "test-flow-run"
    mock_context = AsyncMock()
    mock_context.flow_run = mock_flow_run

    with (
        patch("flows.sync_concepts.get_slack_client", return_value=mock_slack_client),
        patch("flows.sync_concepts.get_run_context", return_value=mock_context),
    ):
        await send_concept_validation_alert(
            failures=failures,
            total_concepts=10,
            aws_env=AwsEnv.staging,
        )

    # Verify main message was posted
    assert mock_slack_client.chat_postMessage.call_count == 2  # Main + 1 thread
    main_call = mock_slack_client.chat_postMessage.call_args_list[0]

    # Check main message structure
    assert "attachments" in main_call.kwargs
    assert main_call.kwargs["attachments"][0]["color"] in ["#e01e5a", "#ecb22e"]

    # Check thread message for validation errors
    thread_call = mock_slack_client.chat_postMessage.call_args_list[1]
    assert "blocks" in thread_call.kwargs
    blocks = thread_call.kwargs["blocks"]

    # Should have section + table
    assert len(blocks) == 2
    assert blocks[0]["type"] == "section"
    assert "Data Quality Issues" in blocks[0]["text"]["text"]
    assert blocks[1]["type"] == "table"


@pytest.mark.asyncio
async def test_send_concept_validation_alert__system_errors_only():
    """Test Slack alert with only system errors."""
    failures = [
        Error(
            msg="Vespa update failed",
            metadata={
                "concept": {"wikibase_id": "Q456"},
            },
        ),
        Error(
            msg="Connection timeout",
            metadata={
                "concept": {"wikibase_id": "Q789"},
            },
        ),
    ]

    mock_slack_client = AsyncMock()
    mock_slack_client.chat_postMessage = AsyncMock(
        return_value={"ok": True, "ts": "1234567890.123456"}
    )

    # Mock the Prefect run context
    mock_flow_run = AsyncMock()
    mock_flow_run.name = "test-flow-run"
    mock_context = AsyncMock()
    mock_context.flow_run = mock_flow_run

    with (
        patch("flows.sync_concepts.get_slack_client", return_value=mock_slack_client),
        patch("flows.sync_concepts.get_run_context", return_value=mock_context),
    ):
        await send_concept_validation_alert(
            failures=failures,
            total_concepts=10,
            aws_env=AwsEnv.staging,
        )

    # Verify main message + system errors thread
    assert mock_slack_client.chat_postMessage.call_count == 2  # Main + 1 thread

    # Check thread message for system errors
    thread_call = mock_slack_client.chat_postMessage.call_args_list[1]
    blocks = thread_call.kwargs["blocks"]

    assert blocks[0]["type"] == "section"
    assert "System Errors" in blocks[0]["text"]["text"]
    assert blocks[1]["type"] == "table"

    # Verify table has header + 2 data rows
    table_rows = blocks[1]["rows"]
    assert len(table_rows) == 3  # Header + 2 errors


@pytest.mark.asyncio
async def test_send_concept_validation_alert__mixed_errors():
    """Test Slack alert with both validation and system errors."""
    # Create a validation error
    try:
        Concept(preferred_label="")
    except PydanticValidationError as e:
        validation_error = e

    failures = [
        Error(
            msg="validation error",
            metadata={
                "concept": {"wikibase_id": "Q111"},
                "validations": validation_error,
            },
        ),
        Error(
            msg="Vespa update failed",
            metadata={
                "concept": {"wikibase_id": "Q222"},
            },
        ),
    ]

    mock_slack_client = AsyncMock()
    mock_slack_client.chat_postMessage = AsyncMock(
        return_value={"ok": True, "ts": "1234567890.123456"}
    )

    # Mock the Prefect run context
    mock_flow_run = AsyncMock()
    mock_flow_run.name = "test-flow-run"
    mock_context = AsyncMock()
    mock_context.flow_run = mock_flow_run

    with (
        patch("flows.sync_concepts.get_slack_client", return_value=mock_slack_client),
        patch("flows.sync_concepts.get_run_context", return_value=mock_context),
    ):
        await send_concept_validation_alert(
            failures=failures,
            total_concepts=10,
            aws_env=AwsEnv.staging,
        )

    # Verify main message + 2 threads (validation + system)
    assert mock_slack_client.chat_postMessage.call_count == 4


@pytest.mark.asyncio
async def test_send_concept_validation_alert__colour_red_high_failure_rate():
    """Test that red colour is used when failure rate >= 50%."""
    failures = [
        Error(msg="error", metadata={"concept": {"wikibase_id": f"Q{i}"}})
        for i in range(5)
    ]

    mock_slack_client = AsyncMock()
    mock_slack_client.chat_postMessage = AsyncMock(
        return_value={"ok": True, "ts": "1234567890.123456"}
    )

    # Mock the Prefect run context
    mock_flow_run = AsyncMock()
    mock_flow_run.name = "test-flow-run"
    mock_context = AsyncMock()
    mock_context.flow_run = mock_flow_run

    with (
        patch("flows.sync_concepts.get_slack_client", return_value=mock_slack_client),
        patch("flows.sync_concepts.get_run_context", return_value=mock_context),
    ):
        await send_concept_validation_alert(
            failures=failures,
            total_concepts=10,  # 50% failure rate
            aws_env=AwsEnv.staging,
        )

    main_call = mock_slack_client.chat_postMessage.call_args_list[0]
    colour = main_call.kwargs["attachments"][0]["color"]
    assert colour == "#e01e5a"  # Red


@pytest.mark.asyncio
async def test_send_concept_validation_alert__colour_orange_low_failure_rate():
    """Test that orange colour is used when failure rate < 50%."""
    failures = [Error(msg="error", metadata={"concept": {"wikibase_id": "Q1"}})]

    mock_slack_client = AsyncMock()
    mock_slack_client.chat_postMessage = AsyncMock(
        return_value={"ok": True, "ts": "1234567890.123456"}
    )

    # Mock the Prefect run context
    mock_flow_run = AsyncMock()
    mock_flow_run.name = "test-flow-run"
    mock_context = AsyncMock()
    mock_context.flow_run = mock_flow_run

    with (
        patch("flows.sync_concepts.get_slack_client", return_value=mock_slack_client),
        patch("flows.sync_concepts.get_run_context", return_value=mock_context),
    ):
        await send_concept_validation_alert(
            failures=failures,
            total_concepts=10,  # 10% failure rate
            aws_env=AwsEnv.staging,
        )

    main_call = mock_slack_client.chat_postMessage.call_args_list[0]
    colour = main_call.kwargs["attachments"][0]["color"]
    assert colour == "#ecb22e"  # Orange


@pytest.mark.asyncio
async def test_send_concept_validation_alert__table_format():
    """Test that thread messages use proper table format with concept IDs."""
    failures = [
        Error(
            msg="Vespa update failed",
            metadata={"concept": {"wikibase_id": "Q100"}},
        ),
        Error(
            msg="Connection timeout",
            metadata={"concept": {"wikibase_id": "Q200"}},
        ),
    ]

    mock_slack_client = AsyncMock()
    mock_slack_client.chat_postMessage = AsyncMock(
        return_value={"ok": True, "ts": "1234567890.123456"}
    )

    # Mock the Prefect run context
    mock_flow_run = AsyncMock()
    mock_flow_run.name = "test-flow-run"
    mock_context = AsyncMock()
    mock_context.flow_run = mock_flow_run

    with (
        patch("flows.sync_concepts.get_slack_client", return_value=mock_slack_client),
        patch("flows.sync_concepts.get_run_context", return_value=mock_context),
    ):
        await send_concept_validation_alert(
            failures=failures,
            total_concepts=10,
            aws_env=AwsEnv.staging,
        )

    thread_call = mock_slack_client.chat_postMessage.call_args_list[1]
    table = thread_call.kwargs["blocks"][1]

    assert table["type"] == "table"
    rows = table["rows"]

    # Check header row
    header_row = rows[0]
    assert header_row[0]["elements"][0]["elements"][0]["text"] == "Concept ID"
    assert header_row[1]["elements"][0]["elements"][0]["text"] == "Error"

    # Check data rows contain wikibase IDs
    assert rows[1][0]["elements"][0]["elements"][0]["text"] == "Q100"
    assert rows[2][0]["elements"][0]["elements"][0]["text"] == "Q200"


@pytest.mark.asyncio
async def test_send_concept_validation_alert__slack_api_error():
    """Test that Slack API errors are handled gracefully."""
    failures = [Error(msg="error", metadata={"concept": {"wikibase_id": "Q1"}})]

    mock_slack_client = AsyncMock()
    mock_slack_client.chat_postMessage = AsyncMock(
        return_value={"ok": False, "error": "channel_not_found"}
    )

    # Mock the Prefect run context
    mock_flow_run = AsyncMock()
    mock_flow_run.name = "test-flow-run"
    mock_context = AsyncMock()
    mock_context.flow_run = mock_flow_run

    with (
        patch("flows.sync_concepts.get_slack_client", return_value=mock_slack_client),
        patch("flows.sync_concepts.get_run_context", return_value=mock_context),
    ):
        # Should not raise - errors are caught and logged
        await send_concept_validation_alert(
            failures=failures,
            total_concepts=10,
            aws_env=AwsEnv.staging,
        )

    # Verify that chat_postMessage was called
    assert mock_slack_client.chat_postMessage.call_count == 1
