from unittest.mock import AsyncMock, patch

import polars as pl
import pytest
from vespa.io import VespaResponse

from flows.result import Err, Error, Ok, is_err, is_ok, unwrap_err, unwrap_ok
from flows.wikibase_to_vespa import (
    concepts_to_dataframe,
    create_vespa_sync_summary_artifact,
    dataframe_to_concepts,
    get_new_versions,
    load_concepts,
    update_concept_in_vespa,
    update_concepts_in_vespa,
)
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

    new_versions = await get_new_versions(
        current_df, existing_ids, tmp_path / "archive"
    )

    assert len(new_versions) == expected_new


@pytest.mark.asyncio
async def test_get_new_versions__all_new(mock_concepts, tmp_path):
    """Test when all concepts are new (empty archive)."""
    archive_path = tmp_path / "archive"
    archive_path.mkdir()

    current_df = concepts_to_dataframe(mock_concepts).lazy()
    # Create a lazy frame from empty Parquet scan
    existing_ids = pl.scan_parquet(f"{archive_path}/*.parquet").select("id")

    new_versions = await get_new_versions(current_df, existing_ids, archive_path)

    assert len(new_versions) == len(mock_concepts)


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
        "flows.wikibase_to_vespa.acreate_table_artifact", new_callable=AsyncMock
    ) as mock_create:
        await create_vespa_sync_summary_artifact(
            results=results, parquet_path="/path/to/concepts_20250101_120000.parquet"
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
        "flows.wikibase_to_vespa.acreate_table_artifact", new_callable=AsyncMock
    ) as mock_create:
        await create_vespa_sync_summary_artifact(
            results=results, parquet_path="/path/to/file.parquet"
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
        "flows.wikibase_to_vespa.acreate_table_artifact", new_callable=AsyncMock
    ) as mock_create:
        await create_vespa_sync_summary_artifact(
            results=results, parquet_path="/path/to/file.parquet"
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
        "flows.wikibase_to_vespa.acreate_table_artifact", new_callable=AsyncMock
    ) as mock_create:
        await create_vespa_sync_summary_artifact(
            results=results, parquet_path="/path/to/file.parquet"
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
        "flows.wikibase_to_vespa.acreate_table_artifact", new_callable=AsyncMock
    ) as mock_create:
        await create_vespa_sync_summary_artifact(
            results=results, parquet_path="/path/to/file.parquet"
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
        "flows.wikibase_to_vespa.acreate_table_artifact", new_callable=AsyncMock
    ) as mock_create:
        await create_vespa_sync_summary_artifact(results=results, parquet_path=None)

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
