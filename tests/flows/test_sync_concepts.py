import polars as pl
import pytest

from flows.sync_concepts import (
    concepts_to_dataframe,
    load_concepts,
    sync_concepts,
)
from knowledge_graph.cloud import AwsEnv
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.wikibase import WikibaseAuth


@pytest.fixture
def wikibase_auth_fixture(mock_wikibase_url: str) -> WikibaseAuth:
    return WikibaseAuth(
        username="test_user",
        password="test_password",
        url=mock_wikibase_url,
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
    # classifier_ids field should be dropped during concepts_to_dataframe
    assert "classifier_ids" not in df.columns


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


def _write_cache(path, concepts):
    path.write_text("".join(c.model_dump_json() + "\n" for c in concepts))


@pytest.mark.asyncio
async def test_sync_concepts__writes_snapshot_and_latest(
    mock_concepts, tmp_path, wikibase_auth_fixture
):
    """Latest only holds live concepts, and the snapshot is kept alongside."""
    archive = tmp_path / "archive"
    snapshots = tmp_path / "archive_snapshots"
    archive.mkdir()
    snapshots.mkdir()
    deleted = Concept(wikibase_id=WikibaseID("Q999"), preferred_label="deleted")
    concepts_to_dataframe([mock_concepts[0], deleted]).write_parquet(
        archive / "latest.parquet"
    )
    concepts_to_dataframe([deleted]).write_parquet(snapshots / "concepts_old.parquet")

    cache_path = tmp_path / "cache.jsonl"
    _write_cache(cache_path, mock_concepts)

    await sync_concepts.fn(
        aws_env=AwsEnv.sandbox,
        wikibase_auth=wikibase_auth_fixture,
        wikibase_cache_path=cache_path,
        concepts_archive_path=archive,
    )

    expected_ids = sorted(str(c.wikibase_id) for c in mock_concepts)

    # Snowflake only sees latest, which has no deleted concepts
    assert [f.name for f in archive.glob("*.parquet")] == ["latest.parquet"]
    latest = pl.read_parquet(archive / "latest.parquet")
    assert sorted(latest["wikibase_id"].to_list()) == expected_ids

    # Old snapshots are never deleted, and the new one matches latest
    snapshot_files = {f.name: f for f in snapshots.glob("*.parquet")}
    assert "concepts_old.parquet" in snapshot_files
    assert len(snapshot_files) == 2
    new_snapshot = next(
        f for n, f in snapshot_files.items() if n != "concepts_old.parquet"
    )
    assert (
        sorted(pl.read_parquet(new_snapshot)["wikibase_id"].to_list()) == expected_ids
    )


@pytest.mark.asyncio
async def test_sync_concepts__creates_archive_if_missing(
    mock_concepts, tmp_path, wikibase_auth_fixture
):
    archive = tmp_path / "archive"
    cache_path = tmp_path / "cache.jsonl"
    _write_cache(cache_path, mock_concepts)

    await sync_concepts.fn(
        aws_env=AwsEnv.sandbox,
        wikibase_auth=wikibase_auth_fixture,
        wikibase_cache_path=cache_path,
        concepts_archive_path=archive,
    )

    assert len(pl.read_parquet(archive / "latest.parquet")) == len(mock_concepts)
    assert len(list((tmp_path / "archive_snapshots").glob("concepts_*.parquet"))) == 1


@pytest.mark.asyncio
async def test_sync_concepts__no_concepts_leaves_latest_untouched(
    tmp_path, mock_concepts, wikibase_auth_fixture
):
    archive = tmp_path / "archive"
    archive.mkdir()
    concepts_to_dataframe(mock_concepts).write_parquet(archive / "latest.parquet")

    cache_path = tmp_path / "cache.jsonl"
    cache_path.write_text("")

    with pytest.raises(ValueError, match="no concepts returned"):
        await sync_concepts.fn(
            aws_env=AwsEnv.sandbox,
            wikibase_auth=wikibase_auth_fixture,
            wikibase_cache_path=cache_path,
            concepts_archive_path=archive,
        )

    assert len(pl.read_parquet(archive / "latest.parquet")) == len(mock_concepts)
    assert not (tmp_path / "archive_snapshots").exists()
