import json

import pytest

from knowledge_graph.operations.wikibase_export.store import write_snapshot
from tests.operations.wikibase_export.helpers import (
    get_text,
    list_keys,
    make_item,
    make_snapshot,
)


@pytest.mark.asyncio
async def test_write_snapshot_layout(s3, bucket, snapshot):
    prefix = await write_snapshot(s3, bucket, snapshot, issues=[])

    assert prefix == "pipelines/wikibase-export/snapshots/20261117T110000Z"
    assert await list_keys(s3, bucket, prefix) == {
        f"{prefix}/{name}"
        for name in [
            "items.jsonl",
            "properties.jsonl",
            "redirects.jsonl",
            "issues.jsonl",
        ]
    }


@pytest.mark.asyncio
async def test_write_snapshot_redirects(s3, bucket, snapshot):
    prefix = await write_snapshot(s3, bucket, snapshot, [])

    body = await get_text(s3, bucket, f"{prefix}/redirects.jsonl")

    assert [json.loads(line) for line in body.splitlines()] == [
        {"from": "Q15", "to": "Q16"}
    ]


@pytest.mark.asyncio
async def test_items_round_trip_one_per_line_with_unicode_and_newlines(s3, bucket):
    item = make_item("Q1", label="Énergie\nrenouvelable ☀")
    prefix = await write_snapshot(s3, bucket, make_snapshot(item), [])

    body = await get_text(s3, bucket, f"{prefix}/items.jsonl")

    lines = body.splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == item
    assert "Énergie" in body
    assert body.endswith("\n")


@pytest.mark.asyncio
async def test_write_snapshot_leaves_out_revision_fields(s3, bucket, snapshot):
    revision = {"lastrevid": 3785, "modified": "2026-11-01T10:00:00Z"}
    with_revision = snapshot.model_copy(
        update={
            "items": [item | revision for item in snapshot.items],
            "properties": [prop | revision for prop in snapshot.properties],
        }
    )

    prefix = await write_snapshot(s3, bucket, with_revision, [])

    for name, expected in [
        ("items", snapshot.items),
        ("properties", snapshot.properties),
    ]:
        body = await get_text(s3, bucket, f"{prefix}/{name}.jsonl")
        assert [json.loads(line) for line in body.splitlines()] == expected
