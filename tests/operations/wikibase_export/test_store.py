import json

import pytest

from knowledge_graph.operations.wikibase_export.rules import Issue
from knowledge_graph.operations.wikibase_export.store import (
    PREFIX,
    advance_current,
    mark_reported,
    read_current,
    read_reported_issues,
    write_snapshot,
)
from tests.operations.wikibase_export.helpers import (
    get_json,
    get_text,
    list_keys,
    make_item,
    make_snapshot,
    put_json,
)

WARNING = Issue(rule="r", severity="warning", entity_id="Q1", message="m")
ERROR = Issue(rule="r", severity="error", entity_id="Q1", message="m")


@pytest.mark.asyncio
async def test_write_snapshot_layout(s3, bucket, snapshot):
    prefix = await write_snapshot(
        s3, bucket, snapshot, issues=[], failed_rules=[], flow_run_name="r"
    )

    assert prefix == "pipelines/wikibase-export/snapshots/20261117T110000Z"
    assert await list_keys(s3, bucket, prefix) == {
        f"{prefix}/{name}"
        for name in [
            "items.jsonl",
            "properties.jsonl",
            "redirects.jsonl",
            "issues.jsonl",
            "manifest.json",
        ]
    }


@pytest.mark.asyncio
async def test_write_snapshot_manifest(s3, bucket, snapshot):
    prefix = await write_snapshot(
        s3,
        bucket,
        snapshot,
        issues=[WARNING, ERROR],
        failed_rules=["broken"],
        flow_run_name="run-name",
    )

    assert await get_json(s3, bucket, f"{prefix}/manifest.json") == {
        "format_version": 1,
        "snapshot": "20261117T110000Z",
        "wikibase_url": "https://test.test.test",
        "counts": {"items": 2, "properties": 1, "redirects": 1},
        "issues": {
            "by_rule": {"r": 2},
            "by_severity": {"warning": 1, "error": 1},
        },
        "failed_rules": ["broken"],
        "flow_run_name": "run-name",
    }


@pytest.mark.asyncio
async def test_write_snapshot_redirects(s3, bucket, snapshot):
    prefix = await write_snapshot(s3, bucket, snapshot, [], [], None)

    body = await get_text(s3, bucket, f"{prefix}/redirects.jsonl")

    assert [json.loads(line) for line in body.splitlines()] == [
        {"from": "Q15", "to": "Q16"}
    ]


@pytest.mark.asyncio
async def test_items_round_trip_one_per_line_with_unicode_and_newlines(s3, bucket):
    item = make_item("Q1", label="Énergie\nrenouvelable ☀")
    prefix = await write_snapshot(s3, bucket, make_snapshot(item), [], [], None)

    body = await get_text(s3, bucket, f"{prefix}/items.jsonl")

    lines = body.splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == item
    assert "Énergie" in body
    assert body.endswith("\n")


@pytest.mark.asyncio
async def test_advance_current_first_run(s3, bucket, snapshot):
    assert await read_current(s3, bucket) is None

    assert await advance_current(s3, bucket, snapshot, issues=[WARNING]) is True

    current = await read_current(s3, bucket)
    assert current is not None
    assert current.snapshot == snapshot.timestamp
    assert current.pinned is False


@pytest.mark.asyncio
async def test_advance_current_respects_pin(s3, bucket, snapshot):
    await put_json(
        s3,
        bucket,
        f"{PREFIX}/current.json",
        {"snapshot": "20260101T000000Z", "pinned": True},
    )

    assert await advance_current(s3, bucket, snapshot, issues=[]) is False

    current = await read_current(s3, bucket)
    assert current is not None
    assert current.snapshot == "20260101T000000Z"
    assert current.pinned is True


@pytest.mark.asyncio
async def test_advance_current_held_back_by_error_issue(s3, bucket, snapshot):
    assert await advance_current(s3, bucket, snapshot, issues=[ERROR]) is False

    assert await read_current(s3, bucket) is None


@pytest.mark.asyncio
async def test_read_current_raises_on_invalid_pointer(s3, bucket):
    await s3.put_object(Bucket=bucket, Key=f"{PREFIX}/current.json", Body=b"not json")

    with pytest.raises(ValueError):
        await read_current(s3, bucket)


@pytest.mark.asyncio
async def test_reported_issues_round_trip(s3, bucket, snapshot):
    assert await read_reported_issues(s3, bucket) is None

    await write_snapshot(s3, bucket, snapshot, [WARNING], [], None)
    await mark_reported(s3, bucket, snapshot)

    assert await read_reported_issues(s3, bucket) == [WARNING]
