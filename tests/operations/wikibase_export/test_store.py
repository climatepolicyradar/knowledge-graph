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
async def test_write_snapshot_leaves_out_revision_fields(s3, bucket, snapshot):
    revision = {"lastrevid": 3785, "modified": "2026-11-01T10:00:00Z"}
    with_revision = snapshot.model_copy(
        update={
            "items": [item | revision for item in snapshot.items],
            "properties": [prop | revision for prop in snapshot.properties],
        }
    )

    prefix = await write_snapshot(s3, bucket, with_revision, [], [], None)

    for name, expected in [
        ("items", snapshot.items),
        ("properties", snapshot.properties),
    ]:
        body = await get_text(s3, bucket, f"{prefix}/{name}.jsonl")
        assert [json.loads(line) for line in body.splitlines()] == expected


@pytest.mark.asyncio
async def test_advance_current_first_run(s3, bucket, snapshot):
    assert await read_current(s3, bucket) is None

    assert await advance_current(s3, bucket, snapshot, [WARNING], []) is True

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

    assert await advance_current(s3, bucket, snapshot, [], []) is False

    current = await read_current(s3, bucket)
    assert current is not None
    assert current.snapshot == "20260101T000000Z"
    assert current.pinned is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("issues", "failed_rules"), [([ERROR], []), ([WARNING], ["broken"])]
)
async def test_advance_current_held_back_keeps_previous_snapshot(
    s3, bucket, snapshot, issues, failed_rules
):
    await put_json(
        s3, bucket, f"{PREFIX}/current.json", {"snapshot": "20260101T000000Z"}
    )

    assert await advance_current(s3, bucket, snapshot, issues, failed_rules) is False

    current = await read_current(s3, bucket)
    assert current is not None and current.snapshot == "20260101T000000Z"


@pytest.mark.asyncio
async def test_reported_issues_round_trip(s3, bucket):
    assert await read_reported_issues(s3, bucket) is None

    await mark_reported(s3, bucket, [WARNING])

    assert await read_reported_issues(s3, bucket) == [WARNING]


@pytest.mark.asyncio
async def test_write_snapshot_writes_the_manifest_last(s3, bucket, snapshot):
    written = []
    original_put_object = s3.put_object

    async def recording_put_object(**kwargs):
        written.append(kwargs["Key"].rsplit("/", 1)[1])
        return await original_put_object(**kwargs)

    s3.put_object = recording_put_object

    await write_snapshot(s3, bucket, snapshot, [], [], None)

    assert len(written) == 5
    assert written[-1] == "manifest.json"
