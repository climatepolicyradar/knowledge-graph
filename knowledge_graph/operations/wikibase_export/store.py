import json
from collections import Counter
from collections.abc import Iterable

from botocore.exceptions import ClientError
from pydantic import BaseModel
from types_aiobotocore_s3.client import S3Client

from knowledge_graph.operations.wikibase_export.rules import Issue
from knowledge_graph.operations.wikibase_export.snapshot import Snapshot

PREFIX = "pipelines/wikibase-export"
REVISION_FIELDS = {"lastrevid", "modified"}

CURRENT_KEY = f"{PREFIX}/current.json"


class Pointer(BaseModel):
    """Names one snapshot. A pinned pointer is never moved by the export."""

    snapshot: str
    pinned: bool = False


def snapshot_prefix(timestamp: str) -> str:
    return f"{PREFIX}/snapshots/{timestamp}"


def _without_revision(entity: dict) -> dict:
    """The entity without its revision details, which the export leaves out."""
    return {key: value for key, value in entity.items() if key not in REVISION_FIELDS}


def _json_lines(rows: Iterable[dict]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode(
        "utf-8"
    )


async def _put(s3_client: S3Client, bucket: str, key: str, body: bytes) -> None:
    await s3_client.put_object(Bucket=bucket, Key=key, Body=body)


async def _get(s3_client: S3Client, bucket: str, key: str) -> str | None:
    """Read an object as text, or None if it doesn't exist."""
    try:
        response = await s3_client.get_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") == "NoSuchKey":
            return None
        raise
    return (await response["Body"].read()).decode("utf-8")


async def write_snapshot(
    s3_client: S3Client,
    bucket: str,
    snapshot: Snapshot,
    issues: list[Issue],
    failed_rules: list[str],
    flow_run_name: str | None,
) -> str:
    """Write one snapshot's files and return the key prefix they are under."""
    prefix = snapshot_prefix(snapshot.timestamp)

    await _put(
        s3_client,
        bucket,
        f"{prefix}/items.jsonl",
        _json_lines(map(_without_revision, snapshot.items)),
    )
    await _put(
        s3_client,
        bucket,
        f"{prefix}/properties.jsonl",
        _json_lines(map(_without_revision, snapshot.properties)),
    )
    await _put(
        s3_client,
        bucket,
        f"{prefix}/redirects.jsonl",
        _json_lines(
            {"from": source, "to": target}
            for source, target in snapshot.redirects.items()
        ),
    )
    await _put(
        s3_client,
        bucket,
        f"{prefix}/issues.jsonl",
        _json_lines(issue.model_dump() for issue in issues),
    )

    manifest = {
        "snapshot": snapshot.timestamp,
        "wikibase_url": snapshot.wikibase_url,
        "counts": {
            "items": len(snapshot.items),
            "properties": len(snapshot.properties),
            "redirects": len(snapshot.redirects),
        },
        "issues": {
            "by_rule": dict(Counter(issue.rule for issue in issues)),
            "by_severity": dict(Counter(issue.severity for issue in issues)),
        },
        "failed_rules": failed_rules,
        "flow_run_name": flow_run_name,
    }
    # Written last: a snapshot without a manifest is incomplete
    await _put(
        s3_client,
        bucket,
        f"{prefix}/manifest.json",
        json.dumps(manifest, ensure_ascii=False).encode("utf-8"),
    )

    return prefix


async def read_current(s3_client: S3Client, bucket: str) -> Pointer | None:
    """The snapshot downstream should read, or None before the first export."""
    body = await _get(s3_client, bucket, CURRENT_KEY)
    if body is None:
        return None
    return Pointer.model_validate_json(body)


async def advance_current(
    s3_client: S3Client,
    bucket: str,
    snapshot: Snapshot,
    issues: list[Issue],
    failed_rules: list[str],
) -> bool:
    """Point downstream at this snapshot, unless pinned, unchecked or erroring."""
    # A rule that raised may have missed an error
    if failed_rules or any(issue.severity == "error" for issue in issues):
        return False
    current = await read_current(s3_client, bucket)
    if current is not None and current.pinned:
        return False

    await _put(
        s3_client,
        bucket,
        CURRENT_KEY,
        Pointer(snapshot=snapshot.timestamp).model_dump_json().encode("utf-8"),
    )
    return True
