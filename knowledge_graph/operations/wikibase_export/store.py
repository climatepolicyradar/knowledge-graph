import json
from collections.abc import Iterable

from types_aiobotocore_s3.client import S3Client

from knowledge_graph.operations.wikibase_export.rules import Issue
from knowledge_graph.operations.wikibase_export.snapshot import Snapshot

PREFIX = "pipelines/wikibase-export"
REVISION_FIELDS = {"lastrevid", "modified"}


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


async def write_snapshot(
    s3_client: S3Client,
    bucket: str,
    snapshot: Snapshot,
    issues: list[Issue],
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

    return prefix
