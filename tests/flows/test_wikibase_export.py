from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from flows.wikibase_export import wikibase_export
from knowledge_graph.cloud import AwsEnv
from knowledge_graph.operations.wikibase_export.rules import Issue
from knowledge_graph.operations.wikibase_export.snapshot import IncompleteCrawlError
from knowledge_graph.operations.wikibase_export.store import PREFIX, read_current
from knowledge_graph.wikibase import WikibaseAuth
from tests.operations.wikibase_export.helpers import (
    get_json,
    list_keys,
    temporary_bucket,
)

BUCKET = "test-wikibase-export-flow"

pytestmark = pytest.mark.no_xdist


@pytest.fixture
def auth(mock_wikibase_url: str) -> WikibaseAuth:
    return WikibaseAuth(
        username="test_user", password="test_password", url=mock_wikibase_url
    )


@pytest_asyncio.fixture
async def bucket(mock_s3_async_client):
    async with temporary_bucket(mock_s3_async_client, BUCKET) as name:
        yield name


async def snapshot_timestamps(s3, bucket: str) -> set[str]:
    keys = await list_keys(s3, bucket, f"{PREFIX}/snapshots/")
    return {key.split("/")[3] for key in keys}


@pytest.mark.asyncio
async def test_flow_writes_snapshot_and_pointer(
    MockedWikibaseSession, mock_s3_async_client, bucket, auth
):
    await wikibase_export.fn(aws_env=AwsEnv.sandbox, wikibase_auth=auth, bucket=bucket)

    current = await read_current(mock_s3_async_client, bucket)
    assert current is not None
    manifest = await get_json(
        mock_s3_async_client,
        bucket,
        f"{PREFIX}/snapshots/{current.snapshot}/manifest.json",
    )
    assert manifest["counts"] == {"items": 10, "properties": 2, "redirects": 1}


@pytest.mark.asyncio
async def test_flow_requires_bucket_outside_staging_and_production(
    MockedWikibaseSession,
):
    with pytest.raises(ValueError, match="no default bucket for sandbox"):
        await wikibase_export.fn(aws_env=AwsEnv.sandbox)


@pytest.mark.asyncio
async def test_flow_writes_but_holds_pointer_back_when_a_rule_raises(
    MockedWikibaseSession, mock_s3_async_client, bucket, auth
):
    def broken_rule(snapshot):
        raise ValueError("boom")

    with patch("flows.wikibase_export.RULES", [broken_rule]):
        with pytest.raises(RuntimeError, match="broken_rule"):
            await wikibase_export.fn(
                aws_env=AwsEnv.sandbox, wikibase_auth=auth, bucket=bucket
            )

    assert await read_current(mock_s3_async_client, bucket) is None
    (timestamp,) = await snapshot_timestamps(mock_s3_async_client, bucket)
    manifest = await get_json(
        mock_s3_async_client, bucket, f"{PREFIX}/snapshots/{timestamp}/manifest.json"
    )
    assert manifest["failed_rules"] == ["broken_rule"]


@pytest.mark.asyncio
async def test_flow_writes_nothing_when_crawl_fails(
    MockedWikibaseSession, mock_s3_async_client, bucket, auth
):
    with patch(
        "flows.wikibase_export.crawl",
        AsyncMock(side_effect=IncompleteCrawlError("nothing listed")),
    ):
        with pytest.raises(IncompleteCrawlError):
            await wikibase_export.fn(
                aws_env=AwsEnv.sandbox, wikibase_auth=auth, bucket=bucket
            )

    assert await list_keys(mock_s3_async_client, bucket, PREFIX) == set()


@pytest.mark.asyncio
@pytest.mark.parametrize("bucket_name", [None, "cpr-cache", "cpr-staging-cache"])
async def test_flow_refuses_another_wikibase_in_a_default_bucket(
    MockedWikibaseSession, auth, bucket_name
):
    crawl = AsyncMock()

    with patch("flows.wikibase_export.crawl", crawl):
        with pytest.raises(ValueError, match="its own bucket"):
            await wikibase_export.fn(
                aws_env=AwsEnv.staging, wikibase_auth=auth, bucket=bucket_name
            )

    crawl.assert_not_called()


@pytest.mark.asyncio
async def test_flow_holds_pointer_back_on_an_error_issue(
    MockedWikibaseSession, mock_s3_async_client, bucket, auth
):
    def error_rule(snapshot) -> list[Issue]:
        return [Issue(severity="error", entity_id="Q10", message="m")]

    with patch("flows.wikibase_export.RULES", [error_rule]):
        await wikibase_export.fn(
            aws_env=AwsEnv.sandbox, wikibase_auth=auth, bucket=bucket
        )

    assert await read_current(mock_s3_async_client, bucket) is None
    (timestamp,) = await snapshot_timestamps(mock_s3_async_client, bucket)
    manifest = await get_json(
        mock_s3_async_client, bucket, f"{PREFIX}/snapshots/{timestamp}/manifest.json"
    )
    assert manifest["issues"]["by_severity"] == {"error": 1}
