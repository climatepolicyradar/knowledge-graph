from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from flows.wikibase_export import wikibase_export
from knowledge_graph.cloud import AwsEnv
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
