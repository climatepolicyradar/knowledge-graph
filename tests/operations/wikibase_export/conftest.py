from typing import AsyncGenerator

import aioboto3
import pytest
import pytest_asyncio
from types_aiobotocore_s3.client import S3Client

from knowledge_graph.operations.wikibase_export.snapshot import Snapshot
from tests.operations.wikibase_export.helpers import make_item, temporary_bucket


@pytest_asyncio.fixture
async def s3(
    mock_aws_creds,
    # From pytest-aioboto3
    moto_patch_session,
) -> AsyncGenerator[S3Client, None]:
    session = aioboto3.Session(region_name="eu-west-1")
    async with session.client("s3") as client:
        yield client


@pytest_asyncio.fixture
async def bucket(s3) -> AsyncGenerator[str, None]:
    async with temporary_bucket(s3, "test-wikibase-export") as name:
        yield name


@pytest.fixture
def snapshot() -> Snapshot:
    return Snapshot(
        timestamp="20261117T110000Z",
        wikibase_url="https://test.test.test",
        items=[make_item("Q1", label="a"), make_item("Q2", label="b")],
        properties=[{"type": "property", "id": "P1", "datatype": "wikibase-item"}],
        redirects={"Q15": "Q16"},
    )
