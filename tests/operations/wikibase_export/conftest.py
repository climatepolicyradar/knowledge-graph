from typing import AsyncGenerator

import aioboto3
import pytest
import pytest_asyncio
from types_aiobotocore_s3.client import S3Client

from knowledge_graph.operations.wikibase_export.snapshot import Snapshot
from tests.operations.wikibase_export.helpers import make_item


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
    name = "test-wikibase-export"
    await s3.create_bucket(
        Bucket=name, CreateBucketConfiguration={"LocationConstraint": "eu-west-1"}
    )
    yield name
    # The mock S3 server outlives each test
    response = await s3.list_objects_v2(Bucket=name)
    for obj in response.get("Contents", []):
        await s3.delete_object(Bucket=name, Key=obj["Key"])
    await s3.delete_bucket(Bucket=name)


@pytest.fixture
def snapshot() -> Snapshot:
    return Snapshot(
        timestamp="20261117T110000Z",
        wikibase_url="https://test.test.test",
        items=[make_item("Q1", label="a"), make_item("Q2", label="b")],
        properties=[{"type": "property", "id": "P1", "datatype": "wikibase-item"}],
        redirects={"Q15": "Q16"},
    )
