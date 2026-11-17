from contextlib import asynccontextmanager

from knowledge_graph.operations.wikibase_export.snapshot import Snapshot


def item_claim(property_id: str, *target_ids: str) -> dict:
    """Statements linking an item to other items, as Wikibase returns them."""
    return {
        property_id: [
            {
                "mainsnak": {
                    "snaktype": "value",
                    "property": property_id,
                    "datatype": "wikibase-item",
                    "datavalue": {
                        "value": {"entity-type": "item", "id": target_id},
                        "type": "wikibase-entityid",
                    },
                },
                "type": "statement",
                "rank": "normal",
            }
            for target_id in target_ids
        ]
    }


def make_item(
    entity_id: str,
    label: str | None = None,
    claims: dict | None = None,
) -> dict:
    return {
        "type": "item",
        "id": entity_id,
        "labels": {"en": {"language": "en", "value": label}} if label else {},
        "descriptions": {},
        "aliases": {},
        "claims": claims or {},
    }


def make_snapshot(*items: dict, redirects: dict[str, str] | None = None) -> Snapshot:
    return Snapshot(
        timestamp="20261117T110000Z",
        wikibase_url="https://test.test.test",
        items=list(items),
        properties=[],
        redirects=redirects or {},
    )


async def list_keys(s3, bucket: str, prefix: str) -> set[str]:
    response = await s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    return {obj["Key"] for obj in response.get("Contents", [])}


@asynccontextmanager
async def temporary_bucket(s3, name: str):
    await s3.create_bucket(
        Bucket=name, CreateBucketConfiguration={"LocationConstraint": "eu-west-1"}
    )
    yield name
    # The mock S3 server outlives each test
    response = await s3.list_objects_v2(Bucket=name)
    for obj in response.get("Contents", []):
        await s3.delete_object(Bucket=name, Key=obj["Key"])
    await s3.delete_bucket(Bucket=name)


async def get_text(s3, bucket: str, key: str) -> str:
    response = await s3.get_object(Bucket=bucket, Key=key)
    return (await response["Body"].read()).decode("utf-8")
