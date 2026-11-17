from datetime import datetime, timezone

from pydantic import BaseModel

from knowledge_graph.wikibase import WikibaseSession

TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"


class IncompleteCrawlError(Exception):
    """The crawl did not return a complete picture of Wikibase."""


class Snapshot(BaseModel):
    """Everything in Wikibase at one point in time, as Wikibase returned it."""

    timestamp: str
    wikibase_url: str
    items: list[dict]
    properties: list[dict]
    redirects: dict[str, str]

    def item_ids(self) -> set[str]:
        """The ID of every item in the snapshot."""
        return {item["id"] for item in self.items}


async def crawl(session: WikibaseSession, now: datetime | None = None) -> Snapshot:
    """Fetch every item and property from Wikibase, without parsing them."""
    now = now or datetime.now(timezone.utc)

    item_ids, property_ids = await session.get_all_entity_ids_async()
    if not item_ids:
        raise IncompleteCrawlError("Wikibase listed no items")

    return Snapshot(
        timestamp=now.astimezone(timezone.utc).strftime(TIMESTAMP_FORMAT),
        wikibase_url=str(session.base_url),
        items=await session.get_raw_entities_async(item_ids),
        properties=await session.get_raw_entities_async(property_ids),
        redirects=await session.get_redirects_async(),
    )
