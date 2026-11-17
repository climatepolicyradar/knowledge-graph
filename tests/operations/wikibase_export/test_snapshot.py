from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from knowledge_graph.operations.wikibase_export.snapshot import (
    IncompleteCrawlError,
    crawl,
)


@pytest.mark.asyncio
async def test_crawl_returns_all_items_and_properties(MockedWikibaseSession):
    snapshot = await crawl(
        MockedWikibaseSession(),
        now=datetime(2026, 11, 17, 11, 0, tzinfo=timezone.utc),
    )

    assert snapshot.timestamp == "20261117T110000Z"
    assert snapshot.wikibase_url == "https://test.test.test"
    assert len(snapshot.items) == 10
    assert {e["type"] for e in snapshot.items} == {"item"}
    assert [e["id"] for e in snapshot.properties] == ["P1", "P10"]
    assert snapshot.item_ids() == {e["id"] for e in snapshot.items}


@pytest.mark.asyncio
async def test_crawl_keeps_entities_untouched(
    MockedWikibaseSession, mock_wikibase_raw_entity
):
    snapshot = await crawl(MockedWikibaseSession())

    assert snapshot.items[0] == mock_wikibase_raw_entity(snapshot.items[0]["id"])


@pytest.mark.asyncio
async def test_crawl_includes_redirects(MockedWikibaseSession):
    session = MockedWikibaseSession()
    await session._get_client()
    session._redirects = {"Q15": "Q16"}

    snapshot = await crawl(session)

    assert snapshot.redirects == {"Q15": "Q16"}


@pytest.mark.asyncio
async def test_crawl_raises_when_nothing_listed(MockedWikibaseSession, monkeypatch):
    session = MockedWikibaseSession()
    monkeypatch.setattr(
        session, "get_all_entity_ids_async", AsyncMock(return_value=([], []))
    )

    with pytest.raises(IncompleteCrawlError):
        await crawl(session)


@pytest.mark.asyncio
async def test_crawl_tolerates_a_few_entities_deleted_mid_crawl(
    MockedWikibaseSession, monkeypatch
):
    session = MockedWikibaseSession()
    ids = [f"Q{i}" for i in range(1000, 1099)] + ["Q999"]
    monkeypatch.setattr(
        session, "get_all_entity_ids_async", AsyncMock(return_value=(ids, ["P1"]))
    )

    snapshot = await crawl(session)

    assert len(snapshot.items) == 99


@pytest.mark.asyncio
async def test_crawl_raises_when_many_listed_entities_are_not_returned(
    MockedWikibaseSession, monkeypatch
):
    session = MockedWikibaseSession()
    monkeypatch.setattr(
        session,
        "get_all_entity_ids_async",
        AsyncMock(return_value=(["Q10", "Q11", "Q12"], [])),
    )
    monkeypatch.setattr(
        session,
        "get_raw_entities_async",
        AsyncMock(side_effect=[[{"id": "Q10", "type": "item"}], []]),
    )

    with pytest.raises(IncompleteCrawlError, match="2 of 3"):
        await crawl(session)
