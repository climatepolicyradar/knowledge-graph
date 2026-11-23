from contextlib import nullcontext
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from knowledge_graph.operations.wikibase_export.snapshot import (
    IncompleteCrawlError,
    crawl,
)


@pytest.mark.asyncio
async def test_crawl_returns_every_entity_and_redirect(MockedWikibaseSession):
    snapshot = await crawl(
        MockedWikibaseSession(),
        now=datetime(2026, 11, 17, 11, 0, tzinfo=timezone.utc),
    )

    assert snapshot.timestamp == "20261117T110000Z"
    assert snapshot.wikibase_url == "https://test.test.test"
    assert len(snapshot.items) == 10
    assert {e["type"] for e in snapshot.items} == {"item"}
    assert [e["id"] for e in snapshot.properties] == ["P1", "P10"]
    assert snapshot.redirects == {"Q15": "Q10"}


@pytest.mark.asyncio
async def test_crawl_keeps_entities_untouched(
    MockedWikibaseSession, mock_wikibase_raw_entity
):
    snapshot = await crawl(MockedWikibaseSession())

    assert snapshot.items[0] == mock_wikibase_raw_entity(snapshot.items[0]["id"])


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
    # The mocked session returns Q999 as deleted, so 99 of 100 come back
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
    # Only Q10 of the three listed items comes back
    monkeypatch.setattr(
        session,
        "get_raw_entities_async",
        AsyncMock(side_effect=[[{"id": "Q10", "type": "item"}], []]),
    )

    with pytest.raises(IncompleteCrawlError, match="2 of 3"):
        await crawl(session)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("returned", "outcome"),
    [
        pytest.param(297, nullcontext(), id="1% missing"),
        pytest.param(
            296, pytest.raises(IncompleteCrawlError, match="4 of 300"), id="over 1%"
        ),
    ],
)
async def test_crawl_allows_up_to_one_percent_not_returned(
    MockedWikibaseSession, monkeypatch, returned, outcome
):
    session = MockedWikibaseSession()
    ids = [f"Q{i}" for i in range(300)]
    monkeypatch.setattr(
        session, "get_all_entity_ids_async", AsyncMock(return_value=(ids, []))
    )
    monkeypatch.setattr(
        session,
        "get_raw_entities_async",
        AsyncMock(side_effect=[[{"id": i} for i in ids[:returned]], []]),
    )

    with outcome:
        await crawl(session)
