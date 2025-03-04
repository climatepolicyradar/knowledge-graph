import pytest

from src.concept import Concept
from src.identifiers import WikibaseID


def test_wikibase____init__(MockedWikibaseSession, monkeypatch, mock_wikibase_url):
    # Login behaviour with env variables
    MockedWikibaseSession()

    # And without env variables
    monkeypatch.delenv("WIKIBASE_USERNAME")
    monkeypatch.delenv("WIKIBASE_PASSWORD")
    monkeypatch.delenv("WIKIBASE_URL")

    with pytest.raises(ValueError, match="must be set"):
        MockedWikibaseSession()

    MockedWikibaseSession(
        username="username", password="password", url=mock_wikibase_url
    )


def test_wikibase__get_concept(MockedWikibaseSession):
    wikibase = MockedWikibaseSession()
    concept = wikibase.get_concept(wikibase_id=WikibaseID("Q10"))
    assert isinstance(concept, Concept)
    assert concept.wikibase_id == "Q10"


def test_wikibase__get_all_concept_ids(MockedWikibaseSession):
    wikibase = MockedWikibaseSession()
    ids = wikibase.get_all_concept_ids()
    assert set(ids) == {
        "Q1000",
        "Q1003",
        "Q1007",
        "Q1001",
        "Q100",
        "Q1002",
        "Q1004",
        "Q10",
        "Q1006",
        "Q1005",
    }


def test_wikibase__get_concepts(MockedWikibaseSession):
    wikibase = MockedWikibaseSession()
    result = wikibase.get_concepts()
    ids = set([r.wikibase_id for r in result])
    assert ids == {
        "Q1000",
        "Q1003",
        "Q1007",
        "Q1001",
        "Q100",
        "Q1002",
        "Q1004",
        "Q10",
        "Q1006",
        "Q1005",
    }


@pytest.mark.skip(reason="Not implemented")
def test_wikibase__get_statements(MockedWikibaseSession):
    raise NotImplementedError(
        "The test_wikibase__get_statements test is not implemented yet."
    )


@pytest.mark.skip(reason="Not implemented")
def test_wikibase__add_statement(MockedWikibaseSession):
    raise NotImplementedError(
        "The test_wikibase__add_statement test is not implemented yet."
    )
