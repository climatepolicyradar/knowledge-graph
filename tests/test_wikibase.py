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


def test_wikibase__get_all_properties(MockedWikibaseSession):
    MockedWikibaseSession().get_all_properties()


def test_wikibase__get_concept(MockedWikibaseSession):
    wikibase = MockedWikibaseSession()
    concept = wikibase.get_concept(wikibase_id=WikibaseID("Q10"))
    assert isinstance(concept, Concept)
    assert concept.wikibase_id == "Q10"


def test_wikibase__get_concepts(MockedWikibaseSession):
    wikibase = MockedWikibaseSession()
    result = wikibase.get_concepts()
    ids = set([r.wikibase_id for r in result])
    assert ids == {"Q10", "Q1000", "Q1002", "Q100", "Q1001"}


def test_wikibase__get_subconcepts(MockedWikibaseSession):
    wikibase = MockedWikibaseSession()
    wikibase.get_subconcepts(wikibase_id="Q10")


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
