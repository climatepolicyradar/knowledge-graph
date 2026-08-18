import asyncio
import logging
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import ClassifierID, StatementRank, WikibaseID


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
    assert concept.wikibase_revision == 12345
    assert concept.classifier_ids == [
        (StatementRank.PREFERRED, ClassifierID("abcd2345"))
    ]


def _classifier_id_claim(rank: str, value: str) -> dict:
    """Build a Wikibase P20 (classifier ID) statement for tests."""
    return {
        "rank": rank,
        "mainsnak": {
            "snaktype": "value",
            "property": "P20",
            "datavalue": {"value": value, "type": "string"},
        },
    }


def test_wikibase__parse_classifier_id_claims(MockedWikibaseSession):
    wikibase = MockedWikibaseSession()

    valid_claims = [
        _classifier_id_claim("preferred", "abcd2345"),
        _classifier_id_claim("deprecated", "wxyz6789"),
    ]
    assert wikibase._parse_classifier_id_claims(WikibaseID("Q10"), valid_claims) == [
        (StatementRank.PREFERRED, ClassifierID("abcd2345")),
        (StatementRank.DEPRECATED, ClassifierID("wxyz6789")),
    ]

    # An empty list of claims yields an empty result
    assert wikibase._parse_classifier_id_claims(WikibaseID("Q10"), []) == []


def test_wikibase__parse_classifier_id_claims__malformed_raises_when_strict(
    MockedWikibaseSession,
):
    wikibase = MockedWikibaseSession()
    bad_claims = [
        _classifier_id_claim("preferred", "not-a-valid-id"),
        _classifier_id_claim("normal", "abcd2345"),
    ]
    with pytest.raises(ValueError, match="Validation error"):
        wikibase._parse_classifier_id_claims(
            WikibaseID("Q10"), bad_claims, raise_on_error=True
        )


def test_wikibase__parse_classifier_id_claims__malformed_skipped_when_not_strict(
    MockedWikibaseSession,
):
    wikibase = MockedWikibaseSession()
    bad_claims = [
        _classifier_id_claim("preferred", "not-a-valid-id"),
        _classifier_id_claim("normal", "abcd2345"),
    ]
    assert wikibase._parse_classifier_id_claims(
        WikibaseID("Q10"), bad_claims, raise_on_error=False
    ) == [(StatementRank.NORMAL, ClassifierID("abcd2345"))]


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


@pytest.mark.asyncio
async def test_get_concepts_async__logs_warning_on_retry(MockedWikibaseSession, caplog):
    """`before_sleep_log` should emit a warning each time `get_concepts_async retries`."""
    wikibase = MockedWikibaseSession()

    with (
        patch.object(
            wikibase,
            "get_all_concept_ids_async",
            side_effect=httpx.RequestError("Connection failed"),
        ),
        patch("asyncio.sleep"),  # Don't actually wait between retries
        caplog.at_level(logging.WARNING),  # Don't get too much noise
    ):
        with pytest.raises(Exception):
            await wikibase.get_concepts_async()

    assert any("Retrying" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_get_concepts_async__terminates_when_negative_concepts_form_a_cycle(
    MockedWikibaseSession,
):
    """
    Concepts which list each other as negative concepts must not recurse forever.

    Real example: "tax" (Q715) has "tax advantage" (Q1269) as a negative concept, and
    "tax advantage" has "tax" as a negative concept.
    """
    wikibase = MockedWikibaseSession()

    cyclic_concepts = {
        WikibaseID("Q10"): Concept(
            wikibase_id=WikibaseID("Q10"),
            preferred_label="tax",
            negative_concepts=[WikibaseID("Q20")],
        ),
        WikibaseID("Q20"): Concept(
            wikibase_id=WikibaseID("Q20"),
            preferred_label="tax advantage",
            negative_concepts=[WikibaseID("Q10")],
        ),
    }

    with patch.object(
        type(wikibase),
        "_parse_wikibase_entity",
        lambda self, wikibase_id, entity, revision_id: cyclic_concepts[
            wikibase_id
        ].model_copy(deep=True),
    ):
        concepts = await asyncio.wait_for(
            wikibase.get_concepts_async(wikibase_ids=[WikibaseID("Q10")]),
            timeout=30,
        )

    assert [concept.wikibase_id for concept in concepts] == [WikibaseID("Q10")]
    # The negative concept's positive labels are still merged in, one level deep
    assert concepts[0].negative_labels == ["tax advantage"]


def test_get_concept_ids_with_property__invalid_property_raises(MockedWikibaseSession):
    wikibase = MockedWikibaseSession()
    with pytest.raises(ValueError, match="Invalid property ID"):
        wikibase.get_concept_ids_with_property("not-a-property")


@pytest.mark.asyncio
async def test_get_concept_ids_with_property__parses_dedupes_and_sorts(
    MockedWikibaseSession,
):
    wikibase = MockedWikibaseSession()
    entity_prefix = wikibase.sparql_entity_prefix
    sparql_json = {
        "head": {"vars": ["entity"]},
        "results": {
            "bindings": [
                {"entity": {"type": "uri", "value": f"{entity_prefix}Q58"}},
                {"entity": {"type": "uri", "value": f"{entity_prefix}Q10"}},
                # A duplicate, which should be collapsed
                {"entity": {"type": "uri", "value": f"{entity_prefix}Q10"}},
            ]
        },
    }

    sparql_response = httpx.Response(
        200,
        json=sparql_json,
        request=httpx.Request("GET", wikibase.sparql_url),
    )
    client = await wikibase._get_client()
    with patch.object(
        client, "get", new=AsyncMock(return_value=sparql_response)
    ) as mock_get:
        ids = await wikibase.get_concept_ids_with_property_async("P20")

    # Deduplicated and sorted
    assert ids == [WikibaseID("Q10"), WikibaseID("Q58")]
    # The query targeted the SPARQL endpoint and filtered on the requested property
    assert mock_get.call_args.kwargs["url"] == wikibase.sparql_url
    assert "P20" in mock_get.call_args.kwargs["params"]["query"]


@pytest.mark.skip(reason="Not implemented")
def test_wikibase__get_statements(MockedWikibaseSession):
    raise NotImplementedError


def test_wikibase__create_concept(MockedWikibaseSession):
    wikibase = MockedWikibaseSession()
    concept = Concept(
        preferred_label="Test concept",
        description="A test description",
        definition="A longer definition of the concept.",
        alternative_labels=["alias one", "alias two"],
    )
    new_id = wikibase.create_concept(
        concept,
        subconcept_of=[WikibaseID("Q1")],
        has_subconcept=[WikibaseID("Q2")],
        related_to=[WikibaseID("Q3")],
        wikidata_id="Q12345",
    )
    assert isinstance(new_id, WikibaseID)


def test_wikibase__create_concept__minimal(MockedWikibaseSession):
    wikibase = MockedWikibaseSession()
    new_id = wikibase.create_concept(Concept(preferred_label="Minimal concept"))
    assert isinstance(new_id, WikibaseID)


def _entity_with_aliases(aliases: dict[str, list[str]]) -> dict:
    """Build a minimal Wikibase entity whose aliases span the given language tags."""
    return {
        "labels": {"en": {"language": "en", "value": "license"}},
        "aliases": {
            language: [{"language": language, "value": value} for value in values]
            for language, values in aliases.items()
        },
    }


def test_wikibase__parse_entity_reads_all_english_alias_languages(
    MockedWikibaseSession,
):
    """British spellings are tagged en-gb in Wikibase, so they must be read too."""
    wikibase = MockedWikibaseSession()

    concept = wikibase._parse_wikibase_entity(
        WikibaseID("Q1432"),
        _entity_with_aliases(
            {
                "en": ["licenses", "licensing"],
                "en-gb": ["licence", "licencing"],
                "en-us": ["license fee"],
                "mul": ["CO2"],
            }
        ),
    )

    assert concept.preferred_label == "license"
    assert set(concept.alternative_labels) == {
        "licenses",
        "licensing",
        "licence",
        "licencing",
        "license fee",
        "CO2",
    }


def test_wikibase__parse_entity_ignores_non_english_alias_languages(
    MockedWikibaseSession,
):
    wikibase = MockedWikibaseSession()

    concept = wikibase._parse_wikibase_entity(
        WikibaseID("Q1432"),
        _entity_with_aliases({"en": ["licenses"], "fr": ["licence d'exploitation"]}),
    )

    assert concept.alternative_labels == ["licenses"]


def test_wikibase__parse_entity_without_aliases(MockedWikibaseSession):
    wikibase = MockedWikibaseSession()

    concept = wikibase._parse_wikibase_entity(
        WikibaseID("Q1432"),
        {"labels": {"en": {"language": "en", "value": "license"}}},
    )

    assert concept.alternative_labels == []


def test_wikibase__add_claim(MockedWikibaseSession):
    wikibase = MockedWikibaseSession()
    # Should not raise
    wikibase.add_claim(
        entity_id=WikibaseID("Q10"),
        property_id="P1",
        target_id=WikibaseID("Q20"),
    )


def test_whether_wikibase_filters_out_negative_labels_from_subconcepts_which_overlap_with_positive_labels(
    MockedWikibaseSession,
):
    """
    We want to remove negative labels which overlap with positives from subconcepts

    Test that when include_labels_from_subconcepts=True, overlapping labels between
    positive and negative labels are filtered out from negative_labels.

    This tests the edge case described in the code where a concept can have subconcepts
    whose positive labels overlap with another subconcept's negative labels.
    """
    # Create a root concept with some subconcepts
    root_concept = Concept(
        preferred_label="fossil fuels",
        alternative_labels=["fossil energy"],
        wikibase_id=WikibaseID("Q123"),
        has_subconcept=[WikibaseID("Q456"), WikibaseID("Q789")],  # oil and coal
    )

    # Create subconcept A (oil) with positive labels, and negative labels that will
    # overlap with the other subconcept's positives
    subconcept_a = Concept(
        preferred_label="oil",
        alternative_labels=["petroleum", "crude oil"],
        negative_labels=["overlapping label"],
        wikibase_id=WikibaseID("Q456"),
    )

    # Create subconcept B (coal) with positive labels that will overlap with negatives
    subconcept_b = Concept(
        preferred_label="coal",
        alternative_labels=["bituminous coal", "overlapping label"],
        wikibase_id=WikibaseID("Q789"),
    )

    # Mock the WikibaseSession methods to return our test data
    with (
        patch.object(MockedWikibaseSession, "get_concepts_async") as mock_get_concepts,
        patch.object(
            MockedWikibaseSession, "get_recursive_has_subconcept_relationships_async"
        ) as mock_get_recursive,
    ):
        # Configure the mocks to handle the two calls to get_concepts_async:
        # 1. First call gets the root concept
        # 2. Second call gets the subconcepts
        def mock_get_concepts_side_effect(wikibase_ids=None, **_kwargs):
            if wikibase_ids == [WikibaseID("Q123")]:
                # First call - return root concept
                return [root_concept]
            elif wikibase_ids == [WikibaseID("Q456"), WikibaseID("Q789")]:
                # Second call - return subconcepts
                return [subconcept_a, subconcept_b]
            else:
                return []

        mock_get_concepts.side_effect = mock_get_concepts_side_effect
        mock_get_recursive.return_value = [WikibaseID("Q456"), WikibaseID("Q789")]

        wikibase = MockedWikibaseSession()

        # Call get_concept with include_labels_from_subconcepts=True
        result = wikibase.get_concept(
            wikibase_id=WikibaseID("Q123"), include_labels_from_subconcepts=True
        )

        # Verify that overlapping labels were removed from negative_labels
        # The original negative_labels were: ["overlapping label"]
        # This should be removed because it appears in subconcept_a's positive labels
        expected_negative_labels = []  # All original negatives overlap with positives
        assert result.negative_labels == expected_negative_labels

        # Verify that positive labels from subconcepts were added
        expected_positive_labels = {
            "fossil fuels",  # from root concept
            "fossil energy",  # from root concept
            "oil",  # from subconcept_a
            "petroleum",  # from subconcept_a
            "crude oil",  # from subconcept_a
            "coal",  # from subconcept_b
            "bituminous coal",  # from subconcept_b
            "overlapping label",  # from subconcept_b, despite being a negative label on subconcept_b
        }
        assert set(result.alternative_labels) == expected_positive_labels


def test_whether_wikibase_filters_out_only_overlapping_negative_labels_from_subconcepts(
    MockedWikibaseSession,
):
    """
    We want to remove ONLY negative labels which overlap with positives from subconcepts

    Test that when include_labels_from_subconcepts=True, only overlapping labels between
    positive and negative labels are filtered out, while non-overlapping negative labels
    are preserved.
    """
    # Create a root concept with mixed negative labels (some overlap, some don't)
    root_concept = Concept(
        preferred_label="fossil fuels",
        alternative_labels=["fossil energy"],
        negative_labels=[
            "overlapping label one",  # This will overlap with subconcept_a's positive labels
            "overlapping label two",  # This will overlap with subconcept_b's positive labels
            "renewable energy",  # This won't overlap, should be preserved as a negative
            "solar power",  # This won't overlap, should be preserved as a negative
        ],
        wikibase_id=WikibaseID("Q123"),
        has_subconcept=[WikibaseID("Q456"), WikibaseID("Q789")],  # oil and coal
    )

    # Create subconcept A (oil) with positive labels that will overlap with some negatives
    # from the root concept
    subconcept_a = Concept(
        preferred_label="oil",
        alternative_labels=["petroleum", "crude oil", "overlapping label one"],
        wikibase_id=WikibaseID("Q456"),
    )

    # Create subconcept B (coal) with positive labels that will overlap with some negatives
    # from the root concept
    subconcept_b = Concept(
        preferred_label="coal",
        alternative_labels=["bituminous coal", "overlapping label two"],
        wikibase_id=WikibaseID("Q789"),
    )

    # Mock the WikibaseSession methods to return our test data
    with (
        patch.object(MockedWikibaseSession, "get_concepts_async") as mock_get_concepts,
        patch.object(
            MockedWikibaseSession, "get_recursive_has_subconcept_relationships_async"
        ) as mock_get_recursive,
    ):
        # Configure the mocks to handle the two calls to get_concepts_async:
        # 1. First call gets the root concept
        # 2. Second call gets the subconcepts
        def mock_get_concepts_side_effect(wikibase_ids=None, **_kwargs):
            if wikibase_ids == [WikibaseID("Q123")]:
                # First call - return root concept
                return [root_concept]
            elif wikibase_ids == [WikibaseID("Q456"), WikibaseID("Q789")]:
                # Second call - return subconcepts
                return [subconcept_a, subconcept_b]
            else:
                return []

        mock_get_concepts.side_effect = mock_get_concepts_side_effect
        mock_get_recursive.return_value = [WikibaseID("Q456"), WikibaseID("Q789")]

        wikibase = MockedWikibaseSession()

        # Call get_concept with include_labels_from_subconcepts=True
        result = wikibase.get_concept(
            wikibase_id=WikibaseID("Q123"), include_labels_from_subconcepts=True
        )

        # Verify that only overlapping labels were removed from negative_labels
        # The original negative_labels were: ["overlapping label one",
        # "overlapping label two", "renewable energy", "solar power"]
        # "overlapping label one" and "overlapping label two" should be removed from the
        # concept's negative_labels because they appear in the subconcepts' positive
        # labels
        # "renewable energy" and "solar power" should be preserved because they don't
        # overlap
        expected_negative_labels = ["renewable energy", "solar power"]
        assert set(result.negative_labels) == set(expected_negative_labels)

        # Verify that positive labels from subconcepts were added
        expected_positive_labels = {
            "fossil fuels",  # from root concept
            "fossil energy",  # from root concept
            "oil",  # from subconcept_a
            "petroleum",  # from subconcept_a
            "crude oil",  # from subconcept_a
            "overlapping label one",  # from subconcept_a
            "overlapping label two",  # from subconcept_b
            "coal",  # from subconcept_b
            "bituminous coal",  # from subconcept_b
        }
        assert set(result.alternative_labels) == expected_positive_labels
