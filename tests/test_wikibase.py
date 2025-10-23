from unittest.mock import patch

import pytest

from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import WikibaseID


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
