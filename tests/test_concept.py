import os

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from src.concept import Concept
from src.identifiers import WikibaseID

label_strategy = st.text(
    min_size=1,
    max_size=25,
    alphabet=st.characters(exclude_categories=("C", "Z")),
).filter(lambda x: x.strip())
wikibase_id_strategy = st.from_regex(r"^Q[1-9]\d*$", fullmatch=True)


@given(
    preferred_label=label_strategy,
    alternative_labels=st.lists(label_strategy, max_size=5),
    description=st.one_of(st.text(max_size=200), st.none()),
    wikibase_id=st.one_of(wikibase_id_strategy, st.none()),
    subconcept_of=st.lists(wikibase_id_strategy, max_size=3),
    has_subconcept=st.lists(wikibase_id_strategy, max_size=3),
    related_concepts=st.lists(wikibase_id_strategy, max_size=3),
    definition=st.one_of(st.text(max_size=200), st.none()),
)
def test_whether_concept_is_successfully_initialized(
    preferred_label,
    alternative_labels,
    description,
    wikibase_id,
    subconcept_of,
    has_subconcept,
    related_concepts,
    definition,
):
    concept = Concept(
        preferred_label=preferred_label,
        alternative_labels=alternative_labels,
        description=description,
        wikibase_id=wikibase_id,
        subconcept_of=subconcept_of,
        has_subconcept=has_subconcept,
        related_concepts=related_concepts,
        definition=definition,
    )

    assert concept.preferred_label == preferred_label
    assert set(concept.alternative_labels) == set(alternative_labels)
    assert concept.description == description
    assert concept.wikibase_id == wikibase_id
    assert set(concept.subconcept_of) == set(subconcept_of)
    assert set(concept.has_subconcept) == set(has_subconcept)
    assert set(concept.related_concepts) == set(related_concepts)
    assert concept.definition == definition


def test_whether_alternative_labels_are_unique():
    concept = Concept(preferred_label="Test", alternative_labels=["A", "B", "A", "C"])
    assert set(concept.alternative_labels) == {"A", "B", "C"}


def test_whether_preferred_label_is_removed_from_alternative_labels():
    concept = Concept(preferred_label="Test", alternative_labels=["Test", "A", "B"])
    assert "Test" not in concept.alternative_labels
    assert set(concept.alternative_labels) == {"A", "B"}


def test_whether_wikibase_url_is_correctly_generated():
    os.environ["WIKIBASE_URL"] = "https://example.com"
    concept = Concept(preferred_label="Test", wikibase_id=WikibaseID("Q123"))
    assert concept.wikibase_url == "https://example.com/wiki/Item:Q123"


def test_whether_wikibase_url_raises_error_with_missing_wikibase_url():
    if "WIKIBASE_URL" in os.environ:
        del os.environ["WIKIBASE_URL"]
    concept = Concept(preferred_label="Test", wikibase_id=WikibaseID("Q123"))
    with pytest.raises(ValueError, match="WIKIBASE_URL environment variable not set"):
        _ = concept.wikibase_url


def test_whether_wikibase_url_raises_error_with_missing_wikibase_id():
    os.environ["WIKIBASE_URL"] = "https://example.com"
    concept = Concept(preferred_label="Test")
    with pytest.raises(ValueError, match="No wikibase_id found for concept"):
        _ = concept.wikibase_url


def test_whether_all_labels_are_correctly_generated():
    concept = Concept(preferred_label="Test", alternative_labels=["A", "B", "C"])
    assert set(concept.all_labels) == {"Test", "A", "B", "C"}


def test_whether_concepts_are_hashable():
    concept1 = Concept(
        preferred_label="Test",
        alternative_labels=["A", "B"],
        wikibase_id=WikibaseID("Q123"),
    )
    concept2 = Concept(
        preferred_label="Test",
        alternative_labels=["B", "A"],
        wikibase_id=WikibaseID("Q123"),
    )
    assert hash(concept1) == hash(concept2)


@given(
    preferred_label=label_strategy,
    alternative_labels=st.lists(label_strategy, min_size=1),
    wikibase_id=wikibase_id_strategy,
)
def test_whether_repr_and_str_are_correctly_formatted(
    preferred_label, alternative_labels, wikibase_id
):
    concept = Concept(
        preferred_label=preferred_label,
        alternative_labels=alternative_labels,
        wikibase_id=wikibase_id,
    )
    assert repr(concept) == f'Concept({wikibase_id}, "{preferred_label}")'
    assert str(concept) != repr(concept)


def test_whether_blank_preferred_label_raises_validation_error():
    with pytest.raises(ValidationError):
        Concept(preferred_label="")


def test_whether_invalid_wikibase_id_raises_validation_error():
    with pytest.raises(ValidationError):
        Concept(preferred_label="Test", wikibase_id="invalid_id")
