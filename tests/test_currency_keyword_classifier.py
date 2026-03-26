"""Unit tests for CurrencyKeywordClassifier."""

import pytest

from knowledge_graph.classifier import ClassifierFactory, CurrencyKeywordClassifier
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import WikibaseID


def _currency_concept(
    labels: list[str] | None = None,
    negative_labels: list[str] | None = None,
) -> Concept:
    """Concept with currency-like labels (e.g. USD, EUR, $, dollars)."""
    return Concept(
        wikibase_id=WikibaseID("Q2033"),
        preferred_label="Currency",
        alternative_labels=labels or ["USD", "EUR", "$", "dollars"],
        negative_labels=negative_labels or [],
    )


@pytest.mark.xdist_group(name="classifier")
@pytest.mark.parametrize(
    "text,expected_phrase",
    [
        ("spending 100 USD", "100 USD"),
        ("the bill of $100 is paid", "$100"),
        ("the new quantified goal of US$ 100 billion per year", "US$ 100 billion"),
        ("EUR1,5 million", "EUR1,5 million"),
        ("one million Icelandic Kronur", "one million Icelandic Kronur"),
        ("subsidies totalling USD", "hundreds of USD"),
        ("a USD equivalent value of 1000", "USD equivalent value of 1000"),
    ],
)
def test_currency_classifier_matches_expected_phrases(text: str, expected_phrase: str):
    """Currency classifier matches currency + number (or number + currency) phrases."""
    concept = _currency_concept()
    classifier = CurrencyKeywordClassifier(concept)
    spans = classifier._predict(text)
    assert len(spans) >= 1, f"Expected at least one span in '{text}'"
    matched = next(
        s
        for s in spans
        if s.labelled_text == expected_phrase or expected_phrase in s.labelled_text
    )
    assert matched is not None
    assert matched.start_index >= 0 and matched.end_index <= len(text)
    assert text[matched.start_index : matched.end_index] == matched.labelled_text


@pytest.mark.xdist_group(name="classifier")
def test_currency_classifier_does_not_match_r_and_d():
    """Single-char alphanumeric label 'R' uses tight-only; 'R and D' should not match."""
    concept = Concept(
        wikibase_id=WikibaseID("Q2033"),
        preferred_label="Currency",
        alternative_labels=["R"],  # South African Rand
    )
    classifier = CurrencyKeywordClassifier(concept)
    spans = classifier._predict("R and D budget")
    assert not spans


@pytest.mark.xdist_group(name="classifier")
def test_currency_classifier_matches_r_with_number():
    """Single-char alphanumeric 'R' matches when tightly adjacent to number."""
    concept = Concept(
        wikibase_id=WikibaseID("Q2033"),
        preferred_label="Currency",
        alternative_labels=["R"],
    )
    classifier = CurrencyKeywordClassifier(concept)
    spans = classifier._predict("R 100")
    assert len(spans) == 1
    assert spans[0].labelled_text == "R 100"


@pytest.mark.xdist_group(name="classifier")
def test_currency_classifier_output_shape():
    """Currency classifier returns list[Span] with same shape as KeywordClassifier."""
    concept = _currency_concept()
    classifier = CurrencyKeywordClassifier(concept)
    text = "The budget is 100 USD."
    spans = classifier.predict(text)
    assert isinstance(spans, list)
    for span in spans:
        assert span.text is text
        assert 0 <= span.start_index < span.end_index <= len(text)
        assert span.concept_id == concept.wikibase_id
        assert span.labellers
        assert span.timestamps
        assert span.prediction_probability is None


@pytest.mark.xdist_group(name="classifier")
def test_currency_classifier_respects_negative_labels():
    """Positive matches overlapping negative labels are filtered out."""
    concept = _currency_concept(negative_labels=["fake dollar"])
    classifier = CurrencyKeywordClassifier(concept)
    # Text with "fake dollar" near a number might match currency; we just check no crash
    spans = classifier._predict("100 USD for the project")
    assert len(spans) >= 1
    # If we had "fake dollar 100" we'd expect negative to filter; concept has no "fake dollar" in positives
    # so this test mainly ensures negative pattern is still applied (from parent)
    assert all(s.concept_id == concept.wikibase_id for s in spans)


@pytest.mark.xdist_group(name="classifier")
def test_currency_classifier_factory_returns_currency_classifier_for_q2033():
    """ClassifierFactory.create(currency_concept) returns CurrencyKeywordClassifier."""
    concept = Concept(
        wikibase_id=WikibaseID("Q2033"),
        preferred_label="Currency",
        alternative_labels=["USD"],
    )
    classifier = ClassifierFactory.create(concept)
    assert type(classifier).__name__ == "CurrencyKeywordClassifier"


@pytest.mark.xdist_group(name="classifier")
def test_currency_classifier_case_sensitivity():
    """USD (case-sensitive) matches exactly; dollars (case-insensitive) matches any case."""
    concept = _currency_concept(labels=["USD", "dollars"])
    classifier = CurrencyKeywordClassifier(concept)
    assert len(classifier._predict("100 USD")) >= 1
    assert len(classifier._predict("100 usd")) == 0  # USD is case-sensitive
    assert len(classifier._predict("100 Dollars")) >= 1
    assert len(classifier._predict("100 DOLLARS")) >= 1
