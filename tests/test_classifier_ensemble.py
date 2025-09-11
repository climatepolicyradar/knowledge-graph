import builtins

import pytest
from hypothesis import given

from src.classifier.classifier import ProbabilityCapableClassifier
from src.classifier.ensemble import IncompatibleSubClassifiersError, VotingClassifier
from src.classifier.keyword import KeywordClassifier
from src.classifier.stemmed_keyword import StemmedKeywordClassifier
from src.concept import Concept
from src.identifiers import ClassifierID, WikibaseID
from tests.common_strategies import concept_strategy


@pytest.mark.xdist_group(name="classifier")
def test_whether_voting_classifier_rejects_classifiers_with_different_concepts():
    """Test that VotingClassifier raises error when classifiers have different concepts."""
    concept1 = Concept(wikibase_id=WikibaseID("Q1"), preferred_label="test1")
    concept2 = Concept(wikibase_id=WikibaseID("Q2"), preferred_label="test2")

    classifier1 = KeywordClassifier(concept1)
    classifier2 = KeywordClassifier(concept2)

    with pytest.raises(IncompatibleSubClassifiersError):
        VotingClassifier(concept1, [classifier1, classifier2])


@pytest.mark.xdist_group(name="classifier")
def test_whether_voting_classifier_rejects_nonunique_lists_of_classifiers():
    """Test that VotingClassifier raises error when classifiers don't have unique IDs."""
    concept1 = Concept(wikibase_id=WikibaseID("Q1"), preferred_label="test1")

    classifier1 = KeywordClassifier(concept1)
    classifier2 = KeywordClassifier(concept1)

    with pytest.raises(IncompatibleSubClassifiersError):
        VotingClassifier(concept1, [classifier1, classifier2])


@pytest.mark.xdist_group(name="classifier")
def test_whether_voting_classifier_combines_predictions_with_probabilities():
    """Test that VotingClassifier combines overlapping predictions and assigns probabilities."""
    concept = Concept(wikibase_id=WikibaseID("Q1"), preferred_label="test")
    text = "This is a test sentence with test words."

    classifier1 = KeywordClassifier(concept)
    classifier2 = StemmedKeywordClassifier(concept)

    voting_classifier = VotingClassifier(concept, [classifier1, classifier2])
    spans = voting_classifier.predict(text)

    assert len(spans) == 2

    for span in spans:
        assert span.prediction_probability is not None
        assert 0 <= span.prediction_probability <= 1
        assert span.labellers == [str(voting_classifier)]


@pytest.mark.xdist_group(name="classifier")
def test_whether_voting_classifier_assigns_correct_probabilities(
    zero_returns_classifier,
):
    """Test that probabilities are calculated as proportion of classifiers that predicted."""
    concept = Concept(wikibase_id=WikibaseID("Q1"), preferred_label="test")
    text = "test"

    classifiers_unanimous = [
        KeywordClassifier(concept),
        StemmedKeywordClassifier(concept),
    ]

    classifiers_contentious = classifiers_unanimous + [zero_returns_classifier(concept)]

    voting_classifier = VotingClassifier(concept, classifiers_unanimous)
    voting_classifier_2 = VotingClassifier(concept, classifiers_contentious)

    unanimous_spans = voting_classifier.predict(text)

    assert len(unanimous_spans) == 1
    assert unanimous_spans[0].prediction_probability == 1.0

    contentious_spans = voting_classifier_2.predict(text)
    assert len(contentious_spans) == 1
    assert contentious_spans[0].prediction_probability == 2 / 3


@pytest.mark.xdist_group(name="classifier")
@given(concept=concept_strategy())
def test_whether_voting_classifier_id_is_deterministic(concept: Concept):
    """Test that VotingClassifier generates deterministic IDs."""
    classifier1 = KeywordClassifier(concept)
    classifier2 = StemmedKeywordClassifier(concept)

    voting_classifier1 = VotingClassifier(concept, [classifier1, classifier2])
    voting_classifier2 = VotingClassifier(concept, [classifier1, classifier2])

    assert voting_classifier1.id == voting_classifier2.id
    assert isinstance(voting_classifier1.id, ClassifierID)


@pytest.mark.xdist_group(name="classifier")
def test_whether_voting_classifier_handles_no_predictions():
    """Test that VotingClassifier returns empty list when no classifiers predict anything."""
    concept = Concept(wikibase_id=WikibaseID("Q1"), preferred_label="nonexistent")
    text = "This text contains no matches."

    classifier = KeywordClassifier(concept)
    voting_classifier = VotingClassifier(concept, [classifier])

    spans = voting_classifier.predict(text)
    assert spans == []


@pytest.mark.xdist_group(name="classifier")
def test_whether_voting_classifier_outputs_correct_probabilities_at_passage_level(
    zero_returns_classifier,
):
    """Test that VotingClassifier outputs correct probabilities at passage level."""
    concept = Concept(wikibase_id=WikibaseID("Q1"), preferred_label="test")
    text = "test test sample text"

    classifier1 = KeywordClassifier(concept)
    classifier2 = StemmedKeywordClassifier(concept)
    classifier3 = zero_returns_classifier(concept)

    voting_classifier1 = VotingClassifier(
        concept, classifiers=[classifier1, classifier2]
    )
    voting_classifier2 = VotingClassifier(
        concept, classifiers=[classifier1, classifier2, classifier3]
    )

    expected_unanimous_prediction = voting_classifier1.predict(text, passage_level=True)

    assert len(expected_unanimous_prediction) == 1
    assert expected_unanimous_prediction[0].prediction_probability == 1

    expected_contentious_prediction = voting_classifier2.predict(
        text, passage_level=True
    )
    assert len(expected_contentious_prediction) == 1
    assert expected_contentious_prediction[0].prediction_probability == 2 / 3


@pytest.mark.xdist_group(name="classifier")
def test_warn_for_any_probability_capable_classifiers(caplog):
    """Test that _warn_for_any_probability_capable_classifiers logs warning for probability-capable classifiers."""
    concept = Concept(wikibase_id=WikibaseID("Q1"), preferred_label="test")

    regular_classifier = KeywordClassifier(concept)
    voting_classifier = VotingClassifier(concept, [regular_classifier])

    voting_classifier._warn_for_any_probability_capable_classifiers(
        [regular_classifier]
    )
    assert len(caplog.records) == 0
    caplog.clear()

    # Patch isinstance to identify all classifiers as probability-capable
    with pytest.MonkeyPatch().context() as m:
        original_isinstance = builtins.isinstance

        def mock_isinstance(obj, cls):
            if cls is ProbabilityCapableClassifier:
                return True
            return original_isinstance(obj, cls)

        m.setattr("builtins.isinstance", mock_isinstance)
        voting_classifier = VotingClassifier(concept, [regular_classifier])

    assert len(caplog.records) == 1
    assert caplog.records[0].levelname == "WARNING"
