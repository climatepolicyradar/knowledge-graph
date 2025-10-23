import pytest
from hypothesis import given

from knowledge_graph.classifier.classifier import Classifier
from knowledge_graph.classifier.keyword import KeywordClassifier
from knowledge_graph.concept import Concept
from knowledge_graph.ensemble import (
    Ensemble,
    IncompatibleSubClassifiersError,
)
from knowledge_graph.identifiers import ClassifierID, WikibaseID
from knowledge_graph.span import Span
from tests.common_strategies import concept_strategy


@pytest.mark.xdist_group(name="classifier")
def test_whether_ensemble_rejects_classifiers_with_different_concepts():
    """Test that Ensemble raises error when classifiers have different concepts."""
    concept1 = Concept(wikibase_id=WikibaseID("Q1"), preferred_label="test1")
    concept2 = Concept(wikibase_id=WikibaseID("Q2"), preferred_label="test2")

    classifier1 = KeywordClassifier(concept1)
    classifier2 = KeywordClassifier(concept2)

    with pytest.raises(IncompatibleSubClassifiersError):
        Ensemble(concept1, [classifier1, classifier2])


@pytest.mark.xdist_group(name="classifier")
def test_whether_ensemble_rejects_nonunique_lists_of_classifiers():
    """Test that Ensemble raises error when classifiers don't have unique IDs."""
    concept1 = Concept(wikibase_id=WikibaseID("Q1"), preferred_label="test1")

    classifier1 = KeywordClassifier(concept1)
    classifier2 = KeywordClassifier(concept1)

    with pytest.raises(IncompatibleSubClassifiersError):
        Ensemble(concept1, [classifier1, classifier2])


@pytest.mark.xdist_group(name="classifier")
@given(concept=concept_strategy())
def test_whether_ensemble_id_is_deterministic(concept: Concept):
    """Test that Ensemble generates deterministic IDs."""

    # Create two simple mock classifiers with different IDs
    class MockClassifier1(Classifier):
        @property
        def id(self) -> ClassifierID:
            return ClassifierID("testmck2")

        def predict(self, text: str) -> list[Span]:
            return []

    class MockClassifier2(Classifier):
        @property
        def id(self) -> ClassifierID:
            return ClassifierID("testmck3")

        def predict(self, text: str) -> list[Span]:
            return []

    classifier1 = MockClassifier1(concept)
    classifier2 = MockClassifier2(concept)

    ensemble1 = Ensemble(concept, [classifier1, classifier2])
    ensemble2 = Ensemble(concept, [classifier1, classifier2])

    assert ensemble1.id == ensemble2.id
    assert isinstance(ensemble1.id, ClassifierID)
