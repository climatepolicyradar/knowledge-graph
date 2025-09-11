"""Tests for isinstance behaviour of classifier mixins."""

from knowledge_graph.classifier.classifier import (
    Classifier,
    GPUBoundClassifier,
    ZeroShotClassifier,
)
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import ClassifierID
from knowledge_graph.span import Span


class DummyClassifier(Classifier):
    """A dummy classifier for testing purposes."""

    def predict(self, text: str) -> list[Span]:
        """Predicts nothing."""
        return []

    @property
    def id(self) -> ClassifierID:
        """Return the ID of the classifier."""
        return ClassifierID("dummy")


class DummyZeroShotClassifier(DummyClassifier, ZeroShotClassifier):
    """A dummy zero-shot classifier."""


class DummyGpuClassifier(DummyClassifier, GPUBoundClassifier):
    """A dummy GPU classifier."""


concept = Concept(wikibase_id="Q1", preferred_label="test")


def test_isinstance_for_a_zero_shot_classifier():
    """Test that a zero-shot classifier is an instance of ZeroShotClassifier."""
    classifier = DummyZeroShotClassifier(concept)
    assert isinstance(classifier, ZeroShotClassifier)


def test_isinstance_for_a_non_zero_shot_classifier():
    """Test that a non-zero-shot classifier is not an instance of ZeroShotClassifier."""
    classifier = DummyClassifier(concept)
    assert not isinstance(classifier, ZeroShotClassifier)


def test_isinstance_for_a_gpu_classifier():
    """Test that a GPU classifier is an instance of the GPU marker class."""
    classifier = DummyGpuClassifier(concept)
    assert isinstance(classifier, GPUBoundClassifier)


def test_isinstance_for_a_non_gpu_classifier():
    """Test that a non-GPU classifier is not an instance of the GPU marker class."""
    classifier = DummyClassifier(concept)
    assert not isinstance(classifier, GPUBoundClassifier)
