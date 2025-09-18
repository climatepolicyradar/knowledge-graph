import logging
from typing import Sequence

from knowledge_graph.classifier.classifier import (
    Classifier,
)
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import ClassifierID
from knowledge_graph.span import Span

logger = logging.getLogger(__name__)


class IncompatibleSubClassifiersError(Exception):
    """Exception raised when classifiers don't share the same concept."""

    def __init__(self, reason: str):
        self.message = f"Classifiers attempting to be ensembled are incompatible.\nReason: {reason}"
        super().__init__(self.message)


class Ensemble:
    """
    A collection of classifiers.

    These can be used to improve the performance of a single classifier or measure
    its stability or uncertainty, by creating an ensemble containing slight variants
    of the same classifier.

    The `predict` and `predict_batch` methods here return lists of the same types as
    their equivalents on Classifier: one for each classifier.
    """

    def __init__(
        self,
        concept: Concept,
        classifiers: Sequence[Classifier],
    ):
        self._validate_classifiers(concept, classifiers)
        self.concept = concept
        self.classifiers = classifiers

    def _validate_classifiers(
        self,
        concept: Concept,
        classifiers: Sequence[Classifier],
    ) -> None:
        """Check that classifiers are compatible to be part of the same ensemble."""

        if invalid_concepts := {
            clf.concept for clf in classifiers if clf.concept != concept
        }:
            raise IncompatibleSubClassifiersError(
                f"All classifiers used in the ensemble must share the concept {concept}. Other concepts found: {invalid_concepts}"
            )

        unique_classifier_ids = {str(clf) for clf in classifiers}
        if len(unique_classifier_ids) < len(classifiers):
            raise IncompatibleSubClassifiersError(
                reason="All classifiers in the ensemble must be unique."
            )

    def predict(self, text: str) -> list[list[Span]]:
        """
        Run prediction for each classifier in the ensemble on the input text.

        :param str text: the text to predict on
        :return list[list[Span]]: a list of spans per classifier
        """

        return [clf.predict(text) for clf in self.classifiers]

    def predict_batch(self, texts: list[str]) -> list[list[list[Span]]]:
        """
        Run prediction for each classifier in the ensemble on the input text batch.

        :param Sequence[str] texts: the text to predict on
        :return list[list[list[Span]]]: a list of batches of spans per classifier
        """

        return [clf.predict_batch(texts) for clf in self.classifiers]

    @property
    def name(self) -> str:
        """Return a string representation of the ensemble type, i.e. name of the class."""
        return __class__.__name__

    @property
    def id(self) -> ClassifierID:
        """Return a deterministic, human-readable identifier for the ensemble"""
        ensembled_classifier_ids = [clf.id for clf in self.classifiers]

        return ClassifierID.generate(
            self.name, self.concept.id, "|".join(ensembled_classifier_ids)
        )

    def __repr__(self) -> str:
        """Return a string representation of the ensemble."""
        return str(self.id)
