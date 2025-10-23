import logging
from typing import Any, Sequence, cast

from knowledge_graph.classifier import ClassifierFactory
from knowledge_graph.classifier.classifier import Classifier, VariantEnabledClassifier
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

        Spans are returned with the outer list being batches, and the inner list being
        classifiers. This is to make the output consistent with `Ensemble().predict`.

        :param Sequence[str] texts: the text to predict on
        :return list[list[list[Span]]]: a list of spans per classifier per batch
        """

        # this is in the format classifier -> batch -> spans
        spans_per_batch_per_classifier = [
            clf.predict(texts, batch_size=len(texts)) for clf in self.classifiers
        ]

        # transpose to batch -> classifier -> spans
        return [
            [
                spans_per_batch_per_classifier[clf_idx][batch_idx]
                for clf_idx in range(len(self.classifiers))
            ]
            for batch_idx in range(len(texts))
        ]

    @property
    def name(self) -> str:
        """Return a string representation of the ensemble type, i.e. name of the class."""
        return self.__class__.__name__

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


def create_ensemble(
    concept: Concept,
    n_classifiers: int,
    classifier_type: str,
    classifier_kwargs: dict[str, Any] = {},
) -> Ensemble:
    """
    Create an ensemble of classifiers for a concept.

    :raises ValueError: if the classifier_type is not variant-enabled.
    """

    initial_classifier = ClassifierFactory.create(
        concept=concept,
        classifier_type=classifier_type,
        classifier_kwargs=classifier_kwargs,
    )
    if not isinstance(initial_classifier, VariantEnabledClassifier):
        raise ValueError(
            f"Classifier type must be variant-enabled to be part of an ensemble.\nClassifier type {classifier_type} is not."
        )

    # TODO: warn that random seed will be ignored for LLMClassifier

    # cast is needed here as list is invariant, so list[Classifier] is incompatible
    # with list[VariantEnabledClassifier]
    classifiers: list[Classifier] = [
        initial_classifier,
        *[
            cast(Classifier, initial_classifier.get_variant())
            for _ in range(n_classifiers - 1)
        ],
    ]

    ensemble = Ensemble(
        concept=concept,
        classifiers=classifiers,
    )

    return ensemble
