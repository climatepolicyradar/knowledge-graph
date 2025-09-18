from abc import ABC
from typing import Sequence

from knowledge_graph.span import Span, UnitInterval


class EnsembleMetric(ABC):
    """Base class for a metric calculated on the outputs of a classifier ensemble."""

    def __call__(self, spans_per_classifier: Sequence[Sequence[Span]]) -> UnitInterval:
        """
        Calculate a metric on spans produced by an ensemble of classifiers.

        :param Sequence[Sequence[Span]] spans_per_classifier: the spans output by
            each classifier predicting on one piece of text
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Return the name of the metric."""
        return self.__class__.__name__


class PositiveRatio(EnsembleMetric):
    """The proportion of classifiers that predicted the concept was in the text."""

    def __call__(self, spans_per_classifier: Sequence[Sequence[Span]]) -> UnitInterval:
        """Calculate positive ratio."""

        if all(not spans for spans in spans_per_classifier):
            return UnitInterval(0)

        binary_predictions = [
            1 if predictions else 0 for predictions in spans_per_classifier
        ]
        positive_ratio = sum(binary_predictions) / len(binary_predictions)

        return UnitInterval(positive_ratio)


class Disagreement(EnsembleMetric):
    """The number of disagreements between classifiers at a passage-level."""

    def __call__(self, spans_per_classifier: Sequence[Sequence[Span]]) -> UnitInterval:
        """Calculate disagreement."""

        binary_predictions = [
            1 if predictions else 0 for predictions in spans_per_classifier
        ]

        num_positives = sum(binary_predictions)
        num_negatives = len(binary_predictions) - sum(binary_predictions)

        # multiplied by two to scale it to between 0 and 1
        disagreement = 2 * min(num_positives, num_negatives) / len(binary_predictions)

        return UnitInterval(disagreement)
