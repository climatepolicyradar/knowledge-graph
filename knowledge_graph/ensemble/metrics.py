from abc import ABC
from statistics import stdev
from typing import Sequence

from knowledge_graph.span import Span, UnitInterval, all_spans_have_probability


def _validate_spans_per_classifier_are_at_passage_level(
    spans_per_classifier: Sequence[Sequence[Span]],
    metric_name: str,
):
    """
    Validate that spans input to ensemble metrics are at passage-level.

    This means there should only be one predicted span per classifier per text passage,
    and spans should all have the same start and end index.

    :raises ValueError: if any classifier's spans contains more than one Span
    """

    if any(len(spans) > 1 for spans in spans_per_classifier):
        raise ValueError(
            f"Spans passed to metric {metric_name} don't appear to be at passage-level. Some classifiers returned more than one span."
        )

    all_span_start_and_end_idxs = [
        (span.start_index, span.end_index)
        for spans in spans_per_classifier
        for span in spans
    ]

    if len(set(all_span_start_and_end_idxs)) > 1:
        raise ValueError(
            f"Spans passed to metric {metric_name} don't appear to be at passage-level. Not all spans have the same start and end index."
        )


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


class ProbabilityBasedEnsembleMetric(EnsembleMetric):
    """An ensemble metric which uses prediction probabilities in its calculation."""

    def __call__(self, spans_per_classifier: Sequence[Sequence[Span]]) -> UnitInterval:
        """
        Calculate a metric on spans produced by an ensemble of classifiers.

        :param Sequence[Sequence[SpanWithProbability]] spans_per_classifier: the spans output by
            each classifier predicting on one piece of text
        """
        raise NotImplementedError


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


class MajorityVote(EnsembleMetric):
    """The majority vote of an ensemble. Returns 0.5 if there's a 50-50 split."""

    def __call__(self, spans_per_classifier: Sequence[Sequence[Span]]) -> UnitInterval:
        """Calculate majority vote."""

        binary_predictions = [
            1 if predictions else 0 for predictions in spans_per_classifier
        ]

        num_positives = sum(binary_predictions)
        num_negatives = len(binary_predictions) - sum(binary_predictions)

        if num_positives == num_negatives:
            return UnitInterval(0.5)

        majority_vote = 1 if num_positives > num_negatives else 0

        return UnitInterval(majority_vote)


class PredictionProbabilityStandardDeviation(ProbabilityBasedEnsembleMetric):
    """The standard deviation of prediction probability for passage-level predictions."""

    def __call__(self, spans_per_classifier: Sequence[Sequence[Span]]) -> UnitInterval:
        """
        Calculate standard deviation of prediction probability.

        :raises ValueError: if there is more than one Span predicted per piece of text
            per classifier. This metric does not handle aggregating prediction
            probabilities for several spans on a passage of text.
        :raises ValueError: if there are insufficient spans to calculate standard deviation
            (need at least 2 spans with probabilities)
        """

        spans_flat = [span for spans in spans_per_classifier for span in spans]

        if len(spans_flat) == 0:
            raise ValueError(
                "Can't calculate probability standard deviation: "
                "no classifiers predicted positive (all span lists are empty)."
            )

        if len(spans_flat) < 2:
            raise ValueError(
                "Can't calculate probability standard deviation: "
                f"only {len(spans_flat)} span(s) found. Need at least 2 to calculate standard deviation."
            )

        if not all_spans_have_probability(spans_flat):
            raise ValueError(
                "For a probability-based ensemble metric, all spans must have prediction probabilities."
            )

        _validate_spans_per_classifier_are_at_passage_level(
            spans_per_classifier,
            self.name,
        )

        prediction_probs = [
            span.prediction_probability
            for spans in spans_per_classifier
            for span in spans
        ]

        return UnitInterval(stdev(prediction_probs))  # type: ignore[arg-type]
