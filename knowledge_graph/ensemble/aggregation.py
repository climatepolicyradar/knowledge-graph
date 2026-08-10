"""
Aggregation of an ensemble's per-classifier spans into a single set of spans.

`Ensemble.predict` returns one list of spans per classifier, which is what the
`EnsembleMetric`s in `knowledge_graph.ensemble.metrics` consume. Span-level
evaluation (`knowledge_graph.metrics.count_span_level_metrics`) instead needs a
single set of spans per passage, as if the ensemble were one classifier. The
aggregators here perform that reduction.
"""

from abc import ABC, abstractmethod
from typing import Sequence

from knowledge_graph.ensemble.metrics import MajorityVote
from knowledge_graph.span import Span, merge_overlapping_spans


class SpanAggregator(ABC):
    """Reduces an ensemble's per-classifier spans to a single list of spans."""

    @abstractmethod
    def __call__(self, spans_per_classifier: Sequence[Sequence[Span]]) -> list[Span]:
        """
        Aggregate spans produced by an ensemble of classifiers.

        :param Sequence[Sequence[Span]] spans_per_classifier: the spans output by
            each classifier predicting on one piece of text
        :return list[Span]: the aggregated, non-overlapping spans
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Return the name of the aggregator."""
        return self.__class__.__name__


class UnionAggregator(SpanAggregator):
    """
    Takes the union of all spans predicted by the ensemble's classifiers.

    If passage-level results are derived from spans, this means that a passage is
    positive if *any of the ensemble's classifiers* predicted a span on it.
    """

    def __init__(self, jaccard_threshold: float = 0):
        self.jaccard_threshold = jaccard_threshold

    def __call__(self, spans_per_classifier: Sequence[Sequence[Span]]) -> list[Span]:
        """Merge all classifiers' spans into a list of non-overlapping spans."""

        all_spans = [span for spans in spans_per_classifier for span in spans]

        if not all_spans:
            return []

        return merge_overlapping_spans(all_spans, self.jaccard_threshold)


class MajorityVoteAggregator(SpanAggregator):
    """
    The union of spans, but only if a majority of classifiers predicted anything at all.

    Note the vote does not require the classifiers to agree on *where* the mention is:
    five classifiers each flagging a different part of a passage are unanimous here, even
    though they agree on no single span.

    A 50/50 split counts as positive, matching `MajorityVote`.
    """

    def __init__(self, jaccard_threshold: float = 0):
        self.jaccard_threshold = jaccard_threshold
        self._majority_vote = MajorityVote()

    def __call__(self, spans_per_classifier: Sequence[Sequence[Span]]) -> list[Span]:
        """Merge all classifiers' spans, or return none if the vote fails."""

        if self._majority_vote(spans_per_classifier) < 0.5:
            return []

        return UnionAggregator(self.jaccard_threshold)(spans_per_classifier)
