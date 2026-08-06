import pytest

from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.metrics import (
    ConfusionMatrix,
    count_passage_level_metrics,
    count_span_level_metrics,
)
from knowledge_graph.span import Span

TEXT = "Solar power, wind power and coal power are all sources of electricity."
CONCEPT_ID = WikibaseID("Q42")


def span(start: int, end: int) -> Span:
    """Build a span over the shared test text."""
    return Span(
        text=TEXT,
        start_index=start,
        end_index=end,
        concept_id=CONCEPT_ID,
        labellers=["labeller"],
    )


def passage(spans: list[Span], passage_id: str = "passage-1") -> LabelledPassage:
    """Build a labelled passage over the shared test text."""
    return LabelledPassage(id=passage_id, text=TEXT, spans=spans)


def test_whether_every_unmatched_gold_span_counts_as_a_false_negative():
    """Each missed gold span is its own false negative, not one per passage."""
    gold = [passage([span(0, 11), span(13, 23), span(28, 38)])]
    predicted = [passage([])]

    cm = count_span_level_metrics(gold, predicted, threshold=0)

    assert cm.false_negatives == 3
    assert cm.true_positives == 0


def test_whether_every_unmatched_predicted_span_counts_as_a_false_positive():
    """Each spurious predicted span is its own false positive."""
    gold = [passage([])]
    predicted = [passage([span(0, 11), span(13, 23), span(28, 38)])]

    cm = count_span_level_metrics(gold, predicted, threshold=0)

    assert cm.false_positives == 3
    assert cm.true_negatives == 0


def test_whether_matches_are_still_counted_after_an_earlier_miss():
    """A missed gold span must not stop the remaining gold spans being scored."""
    gold = [passage([span(0, 11), span(13, 23), span(28, 38)])]
    # Matches the second and third gold spans, misses the first
    predicted = [passage([span(13, 23), span(28, 38)])]

    cm = count_span_level_metrics(gold, predicted, threshold=0)

    assert cm.true_positives == 2
    assert cm.false_negatives == 1
    assert cm.false_positives == 0


def test_whether_a_passage_with_no_spans_either_side_is_a_true_negative():
    """Agreement that a passage contains nothing is a true negative."""
    cm = count_span_level_metrics([passage([])], [passage([])], threshold=0)

    assert cm.true_negatives == 1
    assert cm.support() == 1


def test_whether_passage_level_metrics_count_passages_not_spans():
    """Passage-level scoring collapses however many spans a passage has into one label."""
    gold = [
        passage([span(0, 11), span(13, 23)], passage_id="a"),
        passage([], passage_id="b"),
    ]
    predicted = [
        passage([span(0, 11)], passage_id="a"),
        passage([span(0, 11)], passage_id="b"),
    ]

    cm = count_passage_level_metrics(gold, predicted)

    assert cm.true_positives == 1
    assert cm.false_positives == 1
    assert cm.support() == 2


def test_whether_the_negative_class_metrics_mirror_the_positive_ones():
    """NPV and specificity are precision and recall read off the negative class."""
    cm = ConfusionMatrix(
        true_positives=4, false_positives=2, true_negatives=10, false_negatives=4
    )

    assert cm.negative_predictive_value() == pytest.approx(10 / 14)
    assert cm.specificity() == pytest.approx(10 / 12)


def test_whether_the_negative_class_metrics_survive_an_empty_denominator():
    """
    Nothing predicted negative and nothing gold-negative, so neither rate is defined.

    They return 0 rather than raising, matching `precision`/`recall` — callers that need
    to tell "undefined" from "scored zero" have to check the counts themselves.
    """
    cm = ConfusionMatrix(true_positives=3, false_positives=0)

    assert cm.negative_predictive_value() == 0
    assert cm.specificity() == 0
