from knowledge_graph.ensemble.aggregation import (
    MajorityVoteAggregator,
    UnionAggregator,
)
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.span import Span

TEXT = "The just transition must protect coal workers and their communities."
CONCEPT_ID = WikibaseID("Q42")


def span(start: int, end: int, labeller: str = "classifier") -> Span:
    """Build a span over the shared test text."""
    return Span(
        text=TEXT,
        start_index=start,
        end_index=end,
        concept_id=CONCEPT_ID,
        labellers=[labeller],
    )


def offsets(spans: list[Span]) -> list[tuple[int, int]]:
    """Return the sorted (start, end) offsets of a list of spans."""
    return sorted((s.start_index, s.end_index) for s in spans)


def test_whether_union_of_empty_predictions_is_no_spans():
    """An ensemble where every classifier predicted nothing aggregates to no spans."""
    assert UnionAggregator()([[], [], []]) == []


def test_whether_union_merges_overlapping_spans():
    """Overlapping spans from different classifiers become one span covering both."""
    aggregated = UnionAggregator()([[span(4, 20)], [span(9, 32)]])

    assert offsets(aggregated) == [(4, 32)]


def test_whether_union_keeps_disjoint_spans_separate():
    """Non-overlapping spans are all retained, unmerged."""
    aggregated = UnionAggregator()([[span(4, 20)], [span(33, 37)]])

    assert offsets(aggregated) == [(4, 20), (33, 37)]


def test_whether_union_keeps_a_span_only_one_classifier_found():
    """The union is permissive: a single classifier's span survives aggregation."""
    aggregated = UnionAggregator()([[span(4, 20)], [], []])

    assert offsets(aggregated) == [(4, 20)]


def test_whether_union_preserves_the_labellers_of_merged_spans():
    """Merging keeps every contributing classifier's labeller, for traceability."""
    aggregated = UnionAggregator()([[span(4, 20, "opus")], [span(9, 32, "glm")]])

    assert set(aggregated[0].labellers) == {"opus", "glm"}


def test_whether_majority_vote_drops_a_span_a_minority_found():
    """One classifier out of five isn't enough to call the passage positive."""
    assert MajorityVoteAggregator()([[span(4, 20)], [], [], [], []]) == []


def test_whether_majority_vote_keeps_spans_a_majority_found():
    """Three of five is a majority, so the union of their spans is returned."""
    aggregated = MajorityVoteAggregator()(
        [[span(4, 20)], [span(9, 32)], [span(13, 20)], [], []]
    )

    assert offsets(aggregated) == [(4, 32)]


def test_whether_majority_vote_counts_classifiers_that_found_different_spans():
    """
    The vote is on whether each classifier found *anything*, not on where.

    Classifiers flagging different parts of a passage are unanimous about the passage,
    even though they agree on no single span, so every span is kept.
    """
    disjoint = [[span(4, 20)], [span(33, 37)], [span(43, 51)]]

    assert offsets(MajorityVoteAggregator()(disjoint)) == [(4, 20), (33, 37), (43, 51)]


def test_whether_majority_vote_returns_nothing_when_no_classifier_found_anything():
    """A unanimous negative is still a negative."""
    assert MajorityVoteAggregator()([[], [], []]) == []


def test_whether_majority_vote_counts_a_tie_as_positive():
    """Documented behaviour for even-sized ensembles; odd sizes can't tie."""
    aggregated = MajorityVoteAggregator()([[span(4, 20)], []])

    assert offsets(aggregated) == [(4, 20)]
