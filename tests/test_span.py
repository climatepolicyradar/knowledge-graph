import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.span import Span, group_overlapping_spans, jaccard_similarity

text_strategy = st.text(min_size=10, max_size=1000)
labeller_strategy = st.text(min_size=1, max_size=10)
wikibase_id_strategy = st.from_regex(r"^Q[1-9]\d*$", fullmatch=True)


@st.composite
def span_inputs_strategy(draw):
    text = draw(text_strategy)
    start = draw(st.integers(min_value=0, max_value=len(text) - 1))
    end = draw(st.integers(min_value=start + 1, max_value=len(text)))
    concept_id = draw(wikibase_id_strategy)
    labellers = draw(st.lists(labeller_strategy, min_size=1, max_size=3))
    return text, start, end, concept_id, labellers


@st.composite
def span_strategy(draw):
    text, start, end, concept_id, labellers = draw(span_inputs_strategy())
    return Span(
        text=text,
        start_index=start,
        end_index=end,
        concept_id=concept_id,
        labellers=labellers,
    )


@st.composite
def overlapping_spans_strategy(draw, text):
    start_a = draw(st.integers(min_value=0, max_value=len(text) - 1))
    end_a = draw(st.integers(min_value=start_a + 1, max_value=len(text)))
    span_a = Span(
        text=text,
        start_index=start_a,
        end_index=end_a,
        concept_id=draw(wikibase_id_strategy),
        labellers=draw(st.lists(labeller_strategy, min_size=1, max_size=3)),
    )

    start_b = draw(st.integers(min_value=start_a, max_value=end_a - 1))
    end_b = draw(st.integers(min_value=start_b + 1, max_value=len(text)))
    span_b = Span(
        text=text,
        start_index=start_b,
        end_index=end_b,
        concept_id=draw(wikibase_id_strategy),
        labellers=draw(st.lists(labeller_strategy, min_size=1, max_size=3)),
    )
    return span_a, span_b


@st.composite
def non_overlapping_spans_strategy(draw, text):
    start_a = draw(st.integers(min_value=0, max_value=len(text) // 2 - 1))
    end_a = draw(st.integers(min_value=start_a + 1, max_value=len(text) // 2))
    span_a = Span(
        text=text,
        start_index=start_a,
        end_index=end_a,
        concept_id=draw(wikibase_id_strategy),
        labellers=draw(st.lists(labeller_strategy, min_size=1, max_size=3)),
    )

    start_b = draw(st.integers(min_value=end_a, max_value=len(text) - 1))
    end_b = draw(st.integers(min_value=start_b + 1, max_value=len(text)))
    span_b = Span(
        text=text,
        start_index=start_b,
        end_index=end_b,
        concept_id=draw(wikibase_id_strategy),
        labellers=draw(st.lists(labeller_strategy, min_size=1, max_size=3)),
    )
    return span_a, span_b


@st.composite
def fully_entailed_spans_strategy(draw, text):
    start_a = draw(st.integers(min_value=0, max_value=len(text) - 2))
    end_a = draw(st.integers(min_value=start_a + 2, max_value=len(text)))
    span_a = Span(
        text=text,
        start_index=start_a,
        end_index=end_a,
        concept_id=draw(wikibase_id_strategy),
        labellers=draw(st.lists(labeller_strategy, min_size=1, max_size=3)),
    )

    start_b = draw(st.integers(min_value=start_a, max_value=end_a - 1))
    end_b = draw(st.integers(min_value=start_b + 1, max_value=end_a))
    span_b = Span(
        text=text,
        start_index=start_b,
        end_index=end_b,
        concept_id=draw(wikibase_id_strategy),
        labellers=draw(st.lists(labeller_strategy, min_size=1, max_size=3)),
    )

    return span_a, span_b


@given(
    inputs=span_inputs_strategy(),
)
def test_whether_span_is_correctly_initialised(inputs):
    text, start_index, end_index, concept_id, labellers = inputs
    span = Span(
        text=text,
        start_index=start_index,
        end_index=end_index,
        concept_id=concept_id,
        labellers=labellers,
    )
    assert span.text == text
    assert span.start_index == start_index
    assert span.end_index == end_index
    assert span.concept_id == concept_id
    assert span.labellers == labellers
    assert span.labelled_text == text[start_index:end_index]


def test_whether_span_raises_value_error_on_with_start_index_greater_than_end_index():
    with pytest.raises(ValueError):
        Span(text="Example text", start_index=5, end_index=3)


def test_whether_span_raises_value_error_on_with_start_index_equal_to_end_index():
    with pytest.raises(ValueError):
        Span(text="Example text", start_index=5, end_index=5)


def test_whether_span_raises_value_error_on_with_start_index_less_than_zero():
    with pytest.raises(ValueError):
        Span(text="Example text", start_index=-1, end_index=3)


def test_whether_span_raises_value_error_on_with_end_index_greater_than_text_length():
    with pytest.raises(ValueError):
        Span(text="Example text", start_index=0, end_index=20)


@given(span=span_strategy())
def test_labelled_text_property(span):
    assert span.labelled_text == span.text[span.start_index : span.end_index]


@given(span=span_strategy())
def test_span_length(span):
    assert len(span) == span.end_index - span.start_index


@given(span=span_strategy())
def test_span_hash(span):
    assert isinstance(hash(span), int)


@given(span=span_strategy())
def test_whether_spans_with_same_values_are_equal(span):
    other_span = span.model_copy(deep=True)
    assert span == other_span


@given(span=span_strategy())
def test_whether_spans_with_different_text_are_non_equal(span):
    other_span = span.model_copy(update={"text": span.text + " more text"}, deep=True)
    assert span != other_span


@given(span=span_strategy())
def test_whether_spans_with_different_start_indices_are_non_equal(span):
    if span.start_index > 0:
        other_span = span.model_copy(
            update={"start_index": span.start_index - 1}, deep=True
        )
        assert span != other_span


@given(span=span_strategy())
def test_whether_spans_with_different_end_indices_are_non_equal(span):
    if span.end_index < len(span.text):
        other_span = span.model_copy(
            update={"end_index": span.end_index + 1}, deep=True
        )
        assert span != other_span


@given(span=span_strategy())
def test_whether_spans_with_different_concept_ids_are_non_equal(span):
    if span.concept_id:
        other_span = span.model_copy(
            update={"concept_id": span.concept_id + "123"}, deep=True
        )
        assert span != other_span


@given(span=span_strategy())
def test_whether_equivalent_spans_with_different_labellers_are_equal(span: Span):
    other_span = span.model_copy(
        update={"labellers": span.labellers + ["new_labeller"]}, deep=True
    )
    assert span.id == other_span.id
    assert span == other_span


@given(span=span_strategy())
def test_whether_jaccard_similarity_raises_value_error_on_different_text(span):
    other_span = span.model_copy(update={"text": span.text + " more text"}, deep=True)
    with pytest.raises(ValueError, match="The spans must have the same text"):
        jaccard_similarity(span, other_span)


@given(text=text_strategy, spans=st.data())
def test_whether_overlapping_spans_overlap(text, spans):
    span_a, span_b = spans.draw(overlapping_spans_strategy(text))
    assert span_a.overlaps(span_b)
    assert span_b.overlaps(span_a)


@given(text=text_strategy, spans=st.data())
def test_whether_non_overlapping_spans_do_not_overlap(text, spans):
    span_a, span_b = spans.draw(non_overlapping_spans_strategy(text))
    assert not span_a.overlaps(span_b)
    assert not span_b.overlaps(span_a)


@given(text=text_strategy, spans=st.data())
def test_whether_fully_entailed_spans_overlap(text, spans):
    span_a, span_b = spans.draw(fully_entailed_spans_strategy(text))
    assert span_a.overlaps(span_b)
    assert span_b.overlaps(span_a)


@given(text=text_strategy, spans=st.data())
def test_whether_group_overlapping_spans_returns_correct_number_of_groups(text, spans):
    overlapping_spans = spans.draw(
        st.lists(overlapping_spans_strategy(text), min_size=2)
    )
    groups = group_overlapping_spans(
        spans=[span for pair in overlapping_spans for span in pair]
    )
    assert len(groups) <= len(overlapping_spans)


@given(text=text_strategy, spans=st.data())
def test_whether_group_overlapping_spans_returns_correct_number_of_groups_with_non_overlapping_spans(
    text, spans
):
    span_a, span_b = spans.draw(non_overlapping_spans_strategy(text))
    groups = group_overlapping_spans(spans=[span_a, span_b])
    assert len(groups) == 2


@given(text=text_strategy, spans=st.data())
def test_whether_group_overlapping_spans_returns_correct_number_of_groups_with_fully_entailed_spans(
    text, spans
):
    fully_entailed_spans = spans.draw(
        st.lists(fully_entailed_spans_strategy(text), min_size=2)
    )
    groups = group_overlapping_spans(
        spans=[span for pair in fully_entailed_spans for span in pair]
    )
    assert len(groups) <= len(fully_entailed_spans)


@given(text=text_strategy, spans=st.data())
def test_whether_span_merging_raises_value_error_on_non_matching_concept_id(
    text, spans
):
    span_a, span_b = spans.draw(overlapping_spans_strategy(text))
    if span_a.concept_id != span_b.concept_id:
        with pytest.raises(ValueError, match="All spans must have the same concept_id"):
            Span._validate_merge_candidates(spans=[span_a, span_b])


@given(text=text_strategy, spans=st.data())
def test_whether_span_merging_raises_value_error_on_non_matching_text(text, spans):
    span_a, span_b = spans.draw(overlapping_spans_strategy(text))
    if span_a.text != span_b.text:
        with pytest.raises(ValueError, match="All spans must have the same text"):
            Span._validate_merge_candidates(spans=[span_a, span_b])


def test_whether_span_merging_raises_value_error_with_no_spans():
    with pytest.raises(ValueError, match="Cannot merge an empty list of spans"):
        Span._validate_merge_candidates(spans=[])


@given(text=text_strategy, spans=st.data())
def test_whether_span_union_returns_span_of_correct_size(text, spans):
    span_a, span_b = spans.draw(overlapping_spans_strategy(text))
    span_b.concept_id = span_a.concept_id
    merged_span = Span.union(spans=[span_a, span_b])
    assert merged_span.start_index == min(span_a.start_index, span_b.start_index)
    assert merged_span.end_index == max(span_a.end_index, span_b.end_index)
    assert merged_span.text == text
    assert merged_span.labellers == list(set(span_a.labellers + span_b.labellers))


@given(text=text_strategy, spans=st.data())
def test_whether_span_intersection_returns_span_of_correct_size(text, spans):
    span_a, span_b = spans.draw(overlapping_spans_strategy(text))
    span_b.concept_id = span_a.concept_id
    merged_span = Span.intersection(spans=[span_a, span_b])
    assert merged_span.start_index == max(span_a.start_index, span_b.start_index)
    assert merged_span.end_index == min(span_a.end_index, span_b.end_index)
    assert merged_span.text == text
    assert merged_span.labellers == list(set(span_a.labellers + span_b.labellers))
