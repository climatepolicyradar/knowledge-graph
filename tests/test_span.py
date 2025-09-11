import pytest
from hypothesis import given
from hypothesis import strategies as st

from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.span import (
    Span,
    SpanXMLConceptFormattingError,
    find_span_text_in_input_text,
    group_overlapping_spans,
    jaccard_similarity,
)
from tests.common_strategies import (
    labeller_strategy,
    span_strategy,
    text_strategy,
    wikibase_id_strategy,
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


@pytest.mark.parametrize(
    "input_text,span_start,expected_span_text",
    [
        (
            "2) Retirement benefits.",
            0,
            "Retirement benefits",
        ),  # LLM has removed the "2) "
        (
            "According to FAO (2018d), safeguards cover a variety of substantive areas",
            27,
            "safeguards",
        ),  # LLM added a space before 'safeguards'
        (
            "updating the National Energy Policy; coordinating and managing the Ministry_#39;s relations on climate change",
            13,
            "National Energy Policy",
        ),  # LLM removed the'#39;'
    ],
)
def test_find_span_text_in_input_text(input_text, span_start, expected_span_text):
    span_start_and_end = find_span_text_in_input_text(
        input_text=input_text,
        span_start_index=span_start,
        span_text=expected_span_text,
    )
    assert span_start_and_end is not None

    span_start_idx, span_end_idx = span_start_and_end
    assert input_text[span_start_idx:span_end_idx] == expected_span_text


@pytest.mark.parametrize(
    "span_xml,expected_start_and_end_idxs",
    [
        (
            "i'm a little model and i <concept>love</concept> annotating concepts",
            [(25, 29)],
        ),
        (
            "The investments projects will include a <concept>gender</concept> strategy with actions to increase <concept>women_s</concept> mobility,",
            [(40, 46), (81, 88)],
        ),
    ],
)
def test_span_from_xml_no_alignment(
    span_xml: str,
    expected_start_and_end_idxs: list[tuple[int, int]],
):
    concept_id = WikibaseID("Q1234")
    labellers = ["Arminel", "Siôn"]
    text_without_tags = span_xml.replace("<concept>", "").replace("</concept>", "")

    found_spans = Span.from_xml(
        span_xml,
        concept_id=concept_id,
        labellers=labellers,
    )

    expected_starts = [a for a, _ in expected_start_and_end_idxs]
    expected_ends = [b for _, b in expected_start_and_end_idxs]

    assert [span.start_index for span in found_spans] == expected_starts
    assert [span.end_index for span in found_spans] == expected_ends
    assert [span.text for span in found_spans] == [text_without_tags] * len(found_spans)


@pytest.mark.parametrize(
    "span_xml,expected_start_and_end_idxs,input_text",
    [
        # Simulates an LLM removing a leading bullet point
        (
            "i'm a little model and i <concept>love</concept> annotating concepts",
            [(29, 33)],
            "43. i'm a little model and i love annotating concepts",
        ),
        # Simulates an LLM removing an extra space between words
        (
            "The investments projects will include a <concept>gender</concept> strategy with actions to increase <concept>women_s</concept> mobility,",
            [(40, 46), (82, 89)],
            "The investments projects will include a gender strategy with actions to increase  women_s mobility,",
        ),
        # Trickier example where small variants of the same span exist multiple times but only one is tagged as a concept.
        # The text in the span itself has also been modified by the LLM (extra space removed)
        (
            "According to FAO (2018d), safeguards cover a variety of substantive areas in environmental and social management. While there is no agreement at an international level regarding what should be covered under a <concept>safeguard system</concept>, most safeguard systems",
            [(214, 231)],
            "According to FAO (2018d),  safeguards  cover a variety of substantive areas in  environmental and social management . While there is no agreement at an international level regarding what should be covered under a  safeguard  system, most  safeguard  systems",
        ),
    ],
)
def test_span_from_xml_with_alignment(
    span_xml: str,
    expected_start_and_end_idxs: list[tuple[int, int]],
    input_text: str,
):
    concept_id = WikibaseID("Q1234")
    labellers = ["Arminel", "Siôn"]
    text_without_tags = input_text

    found_spans = Span.from_xml(
        span_xml,
        concept_id=concept_id,
        labellers=labellers,
        input_text=input_text,
    )

    expected_starts = [a for a, _ in expected_start_and_end_idxs]
    expected_ends = [b for _, b in expected_start_and_end_idxs]

    assert [span.start_index for span in found_spans] == expected_starts
    assert [span.end_index for span in found_spans] == expected_ends
    assert [span.text for span in found_spans] == [text_without_tags] * len(found_spans)


@pytest.mark.parametrize(
    ("xml,is_valid"),
    [
        (
            "<concept>the Government plans to scale up programs to match up people with job opportunities and provide the unemployed people with <concept>unemployment insurance</concept>. <concept>Unemployment benefits</concept>, <concept>job training</concept> opportunities and <concept>job counselling</concept> will also be provided to strengthen the <concept>social safety net</concept>. Through such support, <concept>vulnerable populations</concept> to be affected by the transition will receive <concept>support</concept> for their livelihood and <concept>retraining opportunities</concept>.</concept>",
            False,
        ),
        ("<concept>hello!</concept> :)", True),
    ],
)
def test_span_from_xml_invalid_concept_annotation(xml: str, is_valid: bool):
    if is_valid:
        spans = Span.from_xml(xml, concept_id=WikibaseID("Q123"), labellers=["me"])
        assert spans

    else:
        with pytest.raises(SpanXMLConceptFormattingError):
            _ = Span.from_xml(xml, concept_id=WikibaseID("Q123"), labellers=["me"])
