import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.labelled_passage import LabelledPassage
from src.span import Span

text_strategy = st.text(min_size=1, max_size=1000)


@st.composite
def span_strategy(draw, text):
    start = draw(st.integers(min_value=0, max_value=len(text) - 1))
    end = draw(st.integers(min_value=start + 1, max_value=len(text)))
    concept_id = draw(st.text(min_size=1, max_size=10))
    labellers = draw(st.lists(st.text(min_size=1, max_size=10), min_size=1, max_size=3))
    return Span(
        text=text,
        start_index=start,
        end_index=end,
        concept_id=concept_id,
        labellers=labellers,
    )


@given(text=text_strategy, spans=st.data())
def test_whether_labelled_passage_is_correctly_initialised(text, spans):
    span_list = spans.draw(st.lists(span_strategy(text), min_size=0, max_size=10))
    passage = LabelledPassage(text=text, spans=span_list)
    assert passage.text == text
    assert passage.spans == span_list
    assert isinstance(passage.id, str)
    assert isinstance(passage.metadata, dict)


def test_whether_labelled_passage_raises_validation_error_on_short_text_and_long_span():
    with pytest.raises(ValueError):
        LabelledPassage(
            text="Short text",
            spans=[
                Span(
                    text="Short text",
                    start_index=0,
                    end_index=1000,
                    concept_id="test",
                    labellers=["user1"],
                )
            ],
        )


@given(text=text_strategy, spans=st.data())
def test_whether_identifier_generation_is_deterministic(text, spans):
    span_list = spans.draw(st.lists(span_strategy(text), min_size=1, max_size=5))
    passage = LabelledPassage(text=text, spans=span_list)
    assert passage.id == LabelledPassage(text=text, spans=span_list).id


@given(text=text_strategy, spans_a=st.data(), spans_b=st.data())
def test_whether_identifier_generation_is_dependent_on_input(text, spans_a, spans_b):
    text_a = text
    text_b = text + " "
    span_list_a = spans_a.draw(st.lists(span_strategy(text_a), min_size=1, max_size=5))
    span_list_b = spans_b.draw(st.lists(span_strategy(text_b), min_size=1, max_size=5))
    assert (
        LabelledPassage(text=text_a, spans=span_list_a).id
        != LabelledPassage(text=text_b, spans=span_list_b).id
    )


@given(text=text_strategy, spans=st.data())
def test_whether_labeller_set_is_correctly_generated(text, spans):
    span_list = spans.draw(st.lists(span_strategy(text), min_size=1, max_size=5))
    passage = LabelledPassage(text=text, spans=span_list)
    expected_labellers = set(
        [labeller for span in span_list for labeller in span.labellers]
    )
    assert set(passage.labellers) == expected_labellers


def test_whether_highlighted_text_is_correctly_generated():
    passage = LabelledPassage(
        text="This is a test passage.",
        spans=[
            Span(
                text="This is a test passage.",
                start_index=0,
                end_index=4,
            )
        ],
    )
    highlighted = passage.get_highlighted_text()
    assert highlighted == "[cyan]This[/cyan] is a test passage."


def test_whether_highlighted_text_is_correctly_generated_with_multiple_spans():
    passage = LabelledPassage(
        text="This is a test passage.",
        spans=[
            Span(
                text="This is a test passage.",
                start_index=0,
                end_index=4,
            ),
            Span(
                text="This is a test passage.",
                start_index=5,
                end_index=7,
            ),
        ],
    )
    highlighted = passage.get_highlighted_text()
    assert highlighted == "[cyan]This[/cyan] [cyan]is[/cyan] a test passage."


def test_whether_highlighted_text_is_correctly_generated_with_overlapping_spans():
    passage = LabelledPassage(
        text="This is a test passage.",
        spans=[
            Span(
                text="This is a test passage.",
                start_index=0,
                end_index=7,
            ),
            Span(
                text="This is a test passage.",
                start_index=5,
                end_index=9,
            ),
        ],
    )
    highlighted = passage.get_highlighted_text()
    assert highlighted == "[cyan]This is a[/cyan] test passage."


def test_whether_highlighted_text_is_correctly_generated_with_alternative_format():
    passage = LabelledPassage(
        text="This is a test passage.",
        spans=[
            Span(
                text="This is a test passage.",
                start_index=0,
                end_index=4,
            )
        ],
    )
    highlighted = passage.get_highlighted_text(format="red")
    assert highlighted == "[red]This[/red] is a test passage."


def test_whether_highlighted_text_correctly_handles_encoded_html():
    passage = LabelledPassage(
        text="This &amp; that",
        spans=[
            Span(
                text="This &amp; that",
                start_index=0,
                end_index=4,
                concept_id="TEST",
                labellers=["user1"],
            )
        ],
    )
    highlighted = passage.get_highlighted_text()
    assert highlighted == "[cyan]This[/cyan] & that"
