import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import (
    LabelledPassage,
    dataframe_to_labelled_passages,
    labelled_passages_to_dataframe,
)
from knowledge_graph.span import Span
from tests.common_strategies import span_strategy, text_strategy


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
    highlighted = passage.get_highlighted_text(
        start_pattern="[red]", end_pattern="[/red]"
    )
    assert highlighted == "[red]This[/red] is a test passage."


def test_whether_highlighted_text_correctly_handles_encoded_html():
    passage = LabelledPassage(
        text="This &amp; that",
        spans=[
            Span(
                text="This &amp; that",
                start_index=0,
                end_index=4,
                labellers=["user1"],
            )
        ],
    )
    highlighted = passage.get_highlighted_text()
    assert highlighted == "[cyan]This[/cyan] & that"


def test_whether_highlighted_text_correctly_handles_html_tags():
    passage = LabelledPassage(
        text="This is a <span>test</span> passage.",
        spans=[
            Span(
                text="This is a <span>test</span> passage.",
                start_index=15,
                end_index=22,
                labellers=["user1"],
            )
        ],
    )
    highlighted = passage.get_highlighted_text()
    assert highlighted == "This is a test [cyan]passage[/cyan]."


@pytest.fixture
def labelled_passages_with_probabilities():
    """
    Fixture providing labelled passages with prediction probabilities.

    :return list[LabelledPassage]: List of labelled passages with spans containing prediction probabilities
    """
    return [
        LabelledPassage(
            text="Climate change is a global issue.",
            spans=[
                Span(
                    text="Climate change is a global issue.",
                    start_index=0,
                    end_index=14,
                    concept_id="Q123",
                    labellers=["user1"],
                    prediction_probability=0.95,
                ),
                Span(
                    text="Climate change is a global issue.",
                    start_index=20,
                    end_index=26,
                    concept_id="Q456",
                    labellers=["user1"],
                    prediction_probability=0.87,
                ),
            ],
            metadata={"source": "test_doc_1", "dataset": "train"},
        ),
        LabelledPassage(
            text="Renewable energy is important.",
            spans=[
                Span(
                    text="Renewable energy is important.",
                    start_index=0,
                    end_index=16,
                    concept_id="Q789",
                    labellers=["user2"],
                    prediction_probability=0.92,
                )
            ],
            metadata={"source": "test_doc_2", "dataset": "train"},
        ),
        LabelledPassage(
            text="This passage has no spans.",
            spans=[],
            metadata={"source": "test_doc_3", "dataset": "test"},
        ),
    ]


@pytest.fixture
def labelled_passages_without_probabilities():
    """
    Fixture providing labelled passages without prediction probabilities.

    :return list[LabelledPassage]: List of labelled passages with spans that don't have prediction probabilities
    """
    return [
        LabelledPassage(
            text="Climate change is a global issue.",
            spans=[
                Span(
                    text="Climate change is a global issue.",
                    start_index=0,
                    end_index=14,
                    concept_id="Q123",
                    labellers=["user1"],
                ),
                Span(
                    text="Climate change is a global issue.",
                    start_index=20,
                    end_index=26,
                    concept_id="Q456",
                    labellers=["user1"],
                ),
            ],
            metadata={"source": "test_doc_1", "dataset": "train"},
        ),
        LabelledPassage(
            text="Renewable energy is important.",
            spans=[
                Span(
                    text="Renewable energy is important.",
                    start_index=0,
                    end_index=16,
                    concept_id="Q789",
                    labellers=["user2"],
                )
            ],
            metadata={"source": "test_doc_2", "dataset": "train"},
        ),
    ]


def test_labelled_passages_to_dataframe_with_probabilities(
    labelled_passages_with_probabilities,
):
    """
    Test that labelled_passages_to_dataframe correctly converts passages with prediction probabilities.

    :param list[LabelledPassage] labelled_passages_with_probabilities: Fixture with test data
    """
    df = labelled_passages_to_dataframe(labelled_passages_with_probabilities)

    assert len(df) == len(labelled_passages_with_probabilities)
    assert "id" in df.columns
    assert "text" in df.columns
    assert "spans" in df.columns
    assert "prediction" in df.columns
    assert "prediction_probability" in df.columns

    assert df["prediction"].tolist() == [True, True, False]
    assert df["prediction_probability"].tolist() == [0.95, 0.92, 0]

    # Check that spans are not exploded (they should be lists/dicts)
    assert isinstance(df["spans"].iloc[0], list)


def test_labelled_passages_to_dataframe_without_probabilities(
    labelled_passages_without_probabilities,
):
    """
    Test that labelled_passages_to_dataframe correctly handles passages without prediction probabilities.

    :param list[LabelledPassage] labelled_passages_without_probabilities: Fixture with test data
    """
    df = labelled_passages_to_dataframe(labelled_passages_without_probabilities)

    assert len(df) == len(labelled_passages_without_probabilities)
    assert df["prediction"].tolist() == [True, True]
    assert df["prediction_probability"].tolist() == [None, None]


def test_dataframe_to_labelled_passages_without_human_labels():
    """Test that passages created without human labels have no spans."""
    df = pd.DataFrame(
        {
            "id": ["abc", "def"],
            "text": ["Climate change is severe", "Another passage"],
            "metadata.source": ["doc1", "doc2"],
        }
    )

    passages = dataframe_to_labelled_passages(df, concept_id=WikibaseID("Q123"))

    assert len(passages) == 2
    assert len(passages[0].spans) == 0
    assert len(passages[1].spans) == 0
    assert passages[0].metadata == {"source": "doc1"}
    assert passages[1].metadata == {"source": "doc2"}


def test_dataframe_to_labelled_passages_with_human_label_boolean():
    """Test dataframe_to_labelled_passages with boolean human label column."""
    df = pd.DataFrame(
        {
            "id": ["abc", "def"],
            "text": ["Climate change is severe", "Another passage"],
            "spans": [[], []],
            "human_label": [True, False],
        }
    )

    passages = dataframe_to_labelled_passages(
        df,
        human_label_column="human_label",
        labeller_names=["siôn"],
        concept_id=WikibaseID("Q42"),
    )

    assert len(passages[0].spans) == 1
    assert passages[0].spans[0].labellers == ["siôn"]
    assert passages[0].spans[0].concept_id == "Q42"
    assert passages[0].spans[0].start_index == 0
    assert passages[0].spans[0].end_index == len("Climate change is severe")

    assert len(passages[1].spans) == 0


def test_dataframe_to_labelled_passages_with_human_label_string():
    """Test dataframe_to_labelled_passages with string human label column."""
    df = pd.DataFrame(
        {
            "id": ["abc", "def", "ghi", "jkl", "mno", "pqr"],
            "text": ["Test 1", "Test 2", "Test 3", "Test 4", "Test 5", "Test 6"],
            "spans": [[], [], [], [], [], []],
            "human_label": ["yes", "no", "Y", "", "N", "false"],
        }
    )

    passages = dataframe_to_labelled_passages(
        df,
        human_label_column="human_label",
        labeller_names=["anne"],
        concept_id=WikibaseID("Q123"),
        skip_unlabelled=False,
    )

    # "yes"
    assert len(passages[0].spans) == 1
    assert passages[0].spans[0].labellers == ["anne"]

    # "no"
    assert len(passages[1].spans) == 0

    # "Y"
    assert len(passages[2].spans) == 1

    # Empty string
    assert len(passages[3].spans) == 0

    # "N"
    assert len(passages[4].spans) == 0

    # "false"
    assert len(passages[5].spans) == 0


def test_dataframe_to_labelled_passages_metadata_reconstruction():
    """Test that metadata is correctly reconstructed from flattened columns."""
    df = pd.DataFrame(
        {
            "id": ["abc"],
            "text": ["Test text"],
            "spans": [[]],
            "metadata.source": ["doc1"],
            "metadata.dataset": ["train"],
            "metadata.page": [42],
        }
    )

    passages = dataframe_to_labelled_passages(df, concept_id=WikibaseID("Q123"))

    assert len(passages) == 1
    assert passages[0].metadata == {
        "source": "doc1",
        "dataset": "train",
        "page": 42,
    }


def test_dataframe_to_labelled_passages_skip_unlabelled():
    """Test that skip_unlabelled parameter filters out unlabelled passages."""
    df = pd.DataFrame(
        {
            "id": ["abc", "def", "ghi", "jkl"],
            "text": ["Test 1", "Test 2", "Test 3", "Test 4"],
            "human_label": [True, False, None, ""],  # True, False, None, None
        }
    )

    # With skip_unlabelled=False (default), all passages returned
    passages = dataframe_to_labelled_passages(
        df,
        human_label_column="human_label",
        skip_unlabelled=False,
        concept_id=WikibaseID("Q123"),
    )
    assert len(passages) == 4

    # With skip_unlabelled=True, only labelled passages returned
    passages = dataframe_to_labelled_passages(
        df,
        human_label_column="human_label",
        skip_unlabelled=True,
        concept_id=WikibaseID("Q123"),
    )
    assert len(passages) == 2
    assert passages[0].text == "Test 1"
    assert passages[1].text == "Test 2"


def test_labelled_passages_roundtrip_conversion():
    """
    Test roundtrip conversion: LabelledPassages -> DataFrame -> LabelledPassages.

    Note: Original span details (start/end indices, concept IDs) are not preserved
    in the roundtrip because dataframe_to_labelled_passages creates new full-text
    spans based on the label column.
    """

    original_passages = [
        # Positive example with multiple spans
        LabelledPassage(
            text="Climate change is a global crisis.",
            spans=[
                Span(
                    text="Climate change is a global crisis.",
                    start_index=0,
                    end_index=14,
                    concept_id=WikibaseID("Q123"),
                    labellers=["original_labeller"],
                )
            ],
            metadata={"source": "doc1", "dataset": "train"},
        ),
        # Negative example (no spans)
        LabelledPassage(
            text="This is not relevant.",
            spans=[],
            metadata={"source": "doc2", "dataset": "test"},
        ),
        # Another positive example
        LabelledPassage(
            text="Renewable energy solutions.",
            spans=[
                Span(
                    text="Renewable energy solutions.",
                    start_index=0,
                    end_index=16,
                    concept_id=WikibaseID("Q456"),
                    labellers=["original_labeller"],
                )
            ],
            metadata={"source": "doc3", "dataset": "train"},
        ),
    ]

    df = labelled_passages_to_dataframe(original_passages)

    roundtrip_passages = dataframe_to_labelled_passages(
        df,
        human_label_column="prediction",
        labeller_names=["roundtrip_labeller"],
        concept_id=WikibaseID("Q999"),
        skip_unlabelled=False,
    )

    assert len(roundtrip_passages) == len(original_passages)

    assert roundtrip_passages[0].text == original_passages[0].text
    assert len(roundtrip_passages[0].spans) == 1  # Should have a span (positive label)
    assert roundtrip_passages[0].spans[0].concept_id == "Q999"
    assert roundtrip_passages[0].spans[0].labellers == ["roundtrip_labeller"]
    assert roundtrip_passages[0].metadata == original_passages[0].metadata

    assert roundtrip_passages[1].text == original_passages[1].text
    assert (
        len(roundtrip_passages[1].spans) == 0
    )  # Should have no spans (negative label)
    assert roundtrip_passages[1].metadata == original_passages[1].metadata

    assert roundtrip_passages[2].text == original_passages[2].text
    assert len(roundtrip_passages[2].spans) == 1  # Should have a span (positive label)
    assert roundtrip_passages[2].metadata == original_passages[2].metadata
