from datetime import datetime, timedelta
from typing import Optional

from hypothesis import given
from hypothesis import strategies as st

from src.argilla import filter_labelled_passages_by_timestamp, is_between_timestamps
from src.labelled_passage import LabelledPassage
from src.span import Span
from tests.common_strategies import (
    labeller_strategy,
    span_strategy,
    text_strategy,
    timestamp_strategy,
)


@given(
    timestamp=timestamp_strategy(),
    before=st.one_of(timestamp_strategy(), st.none()),
    after=st.one_of(timestamp_strategy(), st.none()),
)
def test_is_between_timestamps(
    timestamp: datetime, before: Optional[datetime], after: Optional[datetime]
):
    result = is_between_timestamps(timestamp, before, after)
    if before and timestamp > before:
        assert not result
    elif after and timestamp < after:
        assert not result
    else:
        assert result


@given(
    text=text_strategy,
    before=st.one_of(timestamp_strategy(), st.none()),
    after=st.one_of(timestamp_strategy(), st.none()),
    data=st.data(),
)
def test_whether_valid_timestamps_are_retained(
    text: str,
    before: Optional[datetime],
    after: Optional[datetime],
    data: st.DataObject,
):
    spans = data.draw(st.lists(span_strategy(text), min_size=1, max_size=5))
    passage = LabelledPassage(text=text, spans=spans)
    filtered_passages = filter_labelled_passages_by_timestamp([passage], before, after)

    for filtered_passage in filtered_passages:
        for span in filtered_passage.spans:
            assert all(
                is_between_timestamps(timestamp, before, after)
                for timestamp in span.timestamps
            )
            assert len(span.timestamps) == len(span.labellers)
        assert len(filtered_passage.spans) > 0


@given(
    text=text_strategy,
    labellers=st.lists(labeller_strategy, min_size=3, max_size=3),
)
def test_whether_a_mixed_set_of_timestamps_are_filtered_correctly(
    text: str, labellers: list[str]
):
    now = datetime.now()
    before = now + timedelta(days=1)
    after = now - timedelta(days=1)

    span = Span(
        text=text,
        start_index=0,
        end_index=min(len(text), 10),
        labellers=labellers,
        timestamps=[
            now - timedelta(days=2),  # should be excluded
            now,  # should be included
            now + timedelta(days=2),  # should be excluded
        ],
    )

    passages = [LabelledPassage(text=text, spans=[span])]
    filtered = filter_labelled_passages_by_timestamp(passages, before, after)

    assert len(filtered) == 1
    assert len(filtered[0].spans) == 1
    filtered_span = filtered[0].spans[0]
    assert len(filtered_span.labellers) == 1

    # The second labeller should be retained
    assert filtered_span.labellers[0] == labellers[1]
    assert len(filtered_span.timestamps) == 1
    assert filtered_span.timestamps[0] == now


@given(text=text_strategy)
def test_whether_invalid_timestamps_are_filtered(text: str):
    # Create a window that the timestamp can't be within
    now = datetime.now()
    before = now - timedelta(days=1)
    after = now + timedelta(days=1)

    span = Span(
        text=text,
        start_index=0,
        end_index=min(len(text), 10),
        labellers=["Alice", "Bob"],
        timestamps=[now, now],
    )

    passages = [LabelledPassage(text=text, spans=[span])]
    filtered = filter_labelled_passages_by_timestamp(passages, before, after)

    # After filtering, there should be no remaining passages
    assert len(filtered) == 0
