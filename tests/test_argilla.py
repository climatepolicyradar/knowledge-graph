from datetime import datetime, timedelta
from typing import Optional

from hypothesis import given
from hypothesis import strategies as st

from src.argilla_legacy import (
    filter_labelled_passages_by_timestamp,
    is_between_timestamps,
)
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
    max_timestamp=st.one_of(timestamp_strategy(), st.none()),
    min_timestamp=st.one_of(timestamp_strategy(), st.none()),
)
def test_whether_timestamp_filtering_works(
    timestamp: datetime,
    max_timestamp: Optional[datetime],
    min_timestamp: Optional[datetime],
):
    result = is_between_timestamps(timestamp, min_timestamp, max_timestamp)
    if max_timestamp and timestamp > max_timestamp:
        assert not result
    elif min_timestamp and timestamp < min_timestamp:
        assert not result
    else:
        assert result


@given(
    text=text_strategy,
    max_timestamp=st.one_of(timestamp_strategy(), st.none()),
    min_timestamp=st.one_of(timestamp_strategy(), st.none()),
    data=st.data(),
)
def test_whether_valid_timestamps_are_retained(
    text: str,
    max_timestamp: Optional[datetime],
    min_timestamp: Optional[datetime],
    data: st.DataObject,
):
    spans = data.draw(st.lists(span_strategy(text), min_size=1, max_size=5))
    passage = LabelledPassage(text=text, spans=spans)
    filtered_passages = filter_labelled_passages_by_timestamp(
        [passage], min_timestamp, max_timestamp
    )

    for filtered_passage in filtered_passages:
        for span in filtered_passage.spans:
            assert all(
                is_between_timestamps(timestamp, min_timestamp, max_timestamp)
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
    max_timestamp = now + timedelta(days=1)
    min_timestamp = now - timedelta(days=1)

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
        concept_id=None,
    )

    passages = [LabelledPassage(text=text, spans=[span])]
    filtered = filter_labelled_passages_by_timestamp(
        passages, min_timestamp, max_timestamp
    )

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
    min_timestamp = now + timedelta(days=1)
    max_timestamp = now - timedelta(days=1)

    span = Span(
        text=text,
        start_index=0,
        end_index=min(len(text), 10),
        labellers=["Alice", "Bob"],
        timestamps=[now, now],
        concept_id=None,
    )

    passages = [LabelledPassage(text=text, spans=[span])]
    filtered = filter_labelled_passages_by_timestamp(
        passages, min_timestamp, max_timestamp
    )

    # After filtering, there should be no remaining passages
    assert len(filtered) == 0
