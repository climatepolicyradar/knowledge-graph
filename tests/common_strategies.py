from datetime import datetime
from typing import Optional

from hypothesis import strategies as st

from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.span import Span

wikibase_id_strategy = st.from_regex(WikibaseID.regex, fullmatch=True)
text_strategy = st.text(min_size=10, max_size=1000)
labeller_strategy = st.text(min_size=1, max_size=10)
concept_label_strategy = (
    st.text(
        min_size=1,
        max_size=25,
        alphabet=st.characters(
            # https://en.wikipedia.org/wiki/Unicode_character_property#General_Category
            exclude_categories=("C", "Zl", "Zp", "P", "M", "S", "N")
        ),
    )
    .map(lambda x: x.strip())
    .filter(lambda x: x)
)


@st.composite
def concept_strategy(draw):
    preferred_label = draw(concept_label_strategy)
    alt_labels = draw(st.lists(concept_label_strategy, max_size=5))
    # negative_labels cannot overlap with the positive labels
    negative_labels = draw(
        st.lists(
            concept_label_strategy.filter(
                lambda x: x.lower()
                not in [label.lower() for label in alt_labels + [preferred_label]]
            ),
            max_size=5,
        )
    )

    return Concept(
        wikibase_id=draw(wikibase_id_strategy),
        preferred_label=preferred_label,
        alternative_labels=alt_labels,
        negative_labels=negative_labels,
    )


@st.composite
def timestamp_strategy(draw) -> datetime:
    """Strategy to generate timestamps within a reasonable range."""
    return draw(
        st.datetimes(
            min_value=datetime(2020, 1, 1),
            max_value=datetime(2025, 12, 31),
        )
    )


@st.composite
def span_strategy(draw, text: Optional[str] = None):
    if text is None:
        text = draw(text_strategy)
    start_index = draw(st.integers(min_value=0, max_value=len(text) - 1))
    end_index = draw(st.integers(min_value=start_index + 1, max_value=len(text)))
    concept_id = draw(st.one_of(wikibase_id_strategy, st.none()))
    num_items = draw(st.integers(min_value=1, max_value=3))
    labellers = draw(
        st.lists(labeller_strategy, min_size=num_items, max_size=num_items)
    )
    timestamps = draw(
        st.one_of(
            st.just([]),
            st.lists(timestamp_strategy(), min_size=num_items, max_size=num_items),
        )
    )
    return Span(
        text=text,
        start_index=start_index,
        end_index=end_index,
        concept_id=concept_id,
        labellers=labellers,
        timestamps=timestamps,
    )
