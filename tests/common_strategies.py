from datetime import datetime
from typing import Optional

from hypothesis import strategies as st

from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.span import Span

wikibase_id_strategy = st.from_regex(WikibaseID.regex, fullmatch=True)
text_strategy = st.text(min_size=10, max_size=1000)
labeller_strategy = st.text(min_size=1, max_size=10)

# generates one word without spaces, hyphens, or punctuation
single_word_label_strategy = (
    st.text(
        min_size=1,
        max_size=12,
        alphabet=st.characters(
            exclude_categories=(
                # https://en.wikipedia.org/wiki/Unicode_character_property#General_Category
                "C",  # Control characters
                "Z",  # Separators
                "P",  # Punctuation
                "M",  # Marks
                "S",  # Symbols
                "N",  # Numbers
            ),
        ),
    )
    .map(lambda x: x.strip())
    .filter(lambda x: x)
)

more_complex_separator_characters = [" ", "-", "\n", "\t"]


@st.composite
def multi_word_label_strategy(
    draw,
    min_words: int = 2,
    max_words: int = 4,
    separators: list[str] = [" "],  # default to space-separated labels for most tests
):
    """Generate a multi-word label, separated by the supplied separator characters"""
    num_words = draw(st.integers(min_value=min_words, max_value=max_words))
    words = [draw(single_word_label_strategy) for _ in range(num_words)]

    output_text = ""
    for word in words[:-1]:
        sep = draw(st.sampled_from(separators))
        output_text += word + sep
    return output_text + words[-1]


concept_label_strategy = st.one_of(
    single_word_label_strategy, multi_word_label_strategy()
)


@st.composite
def concept_strategy(draw):
    preferred_label = draw(concept_label_strategy)
    alt_labels = draw(st.lists(concept_label_strategy, max_size=5))
    # negative_labels cannot overlap with the positive labels
    all_positive = alt_labels + [preferred_label]
    negative_labels = draw(
        st.lists(
            st.one_of(single_word_label_strategy, multi_word_label_strategy()).filter(
                lambda x: x.lower() not in [label.lower() for label in all_positive]
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
    generated_text: str = text if text is not None else draw(text_strategy)
    start_index = draw(st.integers(min_value=0, max_value=len(generated_text) - 1))
    end_index = draw(
        st.integers(min_value=start_index + 1, max_value=len(generated_text))
    )
    concept_id = draw(st.one_of(wikibase_id_strategy, st.none()))
    num_items = draw(st.integers(min_value=1, max_value=3))
    labellers = draw(
        st.lists(labeller_strategy, min_size=num_items, max_size=num_items)
    )
    timestamps = draw(
        st.lists(timestamp_strategy(), min_size=num_items, max_size=num_items)
    )
    return Span(
        text=generated_text,
        start_index=start_index,
        end_index=end_index,
        concept_id=concept_id,
        labellers=labellers,
        timestamps=timestamps,
    )
