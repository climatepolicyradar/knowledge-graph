import re
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


@st.composite
def multi_word_label_strategy(
    draw,
    min_words: int = 2,
    max_words: int = 4,
    separators: Optional[
        list[str]
    ] = None,  # default to space-separated labels for most tests
):
    """Generate a multi-word label, separated by the supplied separator characters."""
    num_words = draw(st.integers(min_value=min_words, max_value=max_words))
    words = [draw(single_word_label_strategy) for _ in range(num_words)]

    output_text = ""
    for word in words[:-1]:
        sep = draw(st.sampled_from(separators or [" "]))
        output_text += word + sep
    return output_text + words[-1]


concept_label_strategy = st.one_of(
    single_word_label_strategy, multi_word_label_strategy()
)


@st.composite
def concept_strategy(draw, ensure_token_disjoint: bool = True):
    """
    Generate a random Concept for testing.

    :param draw: Hypothesis strategy drawing function.
    :param ensure_token_disjoint: If True (default), ensures that no tokens from negative
                                  labels appear in positive labels. This prevents issues with
                                  KeywordClassifier where overlapping tokens cause positive
                                  matches to be filtered out. Set to False if you want to test
                                  edge cases with token overlap.
    :return: A randomly generated Concept.
    """
    preferred_label = draw(concept_label_strategy)
    alt_labels = draw(st.lists(concept_label_strategy, max_size=5))
    all_positive = alt_labels + [preferred_label]

    if ensure_token_disjoint:
        # Ensure negative labels don't share tokens with positive labels
        # This prevents KeywordClassifier from filtering out valid positive matches
        negative_labels = draw(
            st.lists(
                st.one_of(
                    single_word_label_strategy, multi_word_label_strategy()
                ).filter(
                    lambda x: _label_tokens_are_disjoint_from_labels(
                        x, all_positive, r"[\s\-–—]+"
                    )
                ),
                max_size=5,
            )
        )
    else:
        # Only ensure negative labels aren't identical to positive labels
        negative_labels = draw(
            st.lists(
                st.one_of(
                    single_word_label_strategy, multi_word_label_strategy()
                ).filter(
                    lambda x: x.lower() not in [label.lower() for label in all_positive]
                ),
                max_size=5,
            )
        )

    return Concept(
        wikibase_id=WikibaseID(draw(wikibase_id_strategy)),
        preferred_label=preferred_label,
        alternative_labels=alt_labels,
        negative_labels=negative_labels,
    )


def _label_tokens_are_disjoint_from_labels(
    label: str, other_labels: list[str], separator_pattern: str
) -> bool:
    """
    Check if a label's tokens are disjoint from all tokens in other labels.

    :param label: The label to check.
    :param other_labels: List of labels to compare against.
    :param separator_pattern: Regex pattern to split labels into tokens.
    :return: True if no tokens from label appear in any of the other_labels.
    """
    # First check if the label is identical to any other label
    if label.lower() in [other.lower() for other in other_labels]:
        return False

    # Then check if any tokens overlap
    label_tokens = {tok.lower() for tok in re.split(separator_pattern, label) if tok}
    other_tokens = {
        tok.lower()
        for other_label in other_labels
        for tok in re.split(separator_pattern, other_label)
        if tok
    }
    return label_tokens.isdisjoint(other_tokens)


@st.composite
def negative_text_strategy(draw, labels: list[str]):
    """
    Generate text which does not contain any of the concept's labels.

    :param draw: Hypothesis strategy drawing function.
    :param labels: Plain string labels (NOT regex patterns).
                   These should be the original concept labels before any transformation.
    """
    return draw(
        st.text(min_size=1, max_size=1000).filter(
            lambda x: all(label.lower() not in x.lower() for label in labels)
        )
    )


@st.composite
def positive_text_strategy(
    draw,
    labels: Optional[list[str]] = None,
    negative_labels: Optional[list[str]] = None,
):
    """
    Generate text containing one of the labels, with pre/post text that avoids negatives.

    :param draw: Hypothesis strategy drawing function.
    :param labels: Plain string labels (NOT regex patterns) to include in the generated text.
    :param negative_labels: Plain string labels (NOT regex patterns) to avoid in the text.
    """
    labels = labels or []
    negative_labels = negative_labels or []

    keyword = draw(st.sampled_from(labels))

    pre_text = draw(
        st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(
                # https://en.wikipedia.org/wiki/Unicode_character_property#General_Category
                exclude_categories=("C", "Zl", "Zp", "P", "M", "S", "N"),
            ),
        ).filter(
            lambda x: x.strip()
            and all(label.lower() not in x.lower() for label in labels)
            and all(
                not any(word in label.lower() for word in x.lower().split())
                for label in labels
            )
            and all(neg_label.lower() not in x.lower() for neg_label in negative_labels)
        )
    )

    post_text = draw(
        st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(
                exclude_categories=("C", "Zl", "Zp"),
            ),
        ).filter(
            lambda x: x.strip()
            and x != pre_text
            and all(label.lower() not in x.lower() for label in labels)
            and all(
                not any(word in label.lower() for word in x.lower().split())
                for label in labels
            )
            and all(neg_label.lower() not in x.lower() for neg_label in negative_labels)
        )
    )

    return f"{pre_text} {keyword} {post_text}"


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
