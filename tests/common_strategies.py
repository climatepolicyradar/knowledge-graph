from typing import Optional

from hypothesis import strategies as st

from src.concept import Concept
from src.identifiers import WikibaseID
from src.span import Span

wikibase_id_strategy = st.from_regex(WikibaseID.regex, fullmatch=True)
text_strategy = st.text(min_size=10, max_size=1000)
labeller_strategy = st.text(min_size=1, max_size=10)
concept_label_strategy = st.text(
    min_size=1,
    max_size=25,
    alphabet=st.characters(exclude_categories=("C", "Zl", "Zp", "P", "M", "S")),
).map(lambda x: x.strip()).filter(lambda x: x)


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
def span_strategy(draw, text: Optional[str] = None):
    if text is None:
        text = draw(text_strategy)
    start_index = draw(st.integers(min_value=0, max_value=len(text) - 1))
    end_index = draw(st.integers(min_value=start_index + 1, max_value=len(text)))
    concept_id = draw(st.one_of(wikibase_id_strategy, st.none()))
    labellers = draw(st.lists(labeller_strategy, min_size=1, max_size=3))
    return Span(
        text=text,
        start_index=start_index,
        end_index=end_index,
        concept_id=concept_id,
        labellers=labellers,
    )
