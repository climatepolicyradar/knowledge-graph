from typing import Type

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.classifier.classifier import Classifier
from src.classifier.keyword import KeywordClassifier
from src.classifier.rules_based import RulesBasedClassifier
from src.concept import Concept
from src.span import Span
from tests.common_strategies import concept_label_strategy, concept_strategy


@st.composite
def negative_text_strategy(draw, labels: list[str]):
    """Generate text which does not contain the any of the concept's labels."""
    return draw(
        st.text(min_size=1, max_size=1000).filter(
            lambda x: all(label.lower() not in x.lower() for label in labels)
        )
    )


@st.composite
def positive_text_strategy(
    draw: st.DataObject, labels: list[str] = [], negative_labels: list[str] = []
):
    """Generate text containing one of the labels, with different before and after text that doesn't match negative labels."""
    keyword = draw(st.sampled_from(labels))

    pre_text = draw(
        st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(exclude_categories=("C", "Zl", "Zp")),
        ).filter(
            lambda x: x.strip()
            and all(label.lower() not in x.lower() for label in labels)
            and (
                negative_labels is None
                or all(
                    neg_label.lower() not in x.lower() for neg_label in negative_labels
                )
            )
        )
    )
    post_text = draw(
        st.text(
            min_size=1,
            max_size=50,
            alphabet=st.characters(exclude_categories=("C", "Zl", "Zp")),
        ).filter(
            lambda x: x.strip()
            and x != pre_text
            and all(label.lower() not in x.lower() for label in labels)
            and (
                negative_labels is None
                or all(
                    neg_label.lower() not in x.lower() for neg_label in negative_labels
                )
            )
        )
    )

    return f"{pre_text} {keyword} {post_text}"


classifier_classes = [KeywordClassifier, RulesBasedClassifier]


@pytest.mark.parametrize("classifier_class", classifier_classes)
@given(concept=concept_strategy(), text=st.data())
def test_whether_classifier_matches_concept_labels_in_text(
    classifier_class: Type[Classifier], concept: Concept, text
):
    text = text.draw(positive_text_strategy(labels=concept.all_labels))
    classifier = classifier_class(concept)
    spans = classifier.predict(text)

    assert spans, f"{classifier} did not match text in '{text}'"
    assert all(
        span.labelled_text.lower() in [label.lower() for label in concept.all_labels]
        for span in spans
    ), f"{classifier} matched incorrect text in '{text}'"


@pytest.mark.parametrize("classifier_class", classifier_classes)
@given(concept=concept_strategy(), text=st.data())
def test_whether_classifier_finds_no_spans_in_negative_text(
    classifier_class: Type[Classifier], concept: Concept, text
):
    text = text.draw(negative_text_strategy(labels=concept.all_labels))
    classifier = classifier_class(concept)
    spans = classifier.predict(text)

    assert not spans, f"{classifier} matched text in '{text}'"


@pytest.mark.parametrize("classifier_class", classifier_classes)
@given(data=st.data())
def test_whether_classifier_respects_negative_labels(
    classifier_class: Type[Classifier], data: st.DataObject
):
    if not issubclass(classifier_class, RulesBasedClassifier):
        pytest.skip("This test only applies to RulesBasedClassifier")

    # Create a positive label and a negative which contains the positive label.
    positive_label = data.draw(concept_label_strategy)
    negative_label = positive_label + " a_modifier_which_changes_its_meaning"

    # create a text containing the negative label but not the positive label
    text = data.draw(
        positive_text_strategy(
            labels=[negative_label], negative_labels=[positive_label]
        )
    )

    concept = Concept(
        wikibase_id="Q1",
        preferred_label=positive_label,
        negative_labels=[negative_label],
    )
    classifier = classifier_class(concept)

    # The classifier should not match the text
    spans = classifier.predict(text)

    assert not spans, f"{classifier} matched text in '{text}'"


@pytest.mark.parametrize("classifier_class", classifier_classes)
@given(concept=concept_strategy(), text=st.data())
def test_whether_returned_spans_are_valid(
    classifier_class: Type[Classifier], concept: Concept, text
):
    text = text.draw(positive_text_strategy(labels=concept.all_labels))
    classifier = classifier_class(concept)
    spans = classifier.predict(text)

    for span in spans:
        assert isinstance(span, Span)
        assert 0 <= span.start_index < span.end_index <= len(text)
        assert span.labelled_text == text[span.start_index : span.end_index]
        assert span.concept_id == concept.wikibase_id


@pytest.mark.parametrize("classifier_class", classifier_classes)
@given(concept=concept_strategy())
def test_whether_classifier_repr_is_correct(
    classifier_class: Type[Classifier], concept: Concept
):
    classifier = classifier_class(concept)
    assert (
        repr(classifier) == f'{classifier_class.__name__}("{concept.preferred_label}")'
    )
