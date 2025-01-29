from typing import Type

import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.classifier.classifier import Classifier
from src.classifier.keyword import KeywordClassifier
from src.classifier.rules_based import RulesBasedClassifier
from src.classifier.stemmed_keyword import StemmedKeywordClassifier
from src.concept import Concept
from src.identifiers import WikibaseID, generate_identifier
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
            alphabet=st.characters(
                # https://en.wikipedia.org/wiki/Unicode_character_property#General_Category
                exclude_categories=("C", "Zl", "Zp", "P", "M", "S", "N")
            ),
        ).filter(
            lambda x: x.strip()
            and all(label.lower() not in x.lower() for label in labels)
            and all(
                not any(word in label.lower() for word in x.lower().split())
                for label in labels
            )  # Prevent extra partial matches
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
            alphabet=st.characters(
                # https://en.wikipedia.org/wiki/Unicode_character_property#General_Category
                exclude_categories=("C", "Zl", "Zp")
            ),
        ).filter(
            lambda x: x.strip()
            and x != pre_text
            and all(label.lower() not in x.lower() for label in labels)
            and all(
                not any(word in label.lower() for word in x.lower().split())
                for label in labels
            )  # Prevent extra partial matches
            and (
                negative_labels is None
                or all(
                    neg_label.lower() not in x.lower() for neg_label in negative_labels
                )
            )
        )
    )

    return f"{pre_text} {keyword} {post_text}"


classifier_classes: list[Type[Classifier]] = [
    KeywordClassifier,
    RulesBasedClassifier,
    StemmedKeywordClassifier,
]


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
        pytest.skip("This test only applies to RulesBasedClassifiers")

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
@pytest.mark.parametrize(
    "concept_data,test_text,should_match",
    [
        (
            {
                "preferred_label": "gas",
                "negative_labels": ["greenhouse gas", "gas industry"],
            },
            "I need to fill up my gas tank.",
            True,
        ),
        (
            {
                "preferred_label": "gas",
                "negative_labels": ["greenhouse gas", "gas industry"],
            },
            "Greenhouse gas emissions are a major contributor to climate change.",
            False,
        ),
        (
            {
                "preferred_label": "conflict",
                "negative_labels": ["conflict of interest"],
            },
            "The conflict in Sudan has major implications for the region.",
            True,
        ),
        (
            {
                "preferred_label": "conflict",
                "negative_labels": ["conflict of interest"],
            },
            "This conflict of interest is a major contributor to climate change.",
            False,
        ),
        # in practice, the following situations are unlikely to occur, but we should
        # check that the classifier behaves as expected anyway. These situations are
        # better suited to non-keyword-based classifiers which will respect the semantic
        # nuance of these sorts of positive and negative labels.
        (
            {
                "preferred_label": "greenhouse gas",
                "negative_labels": ["gas"],
            },
            "Greenhouse gas emissions are a major contributor to climate change.",
            False,
        ),
        (
            {
                "preferred_label": "greenhouse gas",
                "negative_labels": ["gas"],
            },
            "I need to fill up my gas tank.",
            False,
        ),
    ],
)
def test_concrete_negative_label_examples(
    classifier_class: Type[Classifier],
    concept_data: dict,
    test_text: str,
    should_match: bool,
):
    """Test specific examples of positive and negative label matching."""
    if not issubclass(classifier_class, RulesBasedClassifier):
        pytest.skip("This test only applies to RulesBasedClassifiers")

    concept = Concept(wikibase_id="Q123", **concept_data)
    classifier = classifier_class(concept)
    spans = classifier.predict(test_text)

    if should_match:
        assert spans, f"{classifier} did not match text in '{test_text}'"
        assert all(
            span.labelled_text.lower()
            in [label.lower() for label in concept.all_labels]
            for span in spans
        )
    else:
        assert not spans, f"{classifier} incorrectly matched text in '{test_text}'"


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


@pytest.mark.parametrize("classifier_class", classifier_classes)
@given(concept=concept_strategy())
def test_whether_classifier_hashes_are_generated_correctly(
    classifier_class: Type[Classifier], concept: Concept
):
    classifier = classifier_class(concept)
    assert classifier.id == generate_identifier(classifier.__hash__())
    assert classifier == classifier_class(concept)


@pytest.mark.parametrize("classifier_class", classifier_classes)
@given(concept=concept_strategy())
def test_whether_classifier_id_generation_is_affected_by_internal_state(
    classifier_class: Type[Classifier],
    concept: Concept,
):
    classifier = classifier_class(concept)

    # do some stuff with the classifier to make sure that the id remains the same
    classifier.fit()
    classifier.predict("some text")

    assert classifier.id == classifier_class(concept).id


@pytest.mark.parametrize("classifier_class", classifier_classes)
@given(concepts=st.sets(concept_strategy(), min_size=10, max_size=10))
def test_whether_different_concepts_produce_different_hashes_when_using_the_same_classifier_class(
    classifier_class: Type[Classifier], concepts: list[Concept]
):
    # classifiers of the same class, for different concepts
    classifiers = [classifier_class(concept) for concept in concepts]
    hashes = [hash(classifier) for classifier in classifiers]
    assert len(set(hashes)) == len(hashes)


@given(concept=concept_strategy())
def test_whether_different_classifier_models_produce_different_hashes_when_based_on_the_same_concept(
    concept: Concept,
):
    # classifiers of different classes, for the same concept
    classifiers = [classifier_class(concept) for classifier_class in classifier_classes]
    hashes = [hash(classifier) for classifier in classifiers]
    assert len(set(hashes)) == len(hashes)


@pytest.mark.parametrize("classifier_class", classifier_classes)
@given(concept=concept_strategy())
def test_whether_a_classifier_with_a_small_change_to_the_internal_concept_produces_a_different_id(
    classifier_class: Type[Classifier], concept: Concept
):
    classifier = classifier_class(concept)

    augmented_concept = concept.model_copy(
        update={"alternative_labels": concept.alternative_labels + ["new_label"]}
    )
    new_classifier = classifier_class(augmented_concept)
    assert classifier.id != new_classifier.id
    assert hash(classifier) != hash(new_classifier)
    assert classifier != new_classifier

    augmented_concept = concept.model_copy(update={"preferred_label": "new_label"})
    new_classifier = classifier_class(augmented_concept)
    assert classifier.id != new_classifier.id
    assert hash(classifier) != hash(new_classifier)
    assert classifier != new_classifier

    augmented_concept = concept.model_copy(
        update={"wikibase_id": concept.wikibase_id + "1"}
    )
    new_classifier = classifier_class(augmented_concept)
    assert classifier.id != new_classifier.id
    assert hash(classifier) != hash(new_classifier)
    assert classifier != new_classifier

    augmented_concept = concept.model_copy(
        update={"negative_labels": concept.negative_labels + ["new_label"]}
    )
    new_classifier = classifier_class(augmented_concept)
    assert classifier.id != new_classifier.id
    assert hash(classifier) != hash(new_classifier)
    assert classifier != new_classifier


def test_whether_a_classifier_which_does_not_specify_allowed_concept_ids_accepts_any_concept():
    class UnrestrictedClassifier(Classifier):
        def predict(self, text: str) -> list[Span]:
            return []

    concept1 = Concept(wikibase_id="Q123", preferred_label="test")
    concept2 = Concept(wikibase_id="Q456", preferred_label="test")

    assert UnrestrictedClassifier(concept1)
    assert UnrestrictedClassifier(concept2)


def test_whether_a_classifier_with_a_single_allowed_concept_id_validates_correctly():
    class SingleIDClassifier(Classifier):
        allowed_concept_ids = [WikibaseID("Q123")]

        def predict(self, text: str) -> list[Span]:
            return []

    valid_concept = Concept(wikibase_id="Q123", preferred_label="test")
    invalid_concept = Concept(wikibase_id="Q456", preferred_label="test")

    assert SingleIDClassifier(valid_concept)

    with pytest.raises(ValueError) as exc_info:
        SingleIDClassifier(invalid_concept)
    assert "must be Q123" in str(exc_info.value)
    assert "not Q456" in str(exc_info.value)


def test_whether_a_classifier_with_multiple_allowed_concept_ids_validates_correctly():
    class MultiIDClassifier(Classifier):
        allowed_concept_ids = [WikibaseID("Q123"), WikibaseID("Q456")]

        def predict(self, text: str) -> list[Span]:
            return []

    valid_concept = Concept(wikibase_id="Q123", preferred_label="test")
    invalid_concept = Concept(wikibase_id="Q789", preferred_label="test")

    assert MultiIDClassifier(valid_concept)

    with pytest.raises(ValueError) as exc_info:
        MultiIDClassifier(invalid_concept)
    assert "must be one of Q123,Q456" in str(exc_info.value)
    assert "not Q789" in str(exc_info.value)


def test_whether_allowed_concept_ids_validation_works_correctly_with_inheritance():
    class ParentClassifier(Classifier):
        allowed_concept_ids = [WikibaseID("Q123"), WikibaseID("Q456")]

        def predict(self, text: str) -> list[Span]:
            return []

    class ChildClassifier(ParentClassifier):
        allowed_concept_ids = [WikibaseID("Q123")]  # More restrictive than parent

    valid_concept = Concept(wikibase_id="Q123", preferred_label="test")
    parent_only_concept = Concept(wikibase_id="Q456", preferred_label="test")

    # Parent should accept both concepts
    assert ParentClassifier(valid_concept)
    assert ParentClassifier(parent_only_concept)

    # Child should only accept its own allowed ID
    assert ChildClassifier(valid_concept)
    with pytest.raises(ValueError) as exc_info:
        ChildClassifier(parent_only_concept)
    assert "must be Q123" in str(exc_info.value)
    assert "not Q456" in str(exc_info.value)


def test_whether_an_empty_allowed_concept_ids_list_accepts_all_concepts():
    """
    Test whether supplying an empty list of allowed_concept_ids is prohibitive.

    The expected behaviour is that the classifier should accept any concept,
    regardless of its wikibase_id.
    """

    class EmptyIDClassifier(Classifier):
        allowed_concept_ids = []

        def predict(self, text: str) -> list[Span]:
            return []

    concept = Concept(wikibase_id="Q123", preferred_label="test")

    assert EmptyIDClassifier(concept)
