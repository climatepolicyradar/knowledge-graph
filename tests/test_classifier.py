import re
from typing import Type

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from knowledge_graph.classifier.classifier import Classifier
from knowledge_graph.classifier.keyword import KeywordClassifier
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import ClassifierID, WikibaseID
from knowledge_graph.span import Span
from tests.common_strategies import (
    concept_label_strategy,
    concept_strategy,
    multi_word_label_strategy,
    negative_text_strategy,
    positive_text_strategy,
    single_word_label_strategy,
)

classifier_classes: list[Type[Classifier]] = [
    KeywordClassifier,
]


@given(concept=concept_strategy(), text_data=st.data())
@settings(max_examples=100, database=None)
@pytest.mark.xdist_group(name="classifier")
@pytest.mark.parametrize("classifier_class", classifier_classes)
def test_whether_classifier_matches_concept_labels_in_text(
    classifier_class: Type[Classifier], concept: Concept, text_data: st.DataObject
):
    # Skip concepts where any negative label token equals any token from a positive label.
    # In those cases, a positive match would be correctly filtered by the classifier
    # due to the overlapping negative label.
    sep = KeywordClassifier.separator_pattern
    positive_tokens = {
        tok.lower()
        for label in concept.all_labels
        for tok in re.split(sep, label)
        if tok
    }
    negative_tokens = {
        tok.lower()
        for label in concept.negative_labels
        for tok in re.split(sep, label)
        if tok
    }
    assume(positive_tokens.isdisjoint(negative_tokens))

    # Ensure the generated positive text does not accidentally include a negative label
    # (e.g. by appending extra tokens after the positive label that complete a negative label).
    text = text_data.draw(
        positive_text_strategy(
            labels=concept.all_labels, negative_labels=concept.negative_labels
        )
    )
    classifier = classifier_class(concept)
    spans = classifier.predict(text)

    assert spans, f"{classifier} did not match text in '{text}'"
    assert all(
        span.labelled_text.lower() in [label.lower() for label in concept.all_labels]
        for span in spans
    ), f"{classifier} matched incorrect text in '{text}'"


@pytest.mark.xdist_group(name="classifier")
@pytest.mark.parametrize("classifier_class", classifier_classes)
@given(concept=concept_strategy(), data=st.data())
@settings(max_examples=100, database=None)
def test_whether_classifier_finds_no_spans_in_negative_text(
    classifier_class: Type[Classifier], concept: Concept, data
):
    text = data.draw(negative_text_strategy(labels=concept.all_labels))
    classifier = classifier_class(concept)
    spans = classifier.predict(text)

    assert not spans, f"{classifier} matched text in '{text}'"


@pytest.mark.xdist_group(name="classifier")
@pytest.mark.parametrize("classifier_class", classifier_classes)
@given(data=st.data())
@settings(max_examples=100, database=None)
def test_whether_classifier_respects_negative_labels(
    classifier_class: Type[Classifier], data: st.DataObject
):
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
        wikibase_id=WikibaseID("Q123"),
        preferred_label=positive_label,
        negative_labels=[negative_label],
    )
    classifier = classifier_class(concept)

    # The classifier should not match the text
    spans = classifier.predict(text)

    assert not spans, f"{classifier} matched text in '{text}'"


@pytest.mark.xdist_group(name="classifier")
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
    concept = Concept(wikibase_id=WikibaseID("Q123"), **concept_data)
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


@pytest.mark.xdist_group(name="classifier")
@pytest.mark.parametrize("classifier_class", classifier_classes)
@given(concept=concept_strategy(), data=st.data())
@settings(max_examples=100, database=None)
def test_whether_returned_spans_are_valid(
    classifier_class: Type[Classifier], concept: Concept, data
):
    text = data.draw(positive_text_strategy(labels=concept.all_labels))
    classifier = classifier_class(concept)
    spans = classifier.predict(text)

    for span in spans:
        assert isinstance(span, Span)
        assert 0 <= span.start_index < span.end_index <= len(text)
        assert span.labelled_text == text[span.start_index : span.end_index]
        assert span.concept_id == concept.wikibase_id


@pytest.mark.xdist_group(name="classifier")
@pytest.mark.parametrize("classifier_class", classifier_classes)
@given(concept=concept_strategy())
@settings(max_examples=100, database=None)
def test_whether_classifier_repr_is_correct(
    classifier_class: Type[Classifier], concept: Concept
):
    classifier = classifier_class(concept)
    assert (
        repr(classifier) == f'{classifier_class.__name__}("{concept.preferred_label}")'
    )


@pytest.mark.xdist_group(name="classifier")
@pytest.mark.parametrize("classifier_class", classifier_classes)
@given(concept=concept_strategy())
@settings(max_examples=100, database=None)
def test_whether_classifier_hashes_are_generated_correctly(
    classifier_class: Type[Classifier], concept: Concept
):
    classifier = classifier_class(concept)
    assert classifier.id == ClassifierID.generate(classifier.name, concept.id)
    assert classifier == classifier_class(concept)


@pytest.mark.xdist_group(name="classifier")
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


@pytest.mark.xdist_group(name="classifier")
@pytest.mark.parametrize("classifier_class", classifier_classes)
@given(concepts=st.sets(concept_strategy(), min_size=10, max_size=10))
@settings(max_examples=100, database=None)
def test_whether_different_concepts_produce_different_hashes_when_using_the_same_classifier_class(
    classifier_class: Type[Classifier], concepts: list[Concept]
):
    # classifiers of the same class, for different concepts
    classifiers = [classifier_class(concept) for concept in concepts]
    hashes = [hash(classifier) for classifier in classifiers]
    assert len(set(hashes)) == len(hashes)


@pytest.mark.xdist_group(name="classifier")
@given(concept=concept_strategy())
@settings(max_examples=100, database=None)
def test_whether_different_classifier_models_produce_different_hashes_when_based_on_the_same_concept(
    concept: Concept,
):
    # classifiers of different classes, for the same concept
    classifiers = [classifier_class(concept) for classifier_class in classifier_classes]
    hashes = [hash(classifier) for classifier in classifiers]
    assert len(set(hashes)) == len(hashes)


@pytest.mark.xdist_group(name="classifier")
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

    assert concept.wikibase_id is not None
    new_wikibase_id = WikibaseID("Q" + str(concept.wikibase_id.numeric + 1))
    augmented_concept = concept.model_copy(update={"wikibase_id": new_wikibase_id})
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


@pytest.mark.xdist_group(name="classifier")
def test_whether_a_classifier_which_does_not_specify_allowed_concept_ids_accepts_any_concept():
    class UnrestrictedClassifier(Classifier):
        @property
        def id(self) -> ClassifierID:
            return ClassifierID("unrestricted")

        def _predict(self, text: str) -> list[Span]:
            return []

    concept1 = Concept(wikibase_id=WikibaseID("Q123"), preferred_label="test")
    concept2 = Concept(wikibase_id=WikibaseID("Q456"), preferred_label="test")

    assert UnrestrictedClassifier(concept1)
    assert UnrestrictedClassifier(concept2)


@pytest.mark.xdist_group(name="classifier")
def test_whether_a_classifier_with_a_single_allowed_concept_id_validates_correctly():
    class SingleIDClassifier(Classifier):
        allowed_concept_ids = [WikibaseID("Q123")]

        @property
        def id(self) -> ClassifierID:
            return ClassifierID("single_id")

        def _predict(self, text: str) -> list[Span]:
            return []

    valid_concept = Concept(wikibase_id=WikibaseID("Q123"), preferred_label="test")
    invalid_concept = Concept(wikibase_id=WikibaseID("Q456"), preferred_label="test")

    assert SingleIDClassifier(valid_concept)

    with pytest.raises(ValueError) as exc_info:
        SingleIDClassifier(invalid_concept)
    assert "must be Q123" in str(exc_info.value)
    assert "not Q456" in str(exc_info.value)


@pytest.mark.xdist_group(name="classifier")
def test_whether_a_classifier_with_multiple_allowed_concept_ids_validates_correctly():
    class MultiIDClassifier(Classifier):
        allowed_concept_ids = [WikibaseID("Q123"), WikibaseID("Q456")]

        @property
        def id(self) -> ClassifierID:
            return ClassifierID("multi_id")

        def _predict(self, text: str) -> list[Span]:
            return []

    valid_concept = Concept(wikibase_id=WikibaseID("Q123"), preferred_label="test")
    invalid_concept = Concept(wikibase_id=WikibaseID("Q789"), preferred_label="test")

    assert MultiIDClassifier(valid_concept)

    with pytest.raises(ValueError) as exc_info:
        MultiIDClassifier(invalid_concept)
    assert "must be one of Q123,Q456" in str(exc_info.value)
    assert "not Q789" in str(exc_info.value)


@pytest.mark.xdist_group(name="classifier")
def test_whether_allowed_concept_ids_validation_works_correctly_with_inheritance():
    class ParentClassifier(Classifier):
        allowed_concept_ids = [WikibaseID("Q123"), WikibaseID("Q456")]

        @property
        def id(self) -> ClassifierID:
            return ClassifierID("parent_id")

        def _predict(self, text: str) -> list[Span]:
            return []

    class ChildClassifier(ParentClassifier):
        allowed_concept_ids = [WikibaseID("Q123")]  # More restrictive than parent

    valid_concept = Concept(wikibase_id=WikibaseID("Q123"), preferred_label="test")
    parent_only_concept = Concept(
        wikibase_id=WikibaseID("Q456"), preferred_label="test"
    )

    # Parent should accept both concepts
    assert ParentClassifier(valid_concept)
    assert ParentClassifier(parent_only_concept)

    # Child should only accept its own allowed ID
    assert ChildClassifier(valid_concept)
    with pytest.raises(ValueError) as exc_info:
        ChildClassifier(parent_only_concept)
    assert "must be Q123" in str(exc_info.value)
    assert "not Q456" in str(exc_info.value)


@pytest.mark.xdist_group(name="classifier")
def test_whether_an_empty_allowed_concept_ids_list_accepts_all_concepts():
    """
    Test whether supplying an empty list of allowed_concept_ids is prohibitive.

    The expected behaviour is that the classifier should accept any concept,
    regardless of its wikibase_id.
    """

    class EmptyIDClassifier(Classifier):
        allowed_concept_ids = []

        @property
        def id(self) -> ClassifierID:
            return ClassifierID("empty_id")

        def _predict(self, text: str) -> list[Span]:
            return []

    concept = Concept(wikibase_id=WikibaseID("Q123"), preferred_label="test")

    assert EmptyIDClassifier(concept)


@pytest.mark.xdist_group(name="classifier")
@pytest.mark.parametrize("classifier_class", classifier_classes)
@given(concept=concept_strategy(), data=st.data())
def test_predict_sequence_returns_predictions_for_all_texts(
    classifier_class: Type[Classifier], concept: Concept, data: st.DataObject
):
    """Test that predict_many returns predictions for all input texts."""
    # Generate multiple positive texts
    num_texts = 5
    texts = [
        data.draw(positive_text_strategy(labels=concept.all_labels))
        for _ in range(num_texts)
    ]

    classifier = classifier_class(concept)
    predictions = classifier.predict(texts, batch_size=2)

    assert len(predictions) == num_texts
    assert all(isinstance(pred_list, list) for pred_list in predictions)


@pytest.mark.xdist_group(name="classifier")
@pytest.mark.parametrize("classifier_class", classifier_classes)
@given(concept=concept_strategy())
def test_predict_sequence_preserves_order(
    classifier_class: Type[Classifier], concept: Concept
):
    """Test that predict_many returns predictions in the same order as input texts."""
    texts = [
        f"Some text with {concept.preferred_label} here",
        "Some unrelated text without the concept",
        f"Another text with {concept.preferred_label}",
    ]

    classifier = classifier_class(concept)
    predictions = classifier.predict(texts, batch_size=2)

    assert len(predictions) == 3
    # First text should have predictions
    assert len(predictions[0]) > 0
    # Second text should have no predictions (doesn't contain the label)
    assert len(predictions[1]) == 0
    # Third text should have predictions
    assert len(predictions[2]) > 0


@st.composite
def label_with_separator_variant_strategy(draw, label: str):
    r"""
    Given a label, return it with a different separator.

    Eg. takes "greenhouse gas" and returns "greenhouse\ngas" or "greenhouse-gas" etc.
    """
    words = [
        w
        for w in re.split(
            pattern=KeywordClassifier.separator_pattern, string=label.strip()
        )
        if w
    ]

    if len(words) == 1:
        return label

    # Draw a single separator character matching the classifier's pattern
    # Remove the '+' quantifier to get just one character
    single_separator_pattern = KeywordClassifier.separator_pattern.rstrip("+")
    variant_sep = draw(st.from_regex(single_separator_pattern, fullmatch=True))
    return variant_sep.join(words)


@given(label_data=st.data(), label_variant_data=st.data(), text_data=st.data())
@settings(max_examples=100, database=None)
@pytest.mark.xdist_group(name="classifier")
@pytest.mark.parametrize("classifier_class", classifier_classes)
def test_whether_multi_word_labels_match_text_with_different_separators(
    classifier_class: Type[Classifier],
    label_data: st.DataObject,
    label_variant_data: st.DataObject,
    text_data: st.DataObject,
):
    """Test that labels defined with one separator match text with different separators."""
    label = label_data.draw(multi_word_label_strategy())
    label_variant = label_variant_data.draw(
        label_with_separator_variant_strategy(label)
    )
    text = text_data.draw(positive_text_strategy(labels=[label_variant]))
    concept = Concept(wikibase_id=WikibaseID("Q123"), preferred_label=label)
    classifier = classifier_class(concept)
    spans = classifier.predict(text)

    assert spans, f"{classifier} did not match label '{label}' in text: '{text}'"


@pytest.mark.xdist_group(name="classifier")
@pytest.mark.parametrize("classifier_class", classifier_classes)
@pytest.mark.parametrize("batch_size", [1, 2, 5, 10])
@given(concept=concept_strategy(), data=st.data())
def test_predict_sequence_works_with_different_batch_sizes(
    classifier_class: Type[Classifier],
    concept: Concept,
    batch_size: int,
    data: st.DataObject,
):
    """Test that predict_many produces consistent results with different batch sizes."""
    num_texts = 10
    texts = [
        data.draw(positive_text_strategy(labels=concept.all_labels))
        for _ in range(num_texts)
    ]

    classifier = classifier_class(concept)
    predictions = classifier.predict(texts, batch_size=batch_size)

    assert len(predictions) == num_texts
    assert all(isinstance(pred_list, list) for pred_list in predictions)


@pytest.mark.parametrize("classifier_class", classifier_classes)
@given(label_data=st.data(), negative_label_data=st.data(), text_data=st.data())
@settings(max_examples=100, database=None)
def test_whether_negative_labels_filter_matches_regardless_of_separator(
    classifier_class: Type[Classifier],
    label_data: st.DataObject,
    negative_label_data: st.DataObject,
    text_data: st.DataObject,
):
    """Test that negative labels defined with one separator filter text with different separators."""
    positive_label = label_data.draw(concept_label_strategy)
    negative_label = (
        positive_label + " " + negative_label_data.draw(single_word_label_strategy)
    )
    negative_variant = negative_label_data.draw(
        label_with_separator_variant_strategy(negative_label)
    )
    text = text_data.draw(positive_text_strategy(labels=[negative_variant]))
    concept = Concept(
        wikibase_id=WikibaseID("Q123"),
        preferred_label=positive_label,
        negative_labels=[negative_label],
    )
    classifier = classifier_class(concept)
    spans = classifier.predict(text)
    assert not spans, (
        f"{classifier} matched text in '{text}', but it shouldn't have! The negative label '{negative_label}' should filter out the positive match."
    )


@pytest.mark.xdist_group(name="classifier")
@pytest.mark.parametrize("classifier_class", classifier_classes)
@given(label=single_word_label_strategy)
@settings(max_examples=100, database=None)
def test_whether_single_word_labels_respect_word_boundaries(
    classifier_class: Type[Classifier], label: str
):
    """Test that single-word labels still respect word boundaries."""
    concept = Concept(wikibase_id=WikibaseID("Q123"), preferred_label=label)
    classifier = classifier_class(concept)
    assert classifier.predict(f"The {label} is important.")
    assert not classifier.predict(f"xyz{label}abc")


@pytest.mark.xdist_group(name="classifier")
def test_whether_classifier_respects_case_sensitivity():
    # Uppercase-containing label should match exactly as-is
    uppercase_concept = Concept(wikibase_id=WikibaseID("Q123"), preferred_label="WHO")
    uppercase_classifier = KeywordClassifier(uppercase_concept)
    assert uppercase_classifier.predict("The WHO released guidance.")
    assert not uppercase_classifier.predict("the who released guidance.")

    # Lowercase-only label should match any case
    lowercase_concept = Concept(wikibase_id=WikibaseID("Q123"), preferred_label="who")
    lowercase_classifier = KeywordClassifier(lowercase_concept)
    assert lowercase_classifier.predict("The WHO released guidance.")
    assert lowercase_classifier.predict("the who released guidance.")


@pytest.mark.xdist_group(name="classifier")
@pytest.mark.parametrize(
    "label,text,should_match",
    [
        ("gas", "gas, prices rose", True),
        ("gas", "(gas) is discussed", True),
        ("greenhouse gas", "greenhouse-gas emissions", True),
        ("greenhouse gas", "(greenhouse gas) emissions", True),
    ],
)
def test_whether_classifier_respects_punctuation_as_word_boundaries(
    label: str, text: str, should_match: bool
):
    concept = Concept(wikibase_id=WikibaseID("Q123"), preferred_label=label)
    classifier = KeywordClassifier(concept)
    spans = classifier.predict(text)
    assert bool(spans) == should_match


@pytest.mark.xdist_group(name="classifier")
def test_whether_classifier_merges_overlapping_spans_to_the_longest_phrase():
    concept = Concept(
        wikibase_id=WikibaseID("Q123"),
        preferred_label="greenhouse gas",
        alternative_labels=["gas"],
    )
    classifier = KeywordClassifier(concept)
    text = "Greenhouse-gas emissions are measured."
    spans = classifier.predict(text)
    # Should return a single merged span matching the longer phrase variant
    assert len(spans) == 1
    assert spans[0].labelled_text.lower().replace("-", " ") == "greenhouse gas"


@pytest.mark.xdist_group(name="classifier")
@pytest.mark.parametrize(
    "label,variant_text",
    [
        ("CO₂", "CO₂ emissions"),
        ("CO₂", "(CO₂) emissions"),
        ("Météo", "Météo report"),
        ("Météo", "Météo\nreport"),
    ],
)
def test_whether_classifier_handles_non_ascii_labels_across_separators(
    label: str, variant_text: str
):
    concept = Concept(wikibase_id=WikibaseID("Q123"), preferred_label=label)
    classifier = KeywordClassifier(concept)
    spans = classifier.predict(variant_text)
    assert spans
