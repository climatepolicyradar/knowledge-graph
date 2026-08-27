import pickle
import re
from typing import Type
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from knowledge_graph.classifier.bert_based import BertBasedClassifier
from knowledge_graph.classifier.classifier import Classifier
from knowledge_graph.classifier.keyword import (
    KeywordClassifier,
    fold_subscript_characters,
    make_suffix_flexible_regex,
)
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
@given(concept=concept_strategy(), data=st.data())
def test_predict_sequence_preserves_order(
    classifier_class: Type[Classifier], concept: Concept, data: st.DataObject
):
    """Test that predict_many returns predictions in the same order as input texts."""
    positive_text_1 = data.draw(positive_text_strategy(labels=concept.all_labels))
    negative_text = data.draw(negative_text_strategy(labels=concept.all_labels))
    positive_text_2 = data.draw(positive_text_strategy(labels=concept.all_labels))

    texts = [
        positive_text_1,
        negative_text,
        positive_text_2,
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


_SUBSCRIPT_DIGITS = {str(digit): chr(0x2080 + digit) for digit in range(10)}


def _to_subscripts(text: str) -> str:
    """Rewrite every ASCII digit in the text as its subscript equivalent."""
    return "".join(_SUBSCRIPT_DIGITS.get(character, character) for character in text)


@pytest.mark.parametrize(
    "text,expected",
    [
        ("CO₂", "CO2"),
        ("CO₂ emissions", "CO2 emissions"),
        ("N₂O and CH₄", "N2O and CH4"),
        ("m³", "m3"),
        ("x⁴", "x4"),
        ("10⁻⁶", "10-6"),
        ("no scripts here", "no scripts here"),
        ("", ""),
    ],
)
def test_fold_subscript_characters(text: str, expected: str):
    assert fold_subscript_characters(text) == expected


@given(text=st.text())
@settings(max_examples=200, database=None)
def test_whether_folding_preserves_length(text: str):
    """
    The whole span-offset argument rests on folding being length-preserving.

    If any mapping were ever 1:many, offsets into the folded text would silently stop
    lining up with the original text.
    """
    assert len(fold_subscript_characters(text)) == len(text)


def test_whether_folding_is_idempotent():
    once = fold_subscript_characters("CO₂ and 10⁻⁶")
    assert fold_subscript_characters(once) == once


# spellchecker:off
@pytest.mark.parametrize(
    "word,matches,does_not_match",
    [
        ("gas", ["gas", "gases"], ["ga", "gasses"]),
        ("policy", ["policy", "policies"], ["polic", "policys"]),
        ("target", ["target", "targets"], ["targe"]),
        ("box", ["box", "boxes"], ["boxs"]),
        ("branch", ["branch", "branches"], ["branchs"]),
        ("day", ["day", "days"], ["daies"]),  # vowel + y, so not the -ies rule
        ("NDC", ["NDC", "NDCs", "NDCS"], ["ND"]),
        ("co2", ["co2"], ["co2s"]),  # trailing digit, left alone
        ("EU", ["EU"], ["EUs"]),  # under the length floor
        # Already-plural labels gain nothing: the rule only widens singular to plural
        ("emissions", ["emissions"], ["emission"]),
    ],
)
# spellchecker:on
def test_suffix_flexible(word: str, matches: list[str], does_not_match: list[str]):
    pattern = re.compile(rf"(?<!\w)(?:{make_suffix_flexible_regex(word)})(?!\w)")

    for candidate in matches:
        assert pattern.search(candidate), f"{word!r} should match {candidate!r}"

    for candidate in does_not_match:
        assert not pattern.search(candidate), f"{word!r} should not match {candidate!r}"


def test_whether_suffix_flexible_escapes_regex_metacharacters():
    pattern = re.compile(rf"^(?:{make_suffix_flexible_regex('c++')})$")
    assert pattern.search("c++")
    assert not pattern.search("cxx")


@pytest.mark.xdist_group(name="classifier")
def test_whether_keyword_matching_relaxations_are_on_by_default():
    concept = Concept(wikibase_id=WikibaseID("Q123"), preferred_label="greenhouse gas")
    classifier = KeywordClassifier(concept)

    assert classifier.fold_subscripts is True
    assert classifier.match_word_forms is True

    assert classifier.predict("greenhouse gases were measured")


@pytest.mark.xdist_group(name="classifier")
def test_whether_the_strict_configuration_keeps_its_original_id():
    """
    Turning the relaxations off must reproduce the pre-relaxation id exactly.

    This is what lets an already-trained classifier be rebuilt at the id its artifact,
    model path and classifier spec entry were recorded under.
    """
    concept = Concept(wikibase_id=WikibaseID("Q123"), preferred_label="greenhouse gas")
    classifier = KeywordClassifier(
        concept, fold_subscripts=False, match_word_forms=False
    )

    assert classifier.id == ClassifierID.generate(classifier.name, concept.id)
    assert classifier.id != KeywordClassifier(concept).id


@given(concept=concept_strategy(), text_data=st.data())
@settings(max_examples=100, database=None)
@pytest.mark.xdist_group(name="classifier")
def test_whether_relaxed_classifier_spans_index_into_the_original_text(
    concept: Concept, text_data: st.DataObject
):
    """
    Spans must always be offsets into the text that was passed in.

    This is the guard for subscript folding being length-preserving: the classifier
    searches folded text, so if the fold ever changed a string's length the offsets
    would silently stop lining up with the original.
    """
    text = _to_subscripts(
        text_data.draw(
            positive_text_strategy(
                labels=concept.all_labels, negative_labels=concept.negative_labels
            )
        )
    )
    classifier = KeywordClassifier(concept, fold_subscripts=True, match_word_forms=True)

    for span in classifier.predict(text):
        assert 0 <= span.start_index < span.end_index <= len(text)
        assert span.labelled_text == text[span.start_index : span.end_index]
        assert span.concept_id == concept.wikibase_id


@pytest.mark.xdist_group(name="classifier")
@pytest.mark.parametrize(
    "label,text,expected_match",
    [
        ("CO2", "CO₂ emissions rose", "CO₂"),
        ("CO₂", "CO2 emissions rose", "CO2"),
        ("CO2", "CO2 emissions rose", "CO2"),
        ("CO₂", "CO₂ emissions rose", "CO₂"),
        ("co2", "the CO₂ emissions rose", "CO₂"),
        ("CH4", "(CH₄) is a potent gas", "CH₄"),
    ],
)
def test_whether_folding_subscripts_matches_across_scripts(
    label: str, text: str, expected_match: str
):
    concept = Concept(wikibase_id=WikibaseID("Q123"), preferred_label=label)
    classifier = KeywordClassifier(concept, fold_subscripts=True)

    spans = classifier.predict(text)

    assert len(spans) == 1
    # The span must quote the original text, not the folded version of it
    assert spans[0].labelled_text == expected_match


@pytest.mark.xdist_group(name="classifier")
def test_whether_folding_subscripts_can_be_disabled():
    concept = Concept(wikibase_id=WikibaseID("Q123"), preferred_label="CO2")

    assert not KeywordClassifier(concept, fold_subscripts=False).predict(
        "CO₂ emissions rose"
    )


@pytest.mark.xdist_group(name="classifier")
@pytest.mark.parametrize(
    "label,text,should_match",
    [
        ("gas", "the gases were measured", True),
        ("policy", "adaptation policies were adopted", True),
        ("greenhouse gas", "greenhouse gases were measured", True),
        ("greenhouse gas", "greenhouse-gases were measured", True),
        ("target", "the targets were missed", True),
        ("gas", "the gas was measured", True),
        ("gas", "the gasify plant", False),
        ("policy", "the polices were adopted", False),
    ],
)
def test_whether_matching_word_forms_matches_plurals(
    label: str, text: str, should_match: bool
):
    concept = Concept(wikibase_id=WikibaseID("Q123"), preferred_label=label)
    classifier = KeywordClassifier(concept, match_word_forms=True)

    assert bool(classifier.predict(text)) == should_match


@pytest.mark.xdist_group(name="classifier")
def test_whether_matching_word_forms_can_be_disabled():
    concept = Concept(wikibase_id=WikibaseID("Q123"), preferred_label="greenhouse gas")

    assert not KeywordClassifier(concept, match_word_forms=False).predict(
        "greenhouse gases were measured"
    )


@pytest.mark.xdist_group(name="classifier")
def test_whether_word_forms_apply_to_negative_labels_too():
    """A relaxed positive match must not slip past a still-strict negative veto."""
    concept = Concept(
        wikibase_id=WikibaseID("Q123"),
        preferred_label="gas",
        negative_labels=["greenhouse gas"],
    )
    classifier = KeywordClassifier(concept, match_word_forms=True)

    assert classifier.predict("the gases were measured")
    assert not classifier.predict("greenhouse gases were measured")


@pytest.mark.xdist_group(name="classifier")
def test_whether_match_options_change_the_classifier_id():
    concept = Concept(wikibase_id=WikibaseID("Q123"), preferred_label="greenhouse gas")

    variants = [
        KeywordClassifier(concept, fold_subscripts=False, match_word_forms=False),
        KeywordClassifier(concept, fold_subscripts=True, match_word_forms=False),
        KeywordClassifier(concept, fold_subscripts=False, match_word_forms=True),
        KeywordClassifier(concept, fold_subscripts=True, match_word_forms=True),
    ]

    ids = [classifier.id for classifier in variants]
    assert len(set(ids)) == len(ids), f"ids collided: {ids}"

    # Passing the defaults explicitly is still the default configuration
    assert KeywordClassifier(concept).id == variants[-1].id


@pytest.mark.xdist_group(name="classifier")
def test_whether_match_options_ids_are_deterministic():
    concept = Concept(wikibase_id=WikibaseID("Q123"), preferred_label="greenhouse gas")
    kwargs = {"fold_subscripts": True, "match_word_forms": True}

    assert (
        KeywordClassifier(concept, **kwargs).id
        == KeywordClassifier(concept, **kwargs).id
    )


@pytest.mark.xdist_group(name="classifier")
def test_whether_a_relaxed_keyword_classifier_survives_a_pickle_round_trip(tmp_path):
    """Classifiers are saved with plain pickle, including their compiled patterns."""
    concept = Concept(wikibase_id=WikibaseID("Q123"), preferred_label="greenhouse gas")
    original = KeywordClassifier(concept, fold_subscripts=True, match_word_forms=True)
    text = "greenhouse gases and CO₂ were measured"

    path = tmp_path / "keyword_classifier.pickle"
    original.save(path)
    loaded = Classifier.load(path)

    assert isinstance(loaded, KeywordClassifier)
    assert loaded.fold_subscripts is True
    assert loaded.match_word_forms is True
    assert loaded.id == original.id
    assert [(span.start_index, span.end_index) for span in loaded.predict(text)] == [
        (span.start_index, span.end_index) for span in original.predict(text)
    ]


@pytest.mark.xdist_group(name="classifier")
def test_whether_keyword_classifiers_pickled_before_match_options_still_work(tmp_path):
    """
    Old pickles predate the match options and must still load, predict and keep their id.

    Every KeywordClassifier already saved to W&B and S3 was pickled without these
    attributes, so falling back to the class-level defaults is what stops those
    artifacts breaking on their first predict after this change.
    """
    concept = Concept(
        wikibase_id=WikibaseID("Q123"),
        preferred_label="greenhouse gas",
        negative_labels=["natural gas"],
    )
    original = KeywordClassifier(concept, fold_subscripts=False, match_word_forms=False)
    expected_id = original.id

    # Simulate a classifier pickled before the match options existed
    for attribute in ["fold_subscripts", "match_word_forms"]:
        del original.__dict__[attribute]

    path = tmp_path / "old_keyword_classifier.pickle"
    original.save(path)
    loaded = Classifier.load(path)

    assert isinstance(loaded, KeywordClassifier)
    assert "fold_subscripts" not in loaded.__dict__
    assert loaded.fold_subscripts is False
    assert loaded.match_word_forms is False
    assert loaded.id == expected_id
    assert [
        span.labelled_text for span in loaded.predict("greenhouse gas emissions")
    ] == ["greenhouse gas"]
    # The negative labels must still veto
    assert not loaded.predict("natural gas prices rose")


def test_classifier_load_reinitializes_bert_based_classifier(tmp_path):
    import torch

    concept = Concept(wikibase_id=WikibaseID("Q123"), preferred_label="test")

    original = BertBasedClassifier(
        concept=concept,
        model_name="test-model",
        download_pretrained_model_on_init=False,
    )
    original.model = "fake_trained_model"
    original.tokenizer = "fake_trained_tokenizer"
    original.device = torch.device("cpu")
    original.pipeline = "fake_trained_pipeline"
    original.model_name = "test-model"
    original.is_fitted = True
    original.prediction_threshold = 0.7

    path = tmp_path / "bert_classifier.pkl"
    with open(path, "wb") as f:
        pickle.dump(original, f)

    # Patch _predict_batch on the class *after* pickling so the pickle contains the
    # old method reference. The loaded instance should use the current (patched) class
    # method, proving it was reconstructed from the current class definition.
    sentinel = object()
    with patch.object(BertBasedClassifier, "_predict_batch", return_value=sentinel):
        loaded = Classifier.load(path)
        assert loaded._predict_batch(["text"]) is sentinel

    assert isinstance(loaded, BertBasedClassifier)
    assert loaded is not original, "Should be a fresh instance, not the pickled object"
    assert loaded.model == "fake_trained_model"
    assert loaded.tokenizer == "fake_trained_tokenizer"
    # Device should be re-resolved to the best available, not preserved from pickle
    assert loaded.device == loaded._resolve_device()
    assert loaded.pipeline == "fake_trained_pipeline"
    assert loaded.is_fitted is True
    assert loaded.prediction_threshold == 0.7


def test_classifier_load_defaults_missing_max_length_to_512(tmp_path):
    """Old pickles predate max_length and must load with the 512 inference default."""
    concept = Concept(wikibase_id=WikibaseID("Q123"), preferred_label="test")

    original = BertBasedClassifier(
        concept=concept,
        model_name="test-model",
        download_pretrained_model_on_init=False,
    )
    original.model = "fake_trained_model"  # type: ignore[assignment]

    # Simulate a classifier pickled before the max_length attribute existed.
    del original.__dict__["max_length"]

    path = tmp_path / "old_bert_classifier.pkl"
    with open(path, "wb") as f:
        pickle.dump(original, f)

    loaded = Classifier.load(path)

    assert isinstance(loaded, BertBasedClassifier)
    assert loaded.max_length == 512


def test_classifier_load_preserves_new_max_length_default(tmp_path):
    """Newly constructed classifiers keep the 1024 default across a load round-trip."""
    concept = Concept(wikibase_id=WikibaseID("Q123"), preferred_label="test")

    original = BertBasedClassifier(
        concept=concept,
        model_name="test-model",
        download_pretrained_model_on_init=False,
    )
    original.model = "fake_trained_model"  # type: ignore[assignment]

    assert original.max_length == 1024

    path = tmp_path / "new_bert_classifier.pkl"
    with open(path, "wb") as f:
        pickle.dump(original, f)

    loaded = Classifier.load(path)

    assert isinstance(loaded, BertBasedClassifier)
    assert loaded.max_length == 1024
