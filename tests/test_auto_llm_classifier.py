"""Tests for AutoLLMClassifier."""

import pytest

from knowledge_graph.classifier.large_language_model import (
    AUTO_DEFAULT_SYSTEM_PROMPT,
    DEFAULT_INSTRUCTIONS,
    AutoLLMClassifier,
)
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.span import Span


def test_autollmclassifier_initialization():
    """Test AutoLLMClassifier can be instantiated."""
    concept = Concept(
        preferred_label="Test Concept",
        wikibase_id=WikibaseID("Q123"),
    )
    classifier = AutoLLMClassifier(concept=concept)

    assert classifier.optimized_instructions is None
    assert not classifier.is_fitted
    assert classifier.optimization_model_name == "gemini-2.0-flash"


def test_autollmclassifier_with_custom_optimization_model():
    """Test AutoLLMClassifier with different optimization model."""
    concept = Concept(
        preferred_label="Test Concept",
        wikibase_id=WikibaseID("Q123"),
    )
    classifier = AutoLLMClassifier(
        concept=concept, model_name="gpt-4o", optimization_model_name="gpt-4o-mini"
    )

    assert classifier.model_name == "gpt-4o"
    assert classifier.optimization_model_name == "gpt-4o-mini"


def test_fit_with_insufficient_data():
    """Test fit() handles insufficient data gracefully."""
    concept = Concept(
        preferred_label="Test",
        wikibase_id=WikibaseID("Q123"),
        labelled_passages=[LabelledPassage(id="1", text="test", spans=[], metadata={})]
        * 5,  # Only 5 passages, need 10
    )
    classifier = AutoLLMClassifier(concept=concept)

    # This should fall back to default instructions without calling LLM
    classifier.fit(min_passages=10)

    assert classifier.is_fitted
    assert classifier.optimized_instructions == DEFAULT_INSTRUCTIONS


def test_prepare_dspy_examples():
    """Test _prepare_dspy_examples correctly converts LabelledPassages."""
    passages = [
        LabelledPassage(
            id=f"{i}",
            text="positive passage",
            spans=[
                Span(
                    text="positive passage",
                    start_index=0,
                    end_index=10,
                    concept_id=WikibaseID("Q123"),
                    labellers=["test"],
                    timestamps=[],
                )
            ],
            metadata={},
        )
        if i < 5
        else LabelledPassage(
            id=f"{i}",
            text="negative passage",
            spans=[],
            metadata={},
        )
        for i in range(10)
    ]  # 5 positive, 5 negative

    concept = Concept(preferred_label="Test", wikibase_id=WikibaseID("Q123"))
    classifier = AutoLLMClassifier(concept=concept)

    train, val = classifier._prepare_dspy_examples(passages, validation_size=0.2)

    # Check split sizes
    assert len(train) == 8
    assert len(val) == 2

    # Check example structure
    assert all(hasattr(ex, "passage_text") for ex in train)
    assert all(hasattr(ex, "gold_spans") for ex in train)
    assert all(hasattr(ex, "passage_id") for ex in train)


@pytest.mark.parametrize(
    "model_name,expected_prefix",
    [
        ("gpt-4o", "openai/"),
        ("claude-3-5-sonnet", "anthropic/"),
        ("gemini-2.0-flash", "google/"),
    ],
)
def test_create_dspy_lm(model_name, expected_prefix):
    """Test _create_dspy_lm maps model names correctly."""
    concept = Concept(preferred_label="Test", wikibase_id=WikibaseID("Q123"))
    classifier = AutoLLMClassifier(concept=concept, optimization_model_name=model_name)

    lm = classifier._create_dspy_lm()
    assert lm.model.startswith(expected_prefix)


def test_classifier_id_changes_after_fit():
    """Test classifier ID changes after optimization (when instructions change)."""
    concept = Concept(
        preferred_label="Test",
        wikibase_id=WikibaseID("Q123"),
        labelled_passages=[
            LabelledPassage(
                id=f"{i}",
                text=f"passage {i}",
                spans=[],
                metadata={},
            )
            for i in range(15)
        ],
    )

    classifier = AutoLLMClassifier(concept=concept)
    id_before = classifier.id

    # Mock fit to avoid real LM calls
    classifier.optimized_instructions = "NEW INSTRUCTIONS"
    classifier.is_fitted = True

    id_after = classifier.id
    assert id_before != id_after


def test_constants_defined():
    """Test that all required constants are defined."""
    assert AUTO_DEFAULT_SYSTEM_PROMPT is not None
    assert "{concept_description}" in AUTO_DEFAULT_SYSTEM_PROMPT
    assert "{instructions}" in AUTO_DEFAULT_SYSTEM_PROMPT

    assert DEFAULT_INSTRUCTIONS is not None
    assert len(DEFAULT_INSTRUCTIONS) > 0
