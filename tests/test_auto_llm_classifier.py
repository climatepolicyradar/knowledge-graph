"""Tests for AutoLLMClassifier."""

from unittest.mock import MagicMock, patch

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
    assert classifier.evaluation_model_name == "gemini-2.0-flash"
    assert classifier.proposal_model_name == "gemini-2.0-flash"


def test_autollmclassifier_with_custom_models():
    """Test AutoLLMClassifier with different models for each phase."""
    concept = Concept(
        preferred_label="Test Concept",
        wikibase_id=WikibaseID("Q123"),
    )
    classifier = AutoLLMClassifier(
        concept=concept,
        classifier_model_name="gpt-4o",
        evaluation_model_name="gpt-4o-mini",
        proposal_model_name="gpt-4o",
    )

    assert classifier.model_name == "gpt-4o"  # classifier model
    assert classifier.evaluation_model_name == "gpt-4o-mini"
    assert classifier.proposal_model_name == "gpt-4o"


def test_autollmclassifier_model_defaults():
    """Test that models default correctly when not specified."""
    concept = Concept(
        preferred_label="Test Concept",
        wikibase_id=WikibaseID("Q123"),
    )
    # Only specify classifier model - others should default
    classifier = AutoLLMClassifier(
        concept=concept,
        classifier_model_name="claude-3-5-sonnet",
    )

    assert classifier.model_name == "claude-3-5-sonnet"
    assert (
        classifier.evaluation_model_name == "claude-3-5-sonnet"
    )  # defaults to classifier
    assert (
        classifier.proposal_model_name == "claude-3-5-sonnet"
    )  # defaults to evaluation


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
    """Test _create_dspy_lm maps model names correctly and sets temperature."""
    concept = Concept(preferred_label="Test", wikibase_id=WikibaseID("Q123"))
    classifier = AutoLLMClassifier(concept=concept, evaluation_model_name=model_name)

    # Test with default (uses evaluation_model_name)
    lm = classifier._create_dspy_lm()
    assert lm.model.startswith(expected_prefix)

    # Test with explicit model name
    lm_explicit = classifier._create_dspy_lm(model_name=model_name, temperature=0.9)
    assert lm_explicit.model.startswith(expected_prefix)


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


class TestExtractOptimizedInstructions:
    """Tests for _extract_optimized_instructions method."""

    def test_extract_with_instructions_attribute(self):
        """Test extraction when signature has instructions attribute."""

        from knowledge_graph.classifier.dspy_components import (
            ConceptTaggerModule,
            ConceptTaggingSignature,
        )

        concept = Concept(preferred_label="Test", wikibase_id=WikibaseID("Q123"))
        classifier = AutoLLMClassifier(concept=concept)

        # Create a module like MIPRO would produce
        module = ConceptTaggerModule(
            concept_description="Test concept",
            signature=ConceptTaggingSignature,
        )

        # Manually set instructions to simulate MIPRO optimization
        optimized_instruction_text = "These are the optimized instructions from MIPRO"
        module.predict.signature = module.predict.signature.with_instructions(
            optimized_instruction_text
        )

        # Extract instructions
        extracted = classifier._extract_optimized_instructions(module)

        assert extracted == optimized_instruction_text

    def test_extract_fallback_to_docstring(self):
        """Test extraction falls back to docstring when no instructions attribute."""

        from knowledge_graph.classifier.dspy_components import (
            ConceptTaggerModule,
            ConceptTaggingSignature,
        )

        concept = Concept(preferred_label="Test", wikibase_id=WikibaseID("Q123"))
        classifier = AutoLLMClassifier(concept=concept)

        # Create an unmodified module
        module = ConceptTaggerModule(
            concept_description="Test concept",
            signature=ConceptTaggingSignature,
        )

        # Extract instructions - should get docstring or DEFAULT_INSTRUCTIONS
        extracted = classifier._extract_optimized_instructions(module)

        # The fallback should return something meaningful
        assert extracted is not None
        assert len(extracted) > 0

    def test_extract_with_none_instructions(self):
        """Test extraction when instructions is None."""

        from knowledge_graph.classifier.dspy_components import (
            ConceptTaggerModule,
            ConceptTaggingSignature,
        )

        concept = Concept(preferred_label="Test", wikibase_id=WikibaseID("Q123"))
        classifier = AutoLLMClassifier(concept=concept)

        module = ConceptTaggerModule(
            concept_description="Test concept",
            signature=ConceptTaggingSignature,
        )

        # Check what the signature looks like
        signature = module.predict.signature
        print(f"Signature type: {type(signature)}")
        print(f"Has instructions attr: {hasattr(signature, 'instructions')}")
        if hasattr(signature, "instructions"):
            print(f"Instructions value: {signature.instructions}")
            print(f"Instructions type: {type(signature.instructions)}")

        extracted = classifier._extract_optimized_instructions(module)
        assert extracted is not None

    def test_extract_with_empty_instructions_returns_fallback(self):
        """Test extraction falls back when instructions is empty string."""
        concept = Concept(preferred_label="Test", wikibase_id=WikibaseID("Q123"))
        classifier = AutoLLMClassifier(concept=concept)

        # Create a mock module with empty instructions
        class MockModule:
            class predict:
                class signature:
                    instructions = ""  # Empty string
                    __doc__ = "Fallback docstring"

        extracted = classifier._extract_optimized_instructions(MockModule())

        # Should fall back to __doc__
        assert extracted == "Fallback docstring"

    def test_extract_with_none_instructions_returns_fallback(self):
        """Test extraction falls back when instructions is None."""
        concept = Concept(preferred_label="Test", wikibase_id=WikibaseID("Q123"))
        classifier = AutoLLMClassifier(concept=concept)

        # Create a mock module with None instructions
        class MockModule:
            class predict:
                class signature:
                    instructions = None
                    __doc__ = "Fallback docstring"

        extracted = classifier._extract_optimized_instructions(MockModule())

        # Should fall back to __doc__
        assert extracted == "Fallback docstring"

    def test_extract_with_whitespace_only_instructions_returns_fallback(self):
        """Test extraction falls back when instructions is only whitespace."""
        concept = Concept(preferred_label="Test", wikibase_id=WikibaseID("Q123"))
        classifier = AutoLLMClassifier(concept=concept)

        # Create a mock module with whitespace-only instructions
        class MockModule:
            class predict:
                class signature:
                    instructions = "   \n\t  "  # Whitespace only
                    __doc__ = "Fallback docstring"

        extracted = classifier._extract_optimized_instructions(MockModule())

        # Should fall back to __doc__
        assert extracted == "Fallback docstring"

    def test_extract_with_empty_docstring_returns_default(self):
        """Test extraction returns DEFAULT_INSTRUCTIONS when both are empty."""
        concept = Concept(preferred_label="Test", wikibase_id=WikibaseID("Q123"))
        classifier = AutoLLMClassifier(concept=concept)

        # Create a mock module with empty instructions and empty __doc__
        class MockModule:
            class predict:
                class signature:
                    instructions = ""
                    __doc__ = ""

        extracted = classifier._extract_optimized_instructions(MockModule())

        # Should fall back to DEFAULT_INSTRUCTIONS
        assert extracted == DEFAULT_INSTRUCTIONS

    def test_extract_handles_attribute_error_gracefully(self):
        """Test extraction handles missing predict attribute gracefully."""
        concept = Concept(preferred_label="Test", wikibase_id=WikibaseID("Q123"))
        classifier = AutoLLMClassifier(concept=concept)

        # Create a mock module without predict attribute
        class BrokenModule:
            pass

        extracted = classifier._extract_optimized_instructions(BrokenModule())

        # Should fall back to DEFAULT_INSTRUCTIONS
        assert extracted == DEFAULT_INSTRUCTIONS


class TestFitWithMockedLLM:
    """Tests for fit() method with mocked LLM calls."""

    @pytest.fixture
    def sample_passages(self):
        """Create sample labelled passages for testing."""
        passages = []
        for i in range(20):
            if i < 10:
                # Positive examples (with spans)
                # "climate change" has length 14
                passages.append(
                    LabelledPassage(
                        id=f"pos_{i}",
                        text=f"This passage mentions climate change policy number {i}.",
                        spans=[
                            Span(
                                text="climate change",
                                start_index=0,
                                end_index=14,  # Length of "climate change"
                                concept_id=WikibaseID("Q123"),
                                labellers=["test"],
                                timestamps=[],
                            )
                        ],
                        metadata={},
                    )
                )
            else:
                # Negative examples (no spans)
                passages.append(
                    LabelledPassage(
                        id=f"neg_{i}",
                        text=f"This passage is about something unrelated number {i}.",
                        spans=[],
                        metadata={},
                    )
                )
        return passages

    def test_fit_persists_optimized_instructions_without_wandb(self, sample_passages):
        """Test that fit() persists optimized instructions when W&B is disabled."""
        from knowledge_graph.classifier.dspy_components import (
            ConceptTaggerModule,
            ConceptTaggingSignature,
        )

        concept = Concept(
            preferred_label="Climate Change",
            wikibase_id=WikibaseID("Q123"),
            labelled_passages=sample_passages,
        )
        classifier = AutoLLMClassifier(concept=concept)

        # Track original instructions state
        assert classifier.optimized_instructions is None
        assert not classifier.is_fitted

        # Create a mock optimized module that simulates MIPRO output
        expected_instructions = "Optimized instructions from MIPRO test"

        def create_mock_optimized_module(*args, **kwargs):
            """Create a mock module that looks like MIPRO output."""
            mock_module = ConceptTaggerModule(
                concept_description="Test", signature=ConceptTaggingSignature
            )
            # Simulate how MIPRO modifies the signature
            mock_module.predict.signature = (
                mock_module.predict.signature.with_instructions(expected_instructions)
            )
            return mock_module

        # Create mock MIPROv2 class that returns our mock when compile is called
        mock_optimizer = MagicMock()
        mock_optimizer.compile = MagicMock(side_effect=create_mock_optimized_module)

        # Mock dspy.configure and MIPROv2 class to avoid needing real API keys
        with (
            patch("dspy.configure"),
            patch(
                "dspy.teleprompt.MIPROv2",
                return_value=mock_optimizer,
            ),
        ):
            # Run fit without W&B
            result = classifier.fit(enable_wandb=False)

            # Verify compile was called
            assert mock_optimizer.compile.called

        # CRITICAL: Verify optimized instructions were persisted
        assert classifier.is_fitted
        assert classifier.optimized_instructions is not None
        assert classifier.optimized_instructions == expected_instructions

        # Verify system prompt was updated
        assert expected_instructions in classifier.system_prompt

        # Verify return value
        assert result is classifier

    def test_fit_handles_extraction_failure_gracefully(self, sample_passages):
        """Test that fit() handles instruction extraction failures gracefully."""
        concept = Concept(
            preferred_label="Climate Change",
            wikibase_id=WikibaseID("Q123"),
            labelled_passages=sample_passages,
        )
        classifier = AutoLLMClassifier(concept=concept)

        # Create a mock module that will cause extraction to fail
        class BrokenModule:
            """Module that causes extraction issues."""

            class predict:
                class signature:
                    # No instructions attribute, empty __doc__
                    __doc__ = None

        mock_optimizer = MagicMock()
        mock_optimizer.compile = MagicMock(return_value=BrokenModule())

        with (
            patch("dspy.configure"),
            patch("dspy.teleprompt.MIPROv2", return_value=mock_optimizer),
        ):
            # Run fit - should fall back to DEFAULT_INSTRUCTIONS
            classifier.fit(enable_wandb=False)

        assert classifier.is_fitted
        # Should fall back to DEFAULT_INSTRUCTIONS when extraction fails
        assert classifier.optimized_instructions == DEFAULT_INSTRUCTIONS

    def test_fit_with_exception_during_optimization(self, sample_passages):
        """Test that fit() handles exceptions during optimization."""
        concept = Concept(
            preferred_label="Climate Change",
            wikibase_id=WikibaseID("Q123"),
            labelled_passages=sample_passages,
        )
        classifier = AutoLLMClassifier(concept=concept)

        mock_optimizer = MagicMock()
        mock_optimizer.compile = MagicMock(side_effect=Exception("Simulated API error"))

        with (
            patch("dspy.configure"),
            patch("dspy.teleprompt.MIPROv2", return_value=mock_optimizer),
        ):
            # Run fit - should handle the exception gracefully
            result = classifier.fit(enable_wandb=False)

        assert classifier.is_fitted
        assert classifier.optimized_instructions == DEFAULT_INSTRUCTIONS
        assert result is classifier


class TestEndToEndWithMockedLLM:
    """End-to-end tests with mocked LLM endpoint to verify full flow."""

    @pytest.fixture
    def sample_passages(self):
        """Create sample labelled passages for testing."""
        passages = []
        for i in range(20):
            if i < 10:
                # Positive examples (with spans)
                passages.append(
                    LabelledPassage(
                        id=f"pos_{i}",
                        text=f"This passage mentions climate change policy number {i}.",
                        spans=[
                            Span(
                                text="climate change",
                                start_index=0,
                                end_index=14,
                                concept_id=WikibaseID("Q123"),
                                labellers=["test"],
                                timestamps=[],
                            )
                        ],
                        metadata={},
                    )
                )
            else:
                # Negative examples (no spans)
                passages.append(
                    LabelledPassage(
                        id=f"neg_{i}",
                        text=f"This passage is about something unrelated number {i}.",
                        spans=[],
                        metadata={},
                    )
                )
        return passages

    def test_full_fit_flow_with_mocked_dspy_lm(self, sample_passages):
        """
        End-to-end test of fit() with a mocked DSPy LM.

        This test mocks at the DSPy LM level to simulate what happens
        when MIPRO runs optimization and returns an optimized module.
        """

        from knowledge_graph.classifier.dspy_components import (
            ConceptTaggerModule,
            ConceptTaggingSignature,
        )

        concept = Concept(
            preferred_label="Climate Change",
            wikibase_id=WikibaseID("Q123"),
            labelled_passages=sample_passages,
        )

        classifier = AutoLLMClassifier(concept=concept)

        # Verify initial state
        assert classifier.optimized_instructions is None
        assert not classifier.is_fitted
        original_system_prompt = classifier.system_prompt

        # The expected optimized instructions that MIPRO would produce
        expected_optimized_instructions = (
            "OPTIMIZED INSTRUCTIONS FOR CLIMATE CHANGE TAGGING:\n"
            "1. Look for explicit mentions of 'climate change' or related terms\n"
            "2. Tag only direct references, not tangential mentions\n"
            "3. Ensure tags are properly balanced XML"
        )

        def mock_mipro_compile(student, trainset, valset, **kwargs):
            """
            Mock MIPRO compile that returns an optimized module.

            This simulates what MIPRO does: it takes the student module,
            runs optimization trials, and returns a module with modified
            signature instructions.
            """
            # Create a new module that looks like an optimized version
            optimized = ConceptTaggerModule(
                concept_description=student.concept_description,
                signature=ConceptTaggingSignature,
            )
            # Simulate MIPRO setting the optimized instructions
            optimized.predict.signature = optimized.predict.signature.with_instructions(
                expected_optimized_instructions
            )
            return optimized

        # Create mock optimizer
        mock_optimizer = MagicMock()
        mock_optimizer.compile = mock_mipro_compile

        # Run fit with mocked MIPRO
        with (
            patch("dspy.configure"),
            patch("dspy.teleprompt.MIPROv2", return_value=mock_optimizer),
        ):
            result = classifier.fit(enable_wandb=False)

        # Verify the classifier is now fitted
        assert classifier.is_fitted
        assert result is classifier

        # CRITICAL: Verify optimized instructions were extracted and persisted
        assert classifier.optimized_instructions is not None
        assert classifier.optimized_instructions == expected_optimized_instructions

        # Verify system prompt was updated with the optimized instructions
        assert classifier.system_prompt != original_system_prompt
        assert expected_optimized_instructions in classifier.system_prompt

        # Verify the classifier ID changed (since instructions changed)
        # The ID should be deterministic based on the instructions
        assert classifier.optimized_instructions in classifier.system_prompt

        print("\n=== Test Results ===")
        print(f"is_fitted: {classifier.is_fitted}")
        print(
            f"optimized_instructions length: {len(classifier.optimized_instructions)}"
        )
        print(
            f"system_prompt contains instructions: {expected_optimized_instructions in classifier.system_prompt}"
        )
        print(f"\nOptimized Instructions:\n{classifier.optimized_instructions}")

    def test_fit_without_wandb_persists_same_as_with_wandb(self, sample_passages):
        """
        Verify that fit() behaves the same with and without W&B.

        This test ensures there's no difference in instruction persistence
        between the W&B enabled and disabled code paths.
        """

        from knowledge_graph.classifier.dspy_components import (
            ConceptTaggerModule,
            ConceptTaggingSignature,
        )

        expected_instructions = "Test optimized instructions for comparison"

        def create_optimized_module(student, *args, **kwargs):
            optimized = ConceptTaggerModule(
                concept_description=student.concept_description,
                signature=ConceptTaggingSignature,
            )
            optimized.predict.signature = optimized.predict.signature.with_instructions(
                expected_instructions
            )
            return optimized

        # Test WITHOUT W&B
        concept_no_wandb = Concept(
            preferred_label="Test",
            wikibase_id=WikibaseID("Q123"),
            labelled_passages=sample_passages,
        )
        classifier_no_wandb = AutoLLMClassifier(concept=concept_no_wandb)

        mock_optimizer = MagicMock()
        mock_optimizer.compile = create_optimized_module

        with (
            patch("dspy.configure"),
            patch("dspy.teleprompt.MIPROv2", return_value=mock_optimizer),
        ):
            classifier_no_wandb.fit(enable_wandb=False)

        # Test WITH W&B (but mock wandb to avoid actual calls)
        concept_with_wandb = Concept(
            preferred_label="Test",
            wikibase_id=WikibaseID("Q123"),
            labelled_passages=sample_passages,
        )
        classifier_with_wandb = AutoLLMClassifier(concept=concept_with_wandb)

        mock_wandb_run = MagicMock()

        with (
            patch("dspy.configure"),
            patch("dspy.teleprompt.MIPROv2", return_value=mock_optimizer),
            patch("wandb.init", return_value=mock_wandb_run),
        ):
            classifier_with_wandb.fit(enable_wandb=True)

        # Both should have the same optimized instructions
        assert classifier_no_wandb.optimized_instructions == expected_instructions
        assert classifier_with_wandb.optimized_instructions == expected_instructions
        assert (
            classifier_no_wandb.optimized_instructions
            == classifier_with_wandb.optimized_instructions
        )

        # Both should have the optimized instructions in their system prompts
        assert expected_instructions in classifier_no_wandb.system_prompt
        assert expected_instructions in classifier_with_wandb.system_prompt

        # Both should be fitted
        assert classifier_no_wandb.is_fitted
        assert classifier_with_wandb.is_fitted

        print("\n=== Comparison Results ===")
        print(f"Without W&B instructions: {classifier_no_wandb.optimized_instructions}")
        print(f"With W&B instructions: {classifier_with_wandb.optimized_instructions}")
        print(
            f"Instructions match: {classifier_no_wandb.optimized_instructions == classifier_with_wandb.optimized_instructions}"
        )
