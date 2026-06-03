from unittest.mock import patch

import pytest

from knowledge_graph.classifier.large_language_model import LLMClassifier
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import WikibaseID


@pytest.fixture(autouse=True)
def _openrouter_api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "dummy")


@pytest.fixture
def concept() -> Concept:
    return Concept(wikibase_id=WikibaseID("Q123"), preferred_label="test")


@pytest.mark.xdist_group(name="classifier")
def test_temperature_defaults_to_zero_when_none(concept: Concept):
    """temperature=None preserves the prior deterministic behaviour (T=0)."""
    assert LLMClassifier(concept, temperature=None).temperature == 0.0

    assert LLMClassifier(concept).temperature == 0.0


@pytest.mark.xdist_group(name="classifier")
def test_explicit_temperature_is_preserved(concept: Concept):
    assert LLMClassifier(concept, temperature=0.5).temperature == 0.5


@pytest.mark.xdist_group(name="classifier")
def test_temperature_is_part_of_classifier_id(concept: Concept):
    """Classifiers differing only in temperature must not collide on id/hash."""
    cold = LLMClassifier(concept, temperature=0.0)
    warm = LLMClassifier(concept, temperature=0.7)
    assert cold.id != warm.id
    assert hash(cold) != hash(warm)

    assert LLMClassifier(concept, temperature=0.5).id == (
        LLMClassifier(concept, temperature=0.5).id
    )


@pytest.mark.xdist_group(name="classifier")
def test_get_variant_floors_temperature_to_0_7(concept: Concept):
    """An ensemble variant needs T>=0.7 to produce genuine disagreement."""
    variant = LLMClassifier(concept, temperature=0.0).get_variant()
    assert variant.temperature == 0.7


@pytest.mark.xdist_group(name="classifier")
def test_get_variant_preserves_temperature_above_floor(concept: Concept):
    variant = LLMClassifier(concept, temperature=0.9).get_variant()
    assert variant.temperature == 0.9


@pytest.mark.xdist_group(name="classifier")
def test_get_variant_warns_when_below_floor(concept: Concept):
    # Patch the module's logger directly rather than relying on caplog: in some
    # contexts get_logger() returns a Prefect run logger that does not propagate
    # to caplog's root handler.
    with patch(
        "knowledge_graph.classifier.large_language_model.get_logger"
    ) as mock_get_logger:
        LLMClassifier(concept, temperature=0.0).get_variant()
    warnings = [str(call.args[0]) for call in mock_get_logger().warning.call_args_list]
    assert any("below 0.7" in message for message in warnings)


@pytest.mark.xdist_group(name="classifier")
def test_zero_random_seed_is_preserved(concept: Concept):
    """random_seed=0 is a valid seed and must not be coerced to the 42 default."""
    assert LLMClassifier(concept, random_seed=0).random_seed == 0
