from pathlib import Path

import pydantic
import pytest
import typer

from knowledge_graph.classifier.large_language_model import LLMClassifierPrompt
from knowledge_graph.identifiers import WikibaseID
from scripts.train import CustomClassifierConfig, _resolve_training_inputs

CONFIG_DIR = Path(__file__).parents[1] / "scripts/custom_concept_training/configs"
CONFIG_PATHS = sorted(CONFIG_DIR.glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIG_PATHS, ids=lambda p: p.stem)
def test_config_loads_and_builds_llm_prompt(path: Path):
    """Every committed config loads and yields a valid LLM prompt."""
    cfg = CustomClassifierConfig.from_yaml(path)
    assert cfg.wikibase_id == WikibaseID(path.stem.upper())
    prompt = cfg.llm.to_classifier_kwargs()["system_prompt_template"]
    assert isinstance(prompt, LLMClassifierPrompt)
    assert "{concept_description}" in prompt.system_prompt_template


def test_short_override_is_rejected():
    """A definition/description that fits the store is rejected on load."""
    with pytest.raises(pydantic.ValidationError) as exc:
        CustomClassifierConfig.model_validate(
            {
                "wikibase_id": "Q1",
                "concept_overrides": {"definition": "too short"},
            }
        )
    assert "put it in Wikibase" in str(exc.value)


def test_unknown_field_is_rejected():
    """extra='forbid' rejects a stray/misplaced key."""
    with pytest.raises(pydantic.ValidationError):
        CustomClassifierConfig.model_validate({"wikibase_id": "Q1", "definition": "x"})


def test_resolve_from_config():
    """--from-yaml-config resolves wikibase_id + LLM kwargs from a committed config."""
    path = CONFIG_PATHS[0]
    expected = CustomClassifierConfig.from_yaml(path).wikibase_id
    wikibase_id, classifier_kwargs, _ = _resolve_training_inputs(
        None, path, "LLMClassifier", {}, {}
    )
    assert wikibase_id == expected
    assert isinstance(classifier_kwargs["system_prompt_template"], LLMClassifierPrompt)


def test_resolve_plain_wikibase_id_passes_through():
    """No --from-yaml-config: inputs pass through unchanged."""
    wid = WikibaseID("Q1")
    kwargs, overrides = {"model_name": "x"}, {"definition": "x"}
    assert _resolve_training_inputs(wid, None, "LLMClassifier", kwargs, overrides) == (
        wid,
        kwargs,
        overrides,
    )


def test_resolve_rejects_both():
    """--wikibase-id and --from-yaml-config are mutually exclusive."""
    with pytest.raises(typer.BadParameter):
        _resolve_training_inputs(
            WikibaseID("Q1"), CONFIG_PATHS[0], "LLMClassifier", {}, {}
        )


def test_resolve_rejects_neither():
    """One of --wikibase-id / --from-yaml-config is required."""
    with pytest.raises(typer.BadParameter):
        _resolve_training_inputs(None, None, "LLMClassifier", {}, {})


def test_resolve_rejects_non_llm_with_config():
    """--from-yaml-config only supports LLMClassifier for now."""
    with pytest.raises(typer.BadParameter):
        _resolve_training_inputs(None, CONFIG_PATHS[0], "BertBasedClassifier", {}, {})
