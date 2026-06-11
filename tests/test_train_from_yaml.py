from pathlib import Path

import pydantic
import pytest
import typer

from knowledge_graph.classifier.large_language_model import LLMClassifierPrompt
from knowledge_graph.identifiers import WikibaseID
from scripts.train import CustomClassifierConfig, _resolve_training_inputs

CONFIG = (
    Path(__file__).parents[1] / "scripts/custom_concept_training/configs/q1829.yaml"
)


def test_q1829_config_loads_and_builds_prompt():
    cfg = CustomClassifierConfig.from_yaml(CONFIG)
    assert cfg.wikibase_id == WikibaseID("Q1829")
    prompt = cfg.llm.to_classifier_kwargs()["system_prompt_template"]
    assert isinstance(prompt, LLMClassifierPrompt)
    assert (
        "{concept_description}" in prompt.system_prompt_template
    )  # bespoke template preserved
    assert "specialist analyst" in prompt.system_prompt_template
    assert prompt.labelling_guidelines is not None  # narrow str | None -> str
    assert "money or financial assets moving" in prompt.labelling_guidelines


def test_short_override_is_rejected():
    # a definition/description that fits the store (<= cap) is rejected on load
    with pytest.raises(pydantic.ValidationError) as exc:
        CustomClassifierConfig.model_validate(
            {
                "wikibase_id": "Q1829",
                "concept_overrides": {"definition": "too short"},
            }
        )
    assert "put it in Wikibase" in str(exc.value)


def test_unknown_field_is_rejected():
    # extra="forbid" (structural) catches a stray/misplaced key, e.g. a top-level definition
    with pytest.raises(pydantic.ValidationError):
        CustomClassifierConfig.model_validate(
            {
                "wikibase_id": "Q1829",
                "definition": "x",
            }
        )


def test_resolve_from_config():
    wikibase_id, classifier_kwargs, concept_overrides = _resolve_training_inputs(
        None, CONFIG, "LLMClassifier", {}, {}
    )
    assert wikibase_id == WikibaseID("Q1829")
    assert classifier_kwargs["model_name"] == "openrouter:openai/gpt-5"
    assert isinstance(classifier_kwargs["system_prompt_template"], LLMClassifierPrompt)
    assert concept_overrides == {}  # Q1829 has none — definition stays in the store


def test_resolve_plain_wikibase_id_passes_through():
    wid = WikibaseID("Q1829")
    kwargs = {"model_name": "x"}
    overrides = {"definition": "x"}
    assert _resolve_training_inputs(wid, None, "LLMClassifier", kwargs, overrides) == (
        wid,
        kwargs,
        overrides,
    )


def test_resolve_rejects_both():
    with pytest.raises(typer.BadParameter):
        _resolve_training_inputs(WikibaseID("Q1829"), CONFIG, "LLMClassifier", {}, {})


def test_resolve_rejects_neither():
    with pytest.raises(typer.BadParameter):
        _resolve_training_inputs(None, None, "LLMClassifier", {}, {})


def test_resolve_rejects_non_llm_with_config():
    with pytest.raises(typer.BadParameter):
        _resolve_training_inputs(None, CONFIG, "BertBasedClassifier", {}, {})
