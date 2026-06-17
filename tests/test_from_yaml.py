from pathlib import Path

import pandas as pd
import pydantic
import pytest
import typer
from typer.testing import CliRunner

import scripts.sample as sample_mod
import scripts.train as train_mod
from knowledge_graph.classifier.large_language_model import LLMClassifierPrompt
from knowledge_graph.custom_classifier_config import (
    CustomClassifierConfig,
    LLMClassifierConfig,
)
from knowledge_graph.identifiers import WikibaseID
from scripts.train import resolve_config_inputs

CONFIG_DIR = Path(__file__).parents[1] / "scripts/custom_concept_training/configs"
CONFIG_PATHS = sorted(CONFIG_DIR.glob("*.yaml"))


@pytest.mark.parametrize("path", CONFIG_PATHS, ids=lambda p: p.stem)
def test_config_loads_and_builds_llm_prompt(path: Path):
    """Every committed config loads, yields a valid LLM prompt, and interpolates its related definitions."""
    cfg = CustomClassifierConfig.from_yaml(path)
    assert cfg.wikibase_id == WikibaseID(path.stem.upper())
    stub = {wid: f"DEF::{wid}" for wid in cfg.llm.related_definitions}
    prompt = cfg.llm.to_classifier_kwargs(definitions=stub)["system_prompt_template"]
    assert isinstance(prompt, LLMClassifierPrompt)
    assert "{concept_description}" in prompt.system_prompt_template
    for wid in cfg.llm.related_definitions:
        assert f"DEF::{wid}" in (prompt.labelling_guidelines or "")


def test_resolve_config_inputs_yaml_returns_customclassifierconfig():
    """--from-yaml-config loads the config and returns its wikibase_id."""
    wid, cfg = resolve_config_inputs(None, CONFIG_PATHS[0])
    assert cfg is not None and wid == cfg.wikibase_id


def test_resolve_config_inputs_plain_returns_none():
    """Plain --wikibase-id passes through with no config."""
    wid = WikibaseID("Q1")
    assert resolve_config_inputs(wid, None) == (wid, None)


def test_slot_without_declaration_is_rejected():
    """A {slot} in labelling_guidelines with no related_definitions entry fails at load."""
    with pytest.raises(pydantic.ValidationError):
        LLMClassifierConfig(model_name="x", labelling_guidelines="see {recognition}")


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


def test_model_name_required():
    """A missing llm.model_name raises a validation error."""
    with pytest.raises(pydantic.ValidationError):
        LLMClassifierConfig.model_validate({})


def test_sample_main_from_yaml_passes_config_to_run_sampling(monkeypatch):
    """--from-yaml-config forwards the YAML's wikibase_id, sampling fields, and overrides."""
    captured = {}
    monkeypatch.setattr(pd, "read_feather", lambda *_a, **_k: pd.DataFrame())
    monkeypatch.setattr(sample_mod, "run_sampling", lambda **kw: captured.update(kw))
    path = CONFIG_PATHS[0]
    cfg = CustomClassifierConfig.from_yaml(path)
    result = CliRunner().invoke(
        sample_mod.app, ["--from-yaml-config", str(path), "--no-track-and-upload"]
    )
    assert result.exit_code == 0, result.output
    assert captured["wikibase_id"] == cfg.wikibase_id
    assert captured["sample_size"] == cfg.sampling.sample_size
    assert captured["concept_overrides"] == cfg.concept_overrides.as_overrides()
    assert "concept_override" not in captured


def test_train_main_from_yaml_passes_llm_kwargs_to_run_training(monkeypatch):
    """--from-yaml-config (LLM) forwards the YAML's wikibase_id + LLM classifier kwargs."""
    captured = {}

    async def _fake_run_training(**kw):
        captured.update(kw)

    monkeypatch.setattr(train_mod, "run_training", _fake_run_training)
    path = CONFIG_PATHS[0]
    cfg = CustomClassifierConfig.from_yaml(path)
    result = CliRunner().invoke(
        train_mod.app,
        [
            "--from-yaml-config",
            str(path),
            "--classifier-type",
            "LLMClassifier",
            "--no-track-and-upload",
            "--no-evaluate",
        ],
    )
    assert result.exit_code == 0, result.output
    assert captured["wikibase_id"] == cfg.wikibase_id
    assert isinstance(
        captured["classifier_kwargs"]["system_prompt_template"], LLMClassifierPrompt
    )


def test_resolve_config_inputs_rejects_both():
    """--wikibase-id and --from-yaml-config are mutually exclusive."""
    with pytest.raises(typer.BadParameter):
        resolve_config_inputs(WikibaseID("Q1"), CONFIG_PATHS[0])


def test_resolve_config_inputs_rejects_neither():
    """One of --wikibase-id / --from-yaml-config is required."""
    with pytest.raises(typer.BadParameter):
        resolve_config_inputs(None, None)
