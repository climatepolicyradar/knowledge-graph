"""Tests for the classifier kwarg overrides exposed by the train CLI."""

from typer.testing import CliRunner

import scripts.train as train_mod
from knowledge_graph.identifiers import WikibaseID


def _capturing_run_training(monkeypatch) -> dict:
    """Replace run_training with a stub that records the kwargs it was called with."""
    captured: dict = {}

    async def _fake_run_training(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(train_mod, "run_training", _fake_run_training)
    return captured


def test_classifier_override_reaches_local_training(monkeypatch):
    """--classifier-override turns the keyword match options off for one run."""
    captured = _capturing_run_training(monkeypatch)

    result = CliRunner().invoke(
        train_mod.app,
        [
            "--wikibase-id",
            "Q123",
            "--classifier-type",
            "KeywordClassifier",
            "--classifier-override",
            "fold_subscripts=false",
            "--classifier-override",
            "match_word_forms=false",
            "--no-track-and-upload",
            "--no-evaluate",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["wikibase_id"] == WikibaseID("Q123")
    assert captured["classifier_type"] == "KeywordClassifier"
    assert captured["classifier_kwargs"] == {
        "fold_subscripts": False,
        "match_word_forms": False,
    }


def test_classifier_override_reaches_the_remote_deployment(monkeypatch):
    """The overrides must survive the hop through the Prefect deployment parameters."""
    captured: dict = {}

    def _fake_run_deployment(*, name, parameters, timeout):
        captured["name"] = name
        captured["parameters"] = parameters
        return None

    monkeypatch.setattr(train_mod, "run_deployment", _fake_run_deployment)
    monkeypatch.setattr(train_mod, "get_flow_run_ui_url", lambda flow_run: "http://x")

    result = CliRunner().invoke(
        train_mod.app,
        [
            "--wikibase-id",
            "Q123",
            "--classifier-type",
            "KeywordClassifier",
            "--classifier-override",
            "fold_subscripts=false",
            "--compute",
            "remote-cpu",
            "--no-track-and-upload",
            "--no-evaluate",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["parameters"]["classifier_type"] == "KeywordClassifier"
    assert captured["parameters"]["classifier_kwargs"] == {"fold_subscripts": False}


def test_no_override_leaves_the_classifier_defaults_in_place(monkeypatch):
    """With no override the flow gets empty kwargs, so the classifier's own defaults win."""
    captured = _capturing_run_training(monkeypatch)

    result = CliRunner().invoke(
        train_mod.app,
        [
            "--wikibase-id",
            "Q123",
            "--classifier-type",
            "KeywordClassifier",
            "--no-track-and-upload",
            "--no-evaluate",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["classifier_kwargs"] == {}
