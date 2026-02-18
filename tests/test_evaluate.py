from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import typer
from syrupy.assertion import SnapshotAssertion

from knowledge_graph.config import wandb_model_artifact_filename
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from scripts.evaluate import (
    build_metrics_path,
    calculate_performance_metrics,
    create_gold_standard_labelled_passages,
    group_passages_by_equity_strata,
    load_classifier_local,
    log_metrics_to_wandb,
    print_metrics,
    save_metrics,
)


def test_print_metrics(capsys, metrics_df: pd.DataFrame):
    print_metrics(metrics_df)

    captured = capsys.readouterr()

    # Assert the output contains some expected formatted values
    assert "0.99" in captured.out  # F1
    assert "0.3" in captured.out  # Precision
    assert "215" in captured.out  # Support


@pytest.mark.parametrize(
    "wikibase_id",
    [
        (WikibaseID("Q123")),
        (WikibaseID("Q330")),
    ],
)
def test_build_metrics_path(wikibase_id: str, snapshot: SnapshotAssertion):
    path = build_metrics_path(wikibase_id)
    # Convert to relative path for consistent snapshot testing across environments
    # Take last 3 parts: processed/classifiers_performance/Q123.json
    relative_path = Path(*path.parts[-3:])
    assert relative_path == snapshot


def test_save_metrics(tmp_path, metrics_df: pd.DataFrame):
    wikibase_id = WikibaseID("Q123")

    # Temporarily override metrics_dir
    import scripts.evaluate as evaluate

    original_metrics_dir = evaluate.metrics_dir
    evaluate.metrics_dir = tmp_path

    try:
        # Save metrics and verify
        metrics_path = save_metrics(metrics_df, wikibase_id)
        assert metrics_path.exists()
        assert metrics_path.suffix == ".json"

        # Load and verify contents
        loaded_df = pd.read_json(metrics_path, orient="records")
        pd.testing.assert_frame_equal(metrics_df, loaded_df)
    finally:
        # Restore original metrics_dir
        evaluate.metrics_dir = original_metrics_dir


@pytest.fixture
def mock_classifier():
    with patch("scripts.evaluate.Classifier") as mock:
        yield mock


def test_load_classifier_local(mock_classifier, tmp_path):
    with patch("scripts.evaluate.classifier_dir", tmp_path):
        wikibase_id = WikibaseID("Q123")
        classifier_path = tmp_path / wikibase_id
        classifier_path.mkdir(parents=True)
        pickle_path = classifier_path / wandb_model_artifact_filename
        pickle_path.touch()  # Create an empty file

        # Call function
        load_classifier_local(wikibase_id)

        # Verify Classifier.load was called with correct path
        mock_classifier.load.assert_called_once_with(pickle_path)


def test_load_classifier_local_not_found(mock_classifier, tmp_path):
    with patch("scripts.evaluate.classifier_dir", tmp_path):
        wikibase_id = WikibaseID("Q999")
        # Don't create any files - this should trigger the FileNotFoundError path

        with pytest.raises(typer.BadParameter) as exc_info:
            load_classifier_local(wikibase_id)

        assert "Classifier for Q999 not found" in str(exc_info.value)
        assert "just train Q999" in str(exc_info.value)


def test_calculate_performance_metrics(concept, snapshot: SnapshotAssertion):
    # Create gold standard passages
    gold_standard = create_gold_standard_labelled_passages(concept.labelled_passages)
    # Use the same passages as both gold and model predictions for testing
    result = calculate_performance_metrics(gold_standard, gold_standard)
    assert result == snapshot


def test_create_gold_standard_labelled_passages(concept, snapshot: SnapshotAssertion):
    result = create_gold_standard_labelled_passages(concept.labelled_passages)
    assert result == snapshot


def test_log_metrics(metrics_df: pd.DataFrame):
    with patch("wandb.Table") as mock_wandb_table:
        mock_run = Mock()
        mock_run.summary = {}

        log_metrics_to_wandb(mock_run, metrics_df)

        mock_wandb_table.assert_called_once_with(
            data=metrics_df.values.tolist(),
            columns=metrics_df.columns.tolist(),
        )

        log_calls = mock_run.log.call_args_list
        assert len(log_calls) > 0
        assert any(["performance" in log_call[0][0] for log_call in log_calls])


def test_group_passages_by_equity_strata_raises_when_equity_column_missing_from_metadata():
    """Test that ValueError is raised when equity strata columns have no values in passages."""
    human_passages = [
        LabelledPassage(text="passage 1", spans=[], metadata={"other_field": "value1"}),
        LabelledPassage(text="passage 2", spans=[], metadata={"other_field": "value2"}),
    ]
    model_passages = [
        LabelledPassage(text="passage 1", spans=[], metadata={"other_field": "value1"}),
        LabelledPassage(text="passage 2", spans=[], metadata={"other_field": "value2"}),
    ]

    with pytest.raises(ValueError) as exc_info:
        group_passages_by_equity_strata(
            human_labelled_passages=human_passages,
            model_labelled_passages=model_passages,
            equity_strata=["missing_column"],
        )

    assert "missing_column" in str(exc_info.value)
