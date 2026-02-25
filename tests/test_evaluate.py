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
    calculate_std_by_equity_strata,
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

        # wandb.Table is called twice: once for main metrics, once for std metrics
        assert mock_wandb_table.call_count == 2

        # First call should be for main metrics
        first_call = mock_wandb_table.call_args_list[0]
        assert first_call.kwargs["data"] == metrics_df.values.tolist()
        assert first_call.kwargs["columns"] == metrics_df.columns.tolist()

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


def test_calculate_std_by_equity_strata(metrics_df: pd.DataFrame):
    """Test that calculate_std_by_equity_strata returns correct structure."""
    result = calculate_std_by_equity_strata(metrics_df)

    # Check that result is a DataFrame with expected columns
    expected_columns = [
        "Equity strata",
        "Agreement at",
        "Precision std",
        "Recall std",
        "Accuracy std",
        "F1 score std",
    ]
    assert list(result.columns) == expected_columns

    # Check that we have results for each equity strata in the fixture
    # The fixture has: translated, world_bank_region, dataset_name
    equity_strata = result["Equity strata"].unique()
    assert "translated" in equity_strata
    assert "world_bank_region" in equity_strata
    assert "dataset_name" in equity_strata


def test_log_metrics_to_wandb_logs_std_metrics(metrics_df: pd.DataFrame):
    """Test that log_metrics_to_wandb logs std metrics to wandb."""
    with patch("wandb.Table") as mock_wandb_table:
        mock_run = Mock()
        mock_run.summary = {}

        log_metrics_to_wandb(mock_run, metrics_df)

        # Check that wandb.Table was called at least twice
        # (once for main metrics, once for std metrics)
        assert mock_wandb_table.call_count >= 2

        # Check that std metrics were logged
        log_calls = mock_run.log.call_args_list
        std_payload_calls = [
            call for call in log_calls if "std_by_equity_strata" in str(call)
        ]
        assert len(std_payload_calls) > 0

        # Check that summary includes std metrics
        summary_keys = list(mock_run.summary.keys())
        std_summary_keys = [k for k in summary_keys if k.startswith("std_")]
        assert len(std_summary_keys) > 0
