from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import typer
from syrupy.assertion import SnapshotAssertion
from wandb.wandb_run import Run

from scripts.evaluate import (
    Source,
    build_metrics_path,
    calculate_performance_metrics,
    create_gold_standard_labelled_passages,
    load_classifier_local,
    load_classifier_remote,
    log_metrics,
    print_metrics,
    save_metrics,
    validate_args,
    validate_local_args,
    validate_remote_args,
)
from src.identifiers import WikibaseID
from src.version import Version


def test_print_metrics(capsys, metrics_df: pd.DataFrame):
    print_metrics(metrics_df)

    captured = capsys.readouterr()

    # Assert the output contains some expected formatted values
    assert "0.99" in captured.out  # F1
    assert "0.3" in captured.out  # Precision
    assert "215" in captured.out  # Support


@pytest.mark.parametrize(
    "classifier,version,expected_error",
    [
        (None, None, None),
        (
            "TestClassifier",
            None,
            "classifier and version should not be specified",
        ),
        (
            None,
            Version("v1"),
            "classifier and version should not be specified",
        ),
        (
            "TestClassifier",
            Version("v1"),
            "classifier and version should not be specified",
        ),
    ],
)
def test_validate_local_args(classifier, version, expected_error):
    if expected_error is None:
        validate_local_args(classifier, version)  # Should not raise
    else:
        with pytest.raises(typer.BadParameter) as excinfo:
            validate_local_args(classifier, version)
        assert expected_error in str(excinfo.value)


@pytest.mark.parametrize(
    "track,classifier,version,expected_error",
    [
        (True, "TestClassifier", Version("v1"), None),
        (False, None, None, None),
        (
            False,
            "TestClassifier",
            None,
            "script was told not to track",
        ),
        (
            False,
            None,
            Version("v1"),
            "script was told not to track",
        ),
        (
            True,
            None,
            Version("v1"),
            "without a classifier name",
        ),
        (
            True,
            "TestClassifier",
            None,
            "without a version",
        ),
    ],
)
def test_validate_remote_args(track, classifier, version, expected_error):
    if expected_error is None:
        validate_remote_args(track, classifier, version)  # Should not raise
    else:
        with pytest.raises(typer.BadParameter) as excinfo:
            validate_remote_args(track, classifier, version)
        assert expected_error in str(excinfo.value)


@pytest.mark.parametrize(
    "track,classifier,version,expected_exception",
    [
        (True, None, None, None),
        (True, None, Version("v1"), pytest.raises(typer.BadParameter)),
        (True, "TestClassifier", None, pytest.raises(typer.BadParameter)),
        (True, "TestClassifier", Version("v1"), None),
        (False, None, Version("v1"), pytest.raises(typer.BadParameter)),
        (False, "TestClassifier", None, pytest.raises(typer.BadParameter)),
        (False, None, None, None),
        (False, "TestClassifier", Version("v1"), pytest.raises(typer.BadParameter)),
    ],
)
def test_validate_args(
    track: bool,
    classifier: str,
    version: Version,
    expected_exception: Exception,
    snapshot: SnapshotAssertion,
):
    source = Source.REMOTE if classifier and version else Source.LOCAL
    if expected_exception:
        # Instead of using the `match` arg on `pytest.raises`, this gives
        # a roundabout way for the snapshot to be used.
        with expected_exception as excinfo:
            validate_args(track, classifier, version, source)
        assert str(excinfo.value) == snapshot
    else:
        result = validate_args(track, classifier, version, source)
        assert result == snapshot


@pytest.mark.parametrize(
    "wikibase_id",
    [
        (WikibaseID("Q123")),
        (WikibaseID("Q330")),
    ],
)
def test_build_metrics_path(wikibase_id: str, snapshot: SnapshotAssertion):
    path = build_metrics_path(wikibase_id)
    # Convert to relative path for consistent snapshot testing across
    # environments
    relative_path = Path(
        *path.parts[-3:]
    )  # Take last 3 parts: classifiers_performance/Q123.json
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


def test_load_classifier_local(mock_classifier):
    with patch("scripts.evaluate.classifier_dir", Path("classifiers")):
        wikibase_id = WikibaseID("Q123")
        expected_path = Path("classifiers") / wikibase_id

        # Call function
        load_classifier_local(wikibase_id)

        # Verify Classifier.load was called with correct path
        mock_classifier.load.assert_called_once_with(expected_path)


def test_load_classifier_local_not_found(mock_classifier):
    with patch("scripts.evaluate.classifier_dir", Path("classifiers")):
        wikibase_id = WikibaseID("Q999")
        mock_classifier.load.side_effect = FileNotFoundError()

        with pytest.raises(typer.BadParameter) as exc_info:
            load_classifier_local(wikibase_id)

    assert "Classifier for Q999 not found" in str(exc_info.value)
    assert "just train Q999" in str(exc_info.value)


def test_load_classifier_remote(mock_classifier):
    # Setup
    run = Mock(spec=Run)
    run.config = {}  # Add dict for config
    classifier_name = "TestClassifier"
    version = Version("v1")
    wikibase_id = WikibaseID("Q123")

    # Mock the artifact
    mock_artifact = Mock()
    mock_artifact.metadata = {"aws_env": "test-env"}
    mock_artifact.download.return_value = "downloaded_path"
    run.use_artifact.return_value = mock_artifact

    # Call function
    result = load_classifier_remote(run, classifier_name, version, wikibase_id)

    # Verify artifact was retrieved correctly
    run.use_artifact.assert_called_once_with("Q123/TestClassifier:v1", type="model")

    # Verify classifier was loaded from downloaded path
    mock_classifier.load.assert_called_once_with(Path("downloaded_path/model.pickle"))

    # Verify AWS environment was set
    assert run.config["aws_env"] == "test-env"
    assert result == mock_classifier.load.return_value


def test_calculate_performance_metrics(concept, snapshot: SnapshotAssertion):
    # Create gold standard passages
    gold_standard = create_gold_standard_labelled_passages(concept)
    # Use the same passages as both gold and model predictions for testing
    result = calculate_performance_metrics(gold_standard, gold_standard)
    assert result == snapshot


def test_create_gold_standard_labelled_passages(concept, snapshot: SnapshotAssertion):
    result = create_gold_standard_labelled_passages(concept)
    assert result == snapshot


def test_log_metrics(metrics_df: pd.DataFrame):
    with patch("wandb.Table") as mock_wandb_table:
        mock_run = Mock()

        log_metrics(mock_run, metrics_df)

        mock_wandb_table.assert_called_once_with(
            data=metrics_df.values.tolist(),
            columns=metrics_df.columns.tolist(),
        )

        mock_run.log.assert_called_once()
        log_call = mock_run.log.call_args[0][0]
        assert "performance" in log_call
        assert log_call["performance"] == mock_wandb_table.return_value
