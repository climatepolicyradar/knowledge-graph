from unittest.mock import Mock, patch

import pandas as pd
import pytest
import typer
from syrupy.assertion import SnapshotAssertion

from knowledge_graph.config import wandb_model_artifact_filename
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.span import Span
from scripts.evaluate import (
    build_metrics_path,
    calculate_performance_metrics,
    calculate_std_by_equity_strata,
    count_annotations,
    create_gold_standard_labelled_passages,
    create_validation_predictions_dataframe,
    create_wandb_model_evaluation_charts,
    evaluate_classifier,
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
def test_build_metrics_path(wikibase_id: str):
    path = build_metrics_path(wikibase_id)
    assert path.parts[-3:] == (
        "processed",
        "classifiers_performance",
        f"{wikibase_id}.json",
    )


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

        # Second call should be for std metrics
        std_df = calculate_std_by_equity_strata(metrics_df)
        second_call = mock_wandb_table.call_args_list[1]
        assert second_call.kwargs["data"] == std_df.values.tolist()
        assert second_call.kwargs["columns"] == std_df.columns.tolist()

        log_calls = mock_run.log.call_args_list
        assert len(log_calls) > 0
        assert any(["performance" in log_call[0][0] for log_call in log_calls])


def test_group_passages_by_equity_strata_logs_warning_when_equity_column_missing_from_metadata(
    caplog,
):
    """Test that a warning is logged and only the 'all' group is returned when the equity strata column is absent from passage metadata."""
    human_passages = [
        LabelledPassage(text="passage 1", spans=[], metadata={"other_field": "value1"}),
        LabelledPassage(text="passage 2", spans=[], metadata={"other_field": "value2"}),
    ]
    model_passages = [
        LabelledPassage(text="passage 1", spans=[], metadata={"other_field": "value1"}),
        LabelledPassage(text="passage 2", spans=[], metadata={"other_field": "value2"}),
    ]

    with caplog.at_level("WARNING", logger="scripts.evaluate"):
        result = group_passages_by_equity_strata(
            human_labelled_passages=human_passages,
            model_labelled_passages=model_passages,
            equity_strata=["missing_column"],
        )

    assert "missing_column" in caplog.text
    assert len(result) == 1
    assert result[0][0] == "all"


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


def test_count_annotations_returns_zero_for_empty_list():
    assert count_annotations([]) == 0


def test_count_annotations_sums_spans_across_passages(concept):
    expected = sum(len(p.spans) for p in concept.labelled_passages)
    assert count_annotations(concept.labelled_passages) == expected


def test_evaluate_classifier_returns_metrics_dataframe(concept):
    mock_classifier = Mock()

    with patch("scripts.evaluate.label_passages_with_classifier") as mock_label:
        mock_label.side_effect = lambda clf, passages, **kwargs: list(passages)

        df, model_passages, cm = evaluate_classifier(
            classifier=mock_classifier,
            labelled_passages=concept.labelled_passages,
        )

    assert isinstance(df, pd.DataFrame)
    assert "Group" in df.columns
    assert "F1 score" in df.columns
    assert len(model_passages) == len(concept.labelled_passages)


def test_evaluate_classifier_with_empty_labelled_passages():
    mock_classifier = Mock()

    with patch("scripts.evaluate.label_passages_with_classifier") as mock_label:
        mock_label.return_value = []

        df, model_passages, cm = evaluate_classifier(
            classifier=mock_classifier,
            labelled_passages=[],
        )

    assert isinstance(df, pd.DataFrame)
    assert len(model_passages) == 0


def test_create_validation_predictions_dataframe_has_expected_columns(concept):
    gold = create_gold_standard_labelled_passages(concept.labelled_passages)
    df = create_validation_predictions_dataframe(gold, gold)

    for col in [
        "passage_id",
        "text",
        "gold_has_concept",
        "predicted_has_concept",
        "correct",
    ]:
        assert col in df.columns
    assert len(df) == len(gold)


def test_create_validation_predictions_dataframe_correct_when_predictions_match_gold(
    concept,
):
    gold = create_gold_standard_labelled_passages(concept.labelled_passages)
    df = create_validation_predictions_dataframe(gold, gold)
    assert df["correct"].all()


def test_group_passages_by_equity_strata_creates_group_per_unique_value():
    passages = [
        LabelledPassage(text="a", spans=[], metadata={"region": "EU"}),
        LabelledPassage(text="b", spans=[], metadata={"region": "NA"}),
        LabelledPassage(text="c", spans=[], metadata={"region": "EU"}),
    ]

    result = group_passages_by_equity_strata(
        human_labelled_passages=passages,
        model_labelled_passages=passages,
        equity_strata=["region"],
    )

    group_names = [g[0] for g in result]
    assert "all" in group_names
    assert "region: EU" in group_names
    assert "region: NA" in group_names


def test_group_passages_by_equity_strata_all_group_contains_every_passage():
    passages = [
        LabelledPassage(text="a", spans=[], metadata={"region": "EU"}),
        LabelledPassage(text="b", spans=[], metadata={"region": "NA"}),
    ]

    result = group_passages_by_equity_strata(
        human_labelled_passages=passages,
        model_labelled_passages=passages,
        equity_strata=["region"],
    )

    all_group = next(g for g in result if g[0] == "all")
    assert len(all_group[1]) == 2


def _make_passages_with_probability(texts, has_span: bool, prob: float):
    passages = []
    for text in texts:
        if has_span:
            span = Span(
                text=text[:5],
                start_index=0,
                end_index=min(5, len(text)),
                prediction_probability=prob,
            )
            passages.append(LabelledPassage(text=text, spans=[span]))
        else:
            passages.append(LabelledPassage(text=text, spans=[]))
    return passages


def test_create_wandb_model_evaluation_charts_does_nothing_for_empty_inputs():
    mock_run = Mock()
    create_wandb_model_evaluation_charts(
        wandb_run=mock_run,
        predictions=[],
        ground_truth=[],
    )
    mock_run.log.assert_not_called()


def test_create_wandb_model_evaluation_charts_always_logs_confusion_matrix():
    texts = [f"passage {i}" for i in range(6)]
    predictions = _make_passages_with_probability(texts[:3], has_span=True, prob=0.9)
    predictions += _make_passages_with_probability(texts[3:], has_span=False, prob=0.0)
    ground_truth = predictions[:]

    mock_run = Mock()
    mock_run.summary = {}

    with (
        patch("wandb.plot.confusion_matrix") as mock_cm,
        patch("wandb.plot.line"),
        patch("wandb.Table"),
    ):
        create_wandb_model_evaluation_charts(
            wandb_run=mock_run,
            predictions=predictions,
            ground_truth=ground_truth,
        )

    mock_cm.assert_called_once()
    assert mock_run.log.called


def test_create_wandb_model_evaluation_charts_logs_roc_and_pr_curves_when_probabilities_provided():
    texts = [f"passage {i}" for i in range(6)]
    predictions = _make_passages_with_probability(texts[:3], has_span=True, prob=0.9)
    predictions += _make_passages_with_probability(texts[3:], has_span=False, prob=0.0)
    ground_truth = predictions[:]

    mock_run = Mock()
    mock_run.summary = {}

    with (
        patch("wandb.plot.line") as mock_line,
        patch("wandb.plot.confusion_matrix"),
        patch("wandb.Table"),
    ):
        create_wandb_model_evaluation_charts(
            wandb_run=mock_run,
            predictions=predictions,
            ground_truth=ground_truth,
        )

    assert mock_line.call_count == 2
    line_titles = {call.kwargs.get("title") for call in mock_line.call_args_list}
    assert "ROC Curve" in line_titles
    assert "Precision-Recall Curve" in line_titles


def test_create_wandb_model_evaluation_charts_skips_curves_when_no_probabilities():
    texts = [f"passage {i}" for i in range(4)]
    predictions = [
        LabelledPassage(
            text=t,
            spans=[
                Span(
                    text=t[:5],
                    start_index=0,
                    end_index=min(5, len(t)),
                    prediction_probability=None,
                )
            ],
        )
        for t in texts
    ]
    ground_truth = predictions[:]

    mock_run = Mock()
    mock_run.summary = {}

    with (
        patch("wandb.plot.line") as mock_line,
        patch("wandb.plot.confusion_matrix"),
        patch("wandb.Table"),
    ):
        create_wandb_model_evaluation_charts(
            wandb_run=mock_run,
            predictions=predictions,
            ground_truth=ground_truth,
        )

    mock_line.assert_not_called()
