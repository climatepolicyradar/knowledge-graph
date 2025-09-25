"""
Evaluate ensemble metrics as predictors of classifier quality.

This script analyzes how ensemble uncertainty metrics (Disagreement, PositiveRatio,
PredictionProbabilityStandardDeviation) correlate with classifier performance.
It generates F1 vs threshold & referral rate plots showing optimal thresholds
for filtering uncertain predictions to maximize F1 scores.
"""

import asyncio
import tempfile
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
import wandb
from rich.console import Console
from rich.progress import Progress

from knowledge_graph.config import ensemble_metrics_dir
from knowledge_graph.ensemble.metrics import (
    Disagreement,
    PredictionProbabilityStandardDeviation,
)
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from scripts.get_concept import get_concept_async

app = typer.Typer()
console = Console()


def load_labelled_passages_json(json_path: Path) -> list[LabelledPassage]:
    """Load labelled passages JSON."""

    with open(json_path, "r", encoding="utf-8") as f:
        labelled_passages = [LabelledPassage.model_validate_json(line) for line in f]

    return labelled_passages


def load_ensemble_runs_from_wandb(
    ensemble_name: str,
    wikibase_id: str,
) -> tuple[list[dict], list[list[LabelledPassage]]]:
    """
    Load ensemble runs and their labelled_passages artifacts from W&B.

    Returns:
        (run_metadata_list, predictions_per_classifier):
        - run_metadata_list: List of run metadata dicts
        - predictions_per_classifier: List of LabelledPassage lists (one per classifier)
    """

    console.log(
        f"Fetching runs with ensemble_name = {ensemble_name} from {wikibase_id}"
    )

    # Initialize W&B API
    api = wandb.Api()

    # Query runs with the specified ensemble name
    runs = api.runs(wikibase_id, filters={"config.ensemble_name": ensemble_name})

    run_metadata = []
    predictions_per_classifier = []

    console.log(f"Found {len(runs)} runs for ensemble {ensemble_name}")

    with Progress() as progress:
        task = progress.add_task(
            f"Loading predictions from {len(runs)} classifiers...", total=len(runs)
        )

        for run in runs:
            # Get run metadata
            metadata = {
                "run_id": run.id,
                "run_name": run.name,
                "state": run.state,
                "config": run.config,
                "summary": run.summary._json_dict,
            }
            run_metadata.append(metadata)

            # Find the labelled_passages artifact
            labelled_passages_artifact = None
            for artifact in run.logged_artifacts():
                if artifact.type == "labelled_passages":
                    labelled_passages_artifact = artifact
                    break

            if labelled_passages_artifact is None:
                console.log(f"⚠️ No labelled_passages artifact found for run {run.name}")
                predictions_per_classifier.append([])
                progress.update(task, advance=1)
                continue

            with tempfile.TemporaryDirectory() as temp_dir:
                artifact_dir = labelled_passages_artifact.download(root=temp_dir)
                json_files = list(Path(artifact_dir).glob("*.jsonl"))

                if not json_files:
                    console.log(
                        f"⚠️ No JSON files found in labelled_passages artifact for run {run.name}"
                    )
                    predictions_per_classifier.append([])
                    progress.update(task, advance=1)
                    continue

                classifier_predictions = []
                for json_file in json_files:
                    classifier_predictions += load_labelled_passages_json(json_file)

                predictions_per_classifier.append(classifier_predictions)

            progress.update(task, advance=1)

    console.log(
        f"✅ Loaded predictions from {len(predictions_per_classifier)} classifiers"
    )

    return run_metadata, predictions_per_classifier


def create_predictions_dataframe(
    predictions_per_classifier: list[list[LabelledPassage]],
    ground_truth_passages: list[LabelledPassage],
) -> pd.DataFrame:
    """
    Create dataframe of ensemble-level predictions vs ground truth.

    :param list[list[LabelledPassage]] predictions_per_classifier: One list of
        predictions per classifier
    :param list[LabelledPassage] ground_truth_passages: Ground truth labelled passages

    Returns DataFrame of ensemble classification results with columns:
    - passage_id: ID of the evaluation passage
    - disagreement: Disagreement metric value
    - prob_std: PredictionProbabilityStandardDeviation metric value (if available)
    - accuracy: 1 if prediction correct, 0 if incorrect
    - ground_truth_positive: True if passage has positive labels
    - predicted_positive: True if majority vote is positive
    """

    metrics = [
        Disagreement(),
        PredictionProbabilityStandardDeviation(),
    ]

    results = []

    ground_truth_map = {passage.id: passage for passage in ground_truth_passages}

    predicted_passage_ids = {
        passage.id
        for classifier_predictions in predictions_per_classifier
        for passage in classifier_predictions
    }

    console.log(
        f"Evaluating {len(predicted_passage_ids)} passages across {len(predictions_per_classifier)} classifiers..."
    )

    for passage_id in predicted_passage_ids:
        # Get ground truth for this passage
        if passage_id not in ground_truth_map:
            console.log(f"⚠️ No ground truth found for passage {passage_id}, skipping")
            continue

        ground_truth_passage = ground_truth_map[passage_id]

        # Get predictions from all classifiers for this passage
        spans_per_classifier = []
        for classifier_predictions in predictions_per_classifier:
            # Find this passage's predictions from this classifier
            classifier_spans = []
            for passage in classifier_predictions:
                if passage.id == passage_id:
                    classifier_spans = passage.spans
                    break
            spans_per_classifier.append(classifier_spans)

        if not spans_per_classifier:
            continue

        metric_values = {}
        for metric in metrics:
            try:
                metric_values[metric.name] = float(metric(spans_per_classifier))
            except (ValueError, TypeError):
                metric_values[metric.name] = None

        # Calculate accuracy based on ensemble's majority vote
        positive_votes = sum(1 for spans in spans_per_classifier if spans)
        majority_prediction = positive_votes > len(predictions_per_classifier) / 2

        ground_truth_positive = len(ground_truth_passage.spans) > 0
        accuracy = 1 if (majority_prediction == ground_truth_positive) else 0

        results.append(
            {
                "passage_id": passage_id,
                "disagreement": metric_values["Disagreement"],
                "prob_std": metric_values.get("PredictionProbabilityStandardDeviation"),
                "accuracy": accuracy,
                "ground_truth_positive": ground_truth_positive,
                "predicted_positive": majority_prediction,
            }
        )

    return pd.DataFrame(results)


def create_plots(predictions_df: pd.DataFrame, output_dir: Path) -> None:
    """
    Create plot for each metric of cumulative F1 score against metric value.

    :param predictions_df: dataframe of ensemble vs ground truth predictions. Output
        by `create_predictions_dataframe`.
    :param output_dir: directory to save plots (.png) to
    """

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics_to_plot = [
        ("disagreement", "Disagreement"),
        ("prob_std", "Probability Std Dev"),
    ]

    for i, (col, title) in enumerate(metrics_to_plot):
        if col == "prob_std" and bool(predictions_df[col].isna().all()):
            axes[i].text(
                0.5,
                0.5,
                "No probability data\navailable",
                ha="center",
                va="center",
                transform=axes[i].transAxes,
            )
            axes[i].set_title(title)
            continue

        thresholds, referral_rates, f1_scores = calculate_cumulative_f1_curve(
            predictions_df,
            col,
        )
        threshold_values = thresholds

        if not f1_scores:
            continue

        # Plot with F1 against threshold values on x-axis
        axes[i].plot(
            threshold_values,
            f1_scores,
            "b-",
            linewidth=2,
            marker="o",
            markersize=3,
            label="F1 Score",
        )
        axes[i].set_xlim(min(threshold_values), max(threshold_values))
        axes[i].set_xlabel(f"{title} Threshold", color="blue")
        axes[i].set_ylabel("F1 Score")
        axes[i].tick_params(axis="x", labelcolor="blue")

        # Create secondary x-axis for referral rates
        ax2 = axes[i].twiny()
        ax2.plot(referral_rates, f1_scores, alpha=0)

        ax2.set_xlim(min(referral_rates), max(referral_rates))
        ax2.set_xlabel("Referral Rate", color="green")
        ax2.tick_params(axis="x", labelcolor="green")

        # Add baseline F1 (all predictions)
        metric_df_dropna = predictions_df.dropna(subset=[col])
        if len(metric_df_dropna) > 0:
            baseline_f1 = f1_scores[-1]

            axes[i].axhline(
                y=baseline_f1,
                color="r",
                linestyle="--",
                alpha=0.7,
                label=f"Baseline F1 = {baseline_f1:.3f}",
            )

        # Find best F1 score and mark it
        best_f1_idx = np.argmax(f1_scores)
        best_threshold = threshold_values[best_f1_idx]
        best_referral_rate = referral_rates[best_f1_idx]
        best_f1 = f1_scores[best_f1_idx]

        axes[i].scatter(
            best_threshold,
            best_f1,
            color="red",
            s=100,
            zorder=5,
            label=f"Best: {best_f1:.3f} (thresh={best_threshold:.2f}, rate={best_referral_rate:.2f})",
        )

        axes[i].set_title(f"F1 vs Threshold & Referral Rate\n({title})")
        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc="lower right")
        axes[i].set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(
        output_dir / "ensemble_metric_vs_f1_vs_referral_rate_plot.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def calculate_cumulative_f1_curve(
    df: pd.DataFrame, metric_col: str
) -> tuple[list[float], list[float], list[float]]:
    """
    Calculate cumulative F1 scores for predictions below each threshold.

    Args:
        df: DataFrame with predictions and metrics
        metric_col: Column name for the uncertainty metric
        direction: "lower" means lower values = more confident, "higher" means opposite

    Returns:
        (thresholds, retention_rates, f1_scores)
    """

    _df = df.dropna(subset=[metric_col]).copy()

    if len(_df) == 0:
        return [], [], []

    # Get sorted thresholds (percentiles) for metric
    thresholds = np.percentile(_df[metric_col], np.linspace(0, 100, 101)).tolist()

    retention_rates = []
    f1_scores = []

    for threshold in thresholds:
        retained = _df[_df[metric_col] <= threshold]

        if len(retained) == 0:
            retention_rates.append(0)
            f1_scores.append(0)
            continue

        retention_rate = len(retained) / len(_df)

        y_true = retained["ground_truth_positive"].astype(int).tolist()
        y_pred = retained["predicted_positive"].astype(int).tolist()

        # Calculate confusion matrix
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

        # Calculate F1
        if tp + fp == 0:  # No positive predictions
            precision = 0
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:  # No positive ground truth
            recall = 0
        else:
            recall = tp / (tp + fn)

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        retention_rates.append(retention_rate)
        f1_scores.append(f1)

    return thresholds, retention_rates, f1_scores


async def calculate_ensemble_metrics(
    ensemble_name: str,
    wikibase_id: WikibaseID,
) -> None:
    """
    Analyse a classifier ensemble with respect to ensemble metrics.

    Saves all outputs to a subdir of the ensemble_metrics_dir defined in config.

    :param str ensemble_name: name of the ensemble stored in W&B
    :param WikibaseID wikibase_id: wikibase ID of the concept used
    """

    output_dir = ensemble_metrics_dir / f"{wikibase_id}_ensemble_{ensemble_name}"

    output_dir.mkdir(exist_ok=True)

    console.log(f"Getting concept {wikibase_id} and its labelled passages")
    concept = await get_concept_async(
        wikibase_id=wikibase_id,
        include_labels_from_subconcepts=True,
        include_recursive_has_subconcept=True,
    )

    if not concept.labelled_passages:
        console.log("❌ No evaluation data found. Exiting.")
        return

    console.log("Loading ensemble runs from Weights & Biases...")
    run_metadata, predictions_per_classifier = load_ensemble_runs_from_wandb(
        ensemble_name=ensemble_name,
        wikibase_id=wikibase_id,
    )

    if not predictions_per_classifier:
        console.log("❌ No predictions found for ensemble. Exiting.")
        return

    console.log("Calculating dataset of ensemble predictions vs ground truth...")
    df = create_predictions_dataframe(
        predictions_per_classifier=predictions_per_classifier,
        ground_truth_passages=concept.labelled_passages,
    )

    console.log(f"Analyzing {len(df)} predictions...")

    console.log("Creating plots...")
    create_plots(df, output_dir)

    # Save summary statistics
    console.log("Generating summary statistics...")
    summary_stats = {
        "Ensemble name": ensemble_name,
        "Concept": str(wikibase_id),
        "Number of classifiers": len(predictions_per_classifier),
        "Total predictions": len(df),
    }

    # Add run information
    summary_stats["Run details"] = "\n" + "\n".join(
        [
            f"  - {run['run_name']} ({run['run_id']}): {run['state']}"
            for run in run_metadata
        ]
    )

    with open(output_dir / "summary_stats.txt", "w") as f:
        for key, value in summary_stats.items():
            f.write(f"{key}: {value}\n")

    # Save the evaluation DataFrame for further analysis
    df.to_csv(output_dir / "evaluation_data.csv", index=False)

    console.log(f"✅ Analysis complete! Results saved to {output_dir}")
    console.log("Generated files:")
    console.log(
        "  - ensemble_metric_vs_f1_vs_referral_rate_plot.png: F1 vs threshold & referral rate"
    )
    console.log("  - summary_stats.txt: Summary statistics and run details")
    console.log("  - evaluation_data.csv: Raw evaluation data")


@app.command()
def main(
    ensemble_name: Annotated[
        str,
        typer.Option(
            help="Name/ID of the trained ensemble to evaluate (from W&B config.ensemble_name)"
        ),
    ],
    wikibase_id: Annotated[
        WikibaseID,
        typer.Option(
            ...,
            help="The Wikibase ID of the concept",
            parser=WikibaseID,
        ),
    ],
):
    """
    Analyse a classifier ensemble with respect to ensemble metrics.

    This script loads ensemble runs from Weights & Biases, calculates uncertainty
    metrics for each prediction, and plots these against F1 score. This helps to pick
    metrics and values of these for downstream applications, like active learning.
    """

    return asyncio.run(
        calculate_ensemble_metrics(
            ensemble_name=ensemble_name,
            wikibase_id=wikibase_id,
        )
    )


if __name__ == "__main__":
    app()
