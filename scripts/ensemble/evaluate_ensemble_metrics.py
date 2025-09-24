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
from typing import Annotated, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
import wandb
from rich.console import Console
from rich.progress import Progress

from knowledge_graph.ensemble.metrics import (
    Disagreement,
    PositiveRatio,
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


def calculate_ensemble_metrics_and_accuracy(
    predictions_per_classifier: list[list[LabelledPassage]],
    ground_truth_passages: list[LabelledPassage],
) -> pd.DataFrame:
    """
    Calculate ensemble metrics and per-prediction accuracy from W&B data.

    Args:
        predictions_per_classifier: List of prediction lists (one per classifier)
        ground_truth_passages: Ground truth labelled passages

    Returns DataFrame with columns:
    - passage_id: ID of the evaluation passage
    - disagreement: Disagreement metric value
    - positive_ratio: PositiveRatio metric value
    - prob_std: PredictionProbabilityStandardDeviation metric value (if available)
    - accuracy: 1 if prediction correct, 0 if incorrect
    - ground_truth_positive: True if passage has positive labels
    - predicted_positive: True if majority vote is positive
    """

    metrics = [
        Disagreement(),
        PositiveRatio(),
        PredictionProbabilityStandardDeviation(),
    ]

    results = []

    # Create a mapping from passage ID to ground truth
    ground_truth_map = {passage.id: passage for passage in ground_truth_passages}

    # Get all unique passage IDs from predictions
    all_passage_ids = set()
    for classifier_predictions in predictions_per_classifier:
        for passage in classifier_predictions:
            all_passage_ids.add(passage.id)

    console.log(
        f"Evaluating {len(all_passage_ids)} passages across {len(predictions_per_classifier)} classifiers..."
    )

    for passage_id in all_passage_ids:
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

        # Calculate ensemble metrics
        metric_values = {}
        for metric in metrics:
            try:
                metric_values[metric.name] = float(metric(spans_per_classifier))
            except (ValueError, TypeError):
                # Skip probability-based metrics if probabilities not available
                if "Probability" in metric.name:
                    metric_values[metric.name] = None
                else:
                    raise

        # Calculate accuracy by majority voting
        positive_votes = sum(1 for spans in spans_per_classifier if spans)
        majority_prediction = positive_votes > len(predictions_per_classifier) / 2

        ground_truth_positive = len(ground_truth_passage.spans) > 0

        # Accuracy: 1 if prediction matches ground truth, 0 otherwise
        accuracy = 1 if (majority_prediction == ground_truth_positive) else 0

        results.append(
            {
                "passage_id": passage_id,
                "disagreement": metric_values["Disagreement"],
                "positive_ratio": metric_values["PositiveRatio"],
                "prob_std": metric_values.get("PredictionProbabilityStandardDeviation"),
                "accuracy": accuracy,
                "ground_truth_positive": ground_truth_positive,
                "predicted_positive": majority_prediction,
            }
        )

    return pd.DataFrame(results)


def create_combined_threshold_referral_plots(
    df: pd.DataFrame, output_dir: Path
) -> None:
    """Create plots with dual x-axes: threshold values and referral rates."""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics_to_plot = [
        ("disagreement", "Disagreement", "lower"),
        ("positive_ratio", "Positive Ratio", "extreme"),
        ("prob_std", "Probability Std Dev", "lower"),
    ]

    for i, (col, title, direction) in enumerate(metrics_to_plot):
        if col == "prob_std" and df[col].isna().all():
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

        if direction == "extreme":
            # For positive ratio, handle extreme values specially
            clean_df = df.dropna(subset=[col]).copy()
            if len(clean_df) == 0:
                continue

            # Calculate distance from 0.5 (most uncertain)
            clean_df["confidence"] = np.abs(clean_df[col] - 0.5) * 2  # Scale to 0-1
            thresholds, referral_rates, f1_scores = calculate_cumulative_f1_curve(
                clean_df, "confidence", "higher"
            )
            threshold_values = (
                clean_df["confidence"]
                .quantile(np.linspace(0, 1, len(thresholds)))
                .values
            )
        else:
            thresholds, referral_rates, f1_scores = calculate_cumulative_f1_curve(
                df, col, direction
            )
            threshold_values = thresholds

        if not f1_scores:
            continue

        # Create the primary plot with threshold values on x-axis
        line = axes[i].plot(
            threshold_values,
            f1_scores,
            "b-",
            linewidth=2,
            marker="o",
            markersize=3,
            label="F1 Score",
        )

        # Create secondary x-axis for referral rates
        ax2 = axes[i].twiny()
        ax2.plot(
            referral_rates, f1_scores, alpha=0
        )  # Invisible line just to set up the axis

        # Set up the axes
        axes[i].set_xlabel(f"{title} Threshold", color="blue")
        ax2.set_xlabel("Referral Rate", color="green")
        axes[i].set_ylabel("F1 Score")

        # Color the tick labels to match the axis labels
        axes[i].tick_params(axis="x", labelcolor="blue")
        ax2.tick_params(axis="x", labelcolor="green")

        # Add baseline F1 (all predictions)
        clean_df = df.dropna(subset=[col])
        if len(clean_df) > 0:
            y_true = clean_df["ground_truth_positive"].astype(int).tolist()
            y_pred = clean_df["predicted_positive"].astype(int).tolist()

            tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
            fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
            fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            baseline_f1 = (
                2 * (precision * recall) / (precision + recall)
                if (precision + recall) > 0
                else 0
            )

            axes[i].axhline(
                y=baseline_f1,
                color="r",
                linestyle="--",
                alpha=0.7,
                label=f"Baseline F1 = {baseline_f1:.3f}",
            )

        # Find best F1 score and mark it
        if f1_scores:
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

        if direction == "lower":
            axes[i].set_title(f"F1 vs Threshold & Referral Rate\n({title})")
        elif direction == "extreme":
            axes[i].set_title(f"F1 vs Confidence & Referral Rate\n({title})")
        else:
            axes[i].set_title(f"F1 vs Threshold & Referral Rate\n({title})")

        axes[i].grid(True, alpha=0.3)
        axes[i].legend(loc="lower right")
        axes[i].set_ylim(0, 1)

        # Align the x-axis ranges for better readability
        axes[i].set_xlim(min(threshold_values), max(threshold_values))
        ax2.set_xlim(min(referral_rates), max(referral_rates))

    plt.tight_layout()
    plt.savefig(
        output_dir / "combined_threshold_referral_plots.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def calculate_cumulative_f1_curve(
    df: pd.DataFrame, metric_col: str, direction: str = "lower"
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

    # Remove NaN values
    clean_df = df.dropna(subset=[metric_col]).copy()

    if len(clean_df) == 0:
        return [], [], []

    # Get sorted thresholds (percentiles)
    thresholds = np.percentile(clean_df[metric_col], np.linspace(0, 100, 101))

    retention_rates = []
    f1_scores = []

    for threshold in thresholds:
        if direction == "lower":
            # Keep predictions with metric <= threshold (more confident)
            retained = clean_df[clean_df[metric_col] <= threshold]
        else:  # direction == "higher"
            # Keep predictions with metric >= threshold (more confident)
            retained = clean_df[clean_df[metric_col] >= threshold]

        if len(retained) == 0:
            retention_rates.append(0)
            f1_scores.append(0)
            continue

        retention_rate = len(retained) / len(clean_df)

        # Calculate F1 score for retained predictions
        # Create ground truth and predicted lists
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


async def evaluate_ensemble_metrics(
    ensemble_name: str,
    wikibase_id: WikibaseID,
    output_dir: Optional[Path] = None,
) -> None:
    """
    Evaluate ensemble metrics as predictors of classifier quality.
    """

    if output_dir is None:
        output_dir = Path(f"ensemble_evaluation_{ensemble_name}_{wikibase_id}")

    output_dir.mkdir(exist_ok=True)

    console.log(f"Getting concept {wikibase_id}")
    concept = await get_concept_async(
        wikibase_id=wikibase_id,
        include_labels_from_subconcepts=True,
        include_recursive_has_subconcept=True,
    )

    # Load ensemble runs and predictions from W&B
    console.log("Loading ensemble runs from Weights & Biases...")
    run_metadata, predictions_per_classifier = load_ensemble_runs_from_wandb(
        ensemble_name=ensemble_name,
        wikibase_id=wikibase_id,
    )

    if not predictions_per_classifier:
        console.log("❌ No predictions found for ensemble. Exiting.")
        return

    # Calculate ensemble metrics and accuracy
    console.log("Calculating ensemble metrics and accuracy...")
    df = calculate_ensemble_metrics_and_accuracy(
        predictions_per_classifier=predictions_per_classifier,
        ground_truth_passages=concept.labelled_passages,
    )

    if df.empty:
        console.log("❌ No evaluation data generated. Exiting.")
        return

    console.log(f"Analyzing {len(df)} predictions...")
    console.log(f"Overall accuracy: {df['accuracy'].mean():.3f}")

    # Generate visualization
    console.log("Creating combined threshold/referral plots...")
    create_combined_threshold_referral_plots(df, output_dir)

    # Save summary statistics
    console.log("Generating summary statistics...")
    summary_stats = {
        "Ensemble name": ensemble_name,
        "Concept": str(wikibase_id),
        "Number of classifiers": len(predictions_per_classifier),
        "Total predictions": len(df),
        "Overall accuracy": df["accuracy"].mean(),
        "Disagreement correlation with accuracy": df["disagreement"].corr(
            df["accuracy"]
        ),
        "Positive ratio correlation with accuracy": df["positive_ratio"].corr(
            df["accuracy"]
        ),
        "Prob std correlation with accuracy": df["prob_std"].corr(df["accuracy"])
        if not df["prob_std"].isna().all()
        else None,
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
        "  - combined_threshold_referral_plots.png: F1 vs threshold & referral rate"
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
    output_dir: Annotated[
        Optional[Path],
        typer.Option(help="Directory to save analysis results"),
    ] = None,
):
    """
    Evaluate ensemble metrics as predictors of classifier quality.

    This script loads ensemble runs from Weights & Biases, calculates uncertainty
    metrics for each prediction, and analyzes how these metrics correlate with
    prediction accuracy. It generates visualizations showing optimal thresholds
    for filtering uncertain predictions to maximize F1 scores.

    Example usage:
        python scripts/ensemble/evaluate_ensemble_metrics.py \\
            --ensemble-name "wztb2f9" \\
            --wikibase-id Q30819957
    """

    return asyncio.run(
        evaluate_ensemble_metrics(
            ensemble_name=ensemble_name,
            wikibase_id=wikibase_id,
            output_dir=output_dir,
        )
    )


if __name__ == "__main__":
    app()
