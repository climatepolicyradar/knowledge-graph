import asyncio
import os
from pathlib import Path
from typing import Annotated

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
from rich.console import Console
from sklearn.metrics import f1_score

from knowledge_graph.classifier import load_classifier_from_wandb
from knowledge_graph.cloud import AwsEnv, get_s3_client
from knowledge_graph.config import ensemble_metrics_dir
from knowledge_graph.ensemble import create_ensemble
from knowledge_graph.ensemble.metrics import (
    Disagreement,
    PredictionProbabilityStandardDeviation,
)
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.labelling import label_passages_with_classifier
from scripts.get_concept import get_concept_async

app = typer.Typer()
console = Console()


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

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

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

        # Calculate F1 using sklearn
        f1 = f1_score(y_true, y_pred)

        retention_rates.append(retention_rate)
        f1_scores.append(f1)

    return thresholds, retention_rates, f1_scores


async def calculate_ensemble_metrics(
    wikibase_id: WikibaseID,
    classifier_wandb_path: str,
    n_classifiers: int,
    batch_size: int,
) -> None:
    """
    Analyse classifier variants with respect to ensemble metrics.

    Saves all outputs to a subdir of the ensemble_metrics_dir defined in config.

    :param WikibaseID wikibase_id: wikibase ID of the concept used
    :param str classifier_wandb_path: W&B path to the classifier artifact
    :param int n_classifiers: number of classifier variants to create
    :param int batch_size: batch size for inference
    """

    # Load concept and ground truth
    console.log(f"Getting concept {wikibase_id} and its labelled passages")
    concept = await get_concept_async(
        wikibase_id=wikibase_id,
        include_labels_from_subconcepts=True,
        include_recursive_has_subconcept=True,
    )

    if not concept.labelled_passages:
        console.log("❌ No evaluation data found. Exiting.")
        return

    console.log("Setting up AWS credentials...")
    region_name = "eu-west-1"
    aws_env = AwsEnv.labs
    os.environ["AWS_PROFILE"] = aws_env
    get_s3_client(aws_env, region_name)

    console.log(f"Loading classifier from W&B: {classifier_wandb_path}")
    classifier = load_classifier_from_wandb(classifier_wandb_path)

    console.log(f"Creating ensemble with {n_classifiers} variants...")
    ensemble = create_ensemble(classifier, n_classifiers=n_classifiers)

    console.log(f"Running inference on {len(concept.labelled_passages)} passages...")
    predictions_per_classifier = []

    for i, variant_classifier in enumerate(ensemble.classifiers):
        console.log(f"Running variant {i + 1}/{n_classifiers}...")
        variant_predictions = label_passages_with_classifier(
            classifier=variant_classifier,
            labelled_passages=concept.labelled_passages,
            batch_size=batch_size,
            show_progress=True,
        )
        predictions_per_classifier.append(variant_predictions)

    console.log("Calculating dataset of ensemble predictions vs ground truth...")
    df = create_predictions_dataframe(
        predictions_per_classifier=predictions_per_classifier,
        ground_truth_passages=concept.labelled_passages,
    )

    console.log(f"Analyzing {len(df)} predictions...")

    classifier_id = classifier.id if hasattr(classifier, "id") else "classifier"
    output_dir = ensemble_metrics_dir / f"{wikibase_id}_classifier_{classifier_id}"
    output_dir.mkdir(exist_ok=True, parents=True)

    console.log("Creating plots...")
    create_plots(df, output_dir)

    # Save summary statistics
    console.log("Generating summary statistics...")
    summary_stats = {
        "Classifier path": classifier_wandb_path,
        "Concept": str(wikibase_id),
        "Number of variants": n_classifiers,
        "Total predictions": len(df),
        "Batch size": batch_size,
    }

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
    console.log("  - summary_stats.txt: Summary statistics")
    console.log("  - evaluation_data.csv: Raw evaluation data")


@app.command()
def main(
    wikibase_id: Annotated[
        WikibaseID,
        typer.Option(
            ...,
            help="The Wikibase ID of the concept",
            parser=WikibaseID,
        ),
    ],
    classifier_wandb_path: Annotated[
        str,
        typer.Option(
            help="Path of the classifier in W&B. E.g. 'climatepolicyradar/Q913/rsgz5ygh:v0'"
        ),
    ],
    n_classifiers: Annotated[
        int,
        typer.Option(help="Number of classifier variants to create for the ensemble"),
    ] = 5,
    batch_size: Annotated[
        int,
        typer.Option(help="Number of passages to process in each batch"),
    ] = 15,
):
    """
    Analyse classifier variants with respect to ensemble metrics.

    This script loads a primary classifier from Weights & Biases, creates variants
    at inference time, runs predictions on evaluation data, calculates uncertainty
    metrics for each prediction, and plots these against F1 score. This helps to pick
    metrics and values of these for downstream applications, like active learning.
    """

    return asyncio.run(
        calculate_ensemble_metrics(
            wikibase_id=wikibase_id,
            classifier_wandb_path=classifier_wandb_path,
            n_classifiers=n_classifiers,
            batch_size=batch_size,
        )
    )


if __name__ == "__main__":
    app()
