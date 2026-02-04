import asyncio
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Optional

import pandas as pd
import typer
import wandb
from rich import box
from rich.console import Console
from rich.table import Table
from sklearn.metrics import precision_recall_curve, roc_curve
from wandb.wandb_run import Run

from knowledge_graph.classifier import Classifier
from knowledge_graph.cloud import AwsEnv, Namespace, parse_aws_env
from knowledge_graph.concept import Concept
from knowledge_graph.config import (
    classifier_dir,
    concept_dir,
    equity_columns,
    metrics_dir,
)
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.labelling import label_passages_with_classifier
from knowledge_graph.metrics import (
    ConfusionMatrix,
    count_passage_level_metrics,
    count_span_level_metrics,
)
from knowledge_graph.span import Span, group_overlapping_spans
from knowledge_graph.wandb_helpers import load_classifier_from_wandb
from scripts.get_concept import get_concept_async

console = Console()


def load_concept(wikibase_id: WikibaseID) -> Concept:
    """Load a concept from local storage by its Wikibase ID."""
    try:
        concept = Concept.load(concept_dir / f"{wikibase_id}.json")
        return concept
    except FileNotFoundError as e:
        raise typer.BadParameter(
            f"Data for {wikibase_id} not found. \n"
            "If you haven't already, you should run:\n"
            f"  just get-concept {wikibase_id}\n"
        ) from e


def load_classifier_local(wikibase_id: WikibaseID) -> Classifier:
    """Load a classifier from local storage by its Wikibase ID."""
    classifier_path = classifier_dir / wikibase_id
    if not classifier_path.exists() or not list(classifier_path.glob("**/*.pickle")):
        raise typer.BadParameter(
            f"Classifier for {wikibase_id} not found. \n"
            "If you haven't already, you should run:\n"
            f"  just train {wikibase_id}\n"
        )

    try:
        most_recent_classifier_path = max(
            classifier_path.glob("**/*.pickle"), key=os.path.getctime
        )
        return Classifier.load(most_recent_classifier_path)
    except (FileNotFoundError, ValueError) as e:
        raise typer.BadParameter(
            f"Classifier for {wikibase_id} not found. \n"
            "If you haven't already, you should run:\n"
            f"  just train {wikibase_id}\n"
        ) from e


def create_gold_standard_labelled_passages(
    labelled_passages: list[LabelledPassage],
) -> list[LabelledPassage]:
    """
    Create gold standard labelled passages.

    This is done from a concept by merging overlapping spans.
    """
    gold_standard_labelled_passages: list[LabelledPassage] = []
    for labelled_passage in labelled_passages:
        merged_spans = []
        for group in group_overlapping_spans(
            spans=labelled_passage.spans, jaccard_threshold=0
        ):
            merged_span = Span.union(spans=group)
            merged_span.labellers = ["gold standard"]
            merged_span.timestamps = [datetime.now()]
            merged_spans.append(merged_span)

        gold_standard_labelled_passages.append(
            labelled_passage.model_copy(
                update={"spans": merged_spans},
                deep=True,
            )
        )

    return gold_standard_labelled_passages


def count_annotations(labelled_passages: list[LabelledPassage]) -> int:
    """Count the total number of span annotations."""
    return sum([len(entry.spans) for entry in labelled_passages])


def calculate_performance_metrics(
    gold_standard_labelled_passages: list[LabelledPassage],
    model_labelled_passages: list[LabelledPassage],
) -> list[Any]:
    """
    Calculate performance metrics for predictions against gold standard.

    Computes confusion matrices for both passage-level and span-level
    metrics, with span-level metrics calculated at multiple Jaccard
    similarity thresholds. Results are grouped by equity strata to
    enable fairness analysis.
    """
    confusion_matrices: dict[str, dict[str, ConfusionMatrix]] = defaultdict(dict)
    for (
        group,
        gold_standard_labelled_passages,
        model_labelled_passages,
    ) in group_passages_by_equity_strata(
        human_labelled_passages=gold_standard_labelled_passages,
        model_labelled_passages=model_labelled_passages,
        equity_strata=equity_columns,
    ):
        confusion_matrices[group]["Passage level"] = count_passage_level_metrics(
            gold_standard_labelled_passages, model_labelled_passages
        )

        # calculate span-level metrics at different thresholds. The
        # thresholds define the minimum Jaccard similarity required
        # for two spans to be considered a match. We take a range of
        # thresholds to get a sense of how the model performs at
        # different levels of agreement (with 0 allowing for any
        # overlap between model and human, and 1 setting a requirement
        # for an exact match)
        span_level_agreement_thresholds = [0, 0.5, 0.9, 0.99]
        for threshold in span_level_agreement_thresholds:
            confusion_matrices[group][f"Span level ({threshold})"] = (
                count_span_level_metrics(
                    gold_standard_labelled_passages,
                    model_labelled_passages,
                    threshold=threshold,
                )
            )

    metrics = []
    # Sort groups for deterministic ordering
    for group in sorted(confusion_matrices.keys()):
        results = confusion_matrices[group]
        # Sort agreement levels for deterministic ordering
        for agreement_level in sorted(results.keys()):
            confusion_matrix = results[agreement_level]
            metrics.append(
                {
                    "Group": group,
                    "Agreement at": agreement_level,
                    "Precision": confusion_matrix.precision(),
                    "Recall": confusion_matrix.recall(),
                    "Accuracy": confusion_matrix.accuracy(),
                    "F1 score": confusion_matrix.f1_score(),
                    "Support": confusion_matrix.support(),
                },
            )

    return metrics


def print_metrics(df) -> None:
    """Print metrics DataFrame as a formatted table to the console."""
    table = Table(box=box.SIMPLE, show_header=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        formatted_row = [
            f"{value:.2f}" if isinstance(value, float) else str(value) for value in row
        ]
        table.add_row(*formatted_row)

    console.log(table)


def build_metrics_path(wikibase_id: WikibaseID) -> Path:
    """Build the file path for storing metrics for a given Wikibase ID."""
    return metrics_dir / f"{wikibase_id}.json"


def save_metrics(df: pd.DataFrame, wikibase_id: WikibaseID) -> Path:
    """Save metrics DataFrame to a JSON file and return the file path."""
    metrics_path = build_metrics_path(wikibase_id)
    df.to_json(metrics_path, orient="records", indent=2)
    return metrics_path


def group_passages_by_equity_strata(
    human_labelled_passages: list[LabelledPassage],
    model_labelled_passages: list[LabelledPassage],
    equity_strata: list[str],
) -> list[tuple[str, list[LabelledPassage], list[LabelledPassage]]]:
    """
    Group passages by their equity strata metadata values.

    Creates groups of passages based on their equity strata metadata fields.
    Returns both human and model labelled passages for each stratum value,
    including an 'all' group containing all passages.
    """
    groups = [("all", human_labelled_passages, model_labelled_passages)]

    # get the unique values for each equity strata from the labelled
    # passages' metadata
    equity_strata_values = {
        equity_stratum: set(
            passage.metadata.get(equity_stratum, "")
            for passage in human_labelled_passages
        )
        for equity_stratum in equity_strata
    }

    # group the passages according to their values
    for equity_stratum, values in equity_strata_values.items():
        for value in values:
            human_labelled_passages_group = [
                passage
                for passage in human_labelled_passages
                if passage.metadata.get(equity_stratum, "") == value
            ]
            model_labelled_passages_group = [
                passage
                for passage in model_labelled_passages
                if passage.metadata.get(equity_stratum, "") == value
            ]
            groups.append(
                (
                    f"{equity_stratum}: {value}",
                    human_labelled_passages_group,
                    model_labelled_passages_group,
                )
            )

    return groups


def log_metrics_to_wandb(
    run: Run,
    df: pd.DataFrame,
):
    """Log metrics to weights and biases."""
    table = wandb.Table(data=df.values.tolist(), columns=df.columns.tolist())
    run.log({"performance": table})

    for _, row in df.reset_index(drop=True).iterrows():
        group = str(row["Group"])
        agreement = str(row["Agreement at"])
        metrics_payload = {
            f"metrics/{group}/{agreement}/precision": float(row["Precision"]),
            f"metrics/{group}/{agreement}/recall": float(row["Recall"]),
            f"metrics/{group}/{agreement}/accuracy": float(row["Accuracy"]),
            f"metrics/{group}/{agreement}/f1": float(row["F1 score"]),
            f"metrics/{group}/{agreement}/support": float(row["Support"]),
        }
        run.log(metrics_payload)

    summary_row = df[(df["Group"] == "all") & (df["Agreement at"] == "Passage level")]
    if not summary_row.empty:
        summary_row = summary_row.iloc[0]
        summary_metrics = {
            "passage_level_precision": float(summary_row["Precision"]),
            "passage_level_recall": float(summary_row["Recall"]),
            "passage_level_f1": float(summary_row["F1 score"]),
            "passage_level_accuracy": float(summary_row["Accuracy"]),
            "passage_level_support": float(summary_row["Support"]),
        }
    else:
        summary_metrics = {}

    for metric_name, metric_value in summary_metrics.items():
        run.summary[metric_name] = metric_value


def create_validation_predictions_dataframe(
    gold_standard_labelled_passages: list[LabelledPassage],
    model_labelled_passages: list[LabelledPassage],
) -> pd.DataFrame:
    """
    Create a DataFrame of validation set predictions.

    :param gold_standard_labelled_passages: Gold standard labelled passages
    :param model_labelled_passages: Model predicted labelled passages
    :return: DataFrame with passage-level predictions and metadata
    """
    table_data = []
    for gold_passage, model_passage in zip(
        gold_standard_labelled_passages, model_labelled_passages
    ):
        text = gold_passage.text
        passage_id = gold_passage.id

        gold_spans_text = [
            text[span.start_index : span.end_index] for span in gold_passage.spans
        ]
        gold_has_spans = len(gold_passage.spans) > 0

        model_spans_text = [
            text[span.start_index : span.end_index] for span in model_passage.spans
        ]
        model_has_spans = len(model_passage.spans) > 0

        passage_level_correct = gold_has_spans == model_has_spans

        metadata = gold_passage.metadata or {}

        row = {
            "passage_id": passage_id,
            "text": text,
            "gold_has_concept": gold_has_spans,
            "predicted_has_concept": model_has_spans,
            "correct": passage_level_correct,
            "gold_span_count": len(gold_passage.spans),
            "predicted_span_count": len(model_passage.spans),
            "gold_spans": "|".join(gold_spans_text) if gold_spans_text else "",
            "predicted_spans": "|".join(model_spans_text) if model_spans_text else "",
        }

        for equity_column in equity_columns:
            if equity_column in metadata:
                row[equity_column] = metadata[equity_column]

        table_data.append(row)

    return pd.DataFrame(table_data)


def log_validation_set_predictions_to_wandb(
    run: Run,
    gold_standard_labelled_passages: list[LabelledPassage],
    model_labelled_passages: list[LabelledPassage],
):
    """Log individual validation set predictions to W&B as a table."""
    df = create_validation_predictions_dataframe(
        gold_standard_labelled_passages, model_labelled_passages
    )
    predictions_df = wandb.Table(dataframe=df)
    run.log({"validation_set_predictions": predictions_df})


def evaluate_classifier(
    classifier: Classifier,
    labelled_passages: list[LabelledPassage],
    wandb_run: Optional[Run] = None,
    batch_size: int = 16,
) -> tuple[pd.DataFrame, list[LabelledPassage], ConfusionMatrix]:
    """
    Evaluate the performance of a classifier using an evaluation dataset.

    :param Classifier classifier: classifier to evaluate
    :param list[LabelledPassage] labelled_passages: labelled passages, as pulled from
        Argilla, to evaluate the classifier against
    :return tuple[pd.DataFrame, list[LabelledPassage], ConfusionMatrix]: dataframe of
        metrics, list of passages labelled by the model, and passage-level confusion
        matrix (for computing f-beta scores with arbitrary beta values)
    """

    console.log("🥇 Creating a list of gold-standard labelled passages")
    gold_standard_labelled_passages = create_gold_standard_labelled_passages(
        labelled_passages
    )
    n_annotations = count_annotations(gold_standard_labelled_passages)
    console.log(
        f"🚚 Loaded {len(gold_standard_labelled_passages)} labelled passages "
        f"with {n_annotations} individual annotations"
    )

    if wandb_run:
        wandb_run.config["n_gold_standard_labelled_passages"] = len(  # type: ignore
            gold_standard_labelled_passages
        )
        wandb_run.config["n_annotations"] = n_annotations  # type: ignore

    console.log("🤖 Labelling passages with the classifier")
    model_labelled_passages = label_passages_with_classifier(
        classifier,
        gold_standard_labelled_passages,  # type: ignore
        batch_size=batch_size,
        show_progress=True,
    )
    n_annotations = count_annotations(model_labelled_passages)
    console.log(
        f"✅ Labelled {len(model_labelled_passages)} passages "
        f"with {n_annotations} individual annotations"
    )
    if wandb_run:
        wandb_run.config["n_model_labelled_passages"] = len(model_labelled_passages)  # type: ignore

    console.log(f"📊 Calculating performance metrics for {classifier.concept}")

    metrics = calculate_performance_metrics(
        gold_standard_labelled_passages, model_labelled_passages
    )

    df = pd.DataFrame(metrics)

    print_metrics(df)

    passage_level_cm = count_passage_level_metrics(
        gold_standard_labelled_passages, model_labelled_passages
    )

    if wandb_run:
        log_metrics_to_wandb(wandb_run, df)  # type: ignore
        console.log("📊 Logging validation set predictions to W&B")
        log_validation_set_predictions_to_wandb(
            wandb_run,
            gold_standard_labelled_passages,
            model_labelled_passages,
        )
        create_wandb_model_evaluation_charts(
            wandb_run,
            predictions=model_labelled_passages,
            ground_truth=gold_standard_labelled_passages,
        )

    return df, model_labelled_passages, passage_level_cm


def create_wandb_model_evaluation_charts(
    wandb_run: Run,
    predictions: list[LabelledPassage],
    ground_truth: list[LabelledPassage],
) -> None:
    """
    Plot ROC, precision-recall and confusion matrix plots in the W&B run.

    The first two are only plotted if predictions have probabilities
    """

    ground_truth_labels = [1 if lp.spans else 0 for lp in ground_truth]
    binary_predictions = [1 if lp.spans else 0 for lp in predictions]

    if all(
        [
            span.prediction_probability is not None
            for pred in predictions
            for span in pred.spans
        ]
    ):
        pred_probabilities = [
            max(
                [
                    span.prediction_probability
                    for span in pred.spans
                    if span.prediction_probability is not None
                ]
                or [0.0]
            )
            for pred in predictions
        ]

        precision, recall, pr_thresholds = precision_recall_curve(
            ground_truth_labels, pred_probabilities
        )
        fpr, tpr, roc_thresholds = roc_curve(ground_truth_labels, pred_probabilities)

        # Find optimal threshold using ROC (maximise TPR-FPR)
        optimal_idx_roc = (tpr - fpr).argmax()
        optimal_threshold_roc = float(roc_thresholds[optimal_idx_roc])

        # Find optimal threshold using F1 score
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx_f1 = f1_scores.argmax()
        optimal_threshold_f1 = (
            float(pr_thresholds[optimal_idx_f1])
            if optimal_idx_f1 < len(pr_thresholds)
            else 0.5
        )

        # Log threshold recommendations and metrics. These are logged alphabetically
        # in the run summary.
        threshold_recommendations = {
            "optimal_ROC_threshold": optimal_threshold_roc,
            "optimal_ROC_threshold_tpr": float(tpr[optimal_idx_roc]),
            "optimal_ROC_threshold_fpr": float(fpr[optimal_idx_roc]),
            "optimal_f1_threshold": optimal_threshold_f1,
            "optimal_f1_threshold_precision": float(precision[optimal_idx_f1]),
            "optimal_f1_threshold_recall": float(recall[optimal_idx_f1]),
            "optimal_f1_threshold_f1_score": float(f1_scores[optimal_idx_f1]),
        }
        for k, v in threshold_recommendations.items():
            wandb_run.summary[k] = v

        Console().log(
            f"[bold]Optimal threshold (ROC):[/bold] {optimal_threshold_roc:.4f} (TPR: {tpr[optimal_idx_roc]:.4f}, FPR: {fpr[optimal_idx_roc]:.4f})"
        )
        Console().log(
            f"[bold]Optimal threshold (F1):[/bold] {optimal_threshold_f1:.4f} (Precision: {precision[optimal_idx_f1]:.4f}, Recall: {recall[optimal_idx_f1]:.4f}, F1: {f1_scores[optimal_idx_f1]:.4f})"
        )

        pr_data = [
            [r, p, t]
            for r, p, t in zip(
                recall.tolist(), precision.tolist(), pr_thresholds.tolist()
            )
        ]
        pr_table = wandb.Table(
            data=pr_data, columns=["recall", "precision", "threshold"]
        )

        wandb_run.log(
            {
                "precision-recall-curve": wandb.plot.line(
                    pr_table, "recall", "precision", title="Precision-Recall Curve"
                )
            }
        )

        roc_data = [
            [f, t, th]
            for f, t, th in zip(fpr.tolist(), tpr.tolist(), roc_thresholds.tolist())
        ]
        roc_table = wandb.Table(data=roc_data, columns=["fpr", "tpr", "threshold"])

        wandb_run.log(
            {"roc-curve": wandb.plot.line(roc_table, "fpr", "tpr", title="ROC Curve")}
        )
    else:
        Console().log(
            "Skipping ROC and precision-recall plots because classifier predictions don't have probabilities."
        )

    wandb_run.log(
        {
            "confusion_matrix": wandb.plot.confusion_matrix(
                y_true=ground_truth_labels,
                preds=binary_predictions,
                class_names=["false", "true"],
                title="Confusion Matrix",
            )
        }
    )


app = typer.Typer()


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
    wandb_model_path: Annotated[
        Optional[str],
        typer.Option(
            help="W&B model artifact path (e.g. 'climatepolicyradar/Q1829/abcdefg:v0'). "
            "If not provided, loads the most recent local classifier.",
        ),
    ] = None,
    aws_env: Annotated[
        AwsEnv,
        typer.Option(
            help="AWS environment to evaluate the model artifact within",
            parser=parse_aws_env,
        ),
    ] = AwsEnv.labs,
    track: Annotated[
        bool,
        typer.Option(
            help="Whether to track the evaluation run in Weights & Biases",
        ),
    ] = True,
):
    """Evaluate a classifier against its validation dataset."""
    console.log("🚀 Starting model evaluation")

    if os.environ.get("USE_AWS_PROFILES", "true").lower() == "true":
        os.environ["AWS_PROFILE"] = aws_env.value
        console.log(f"🔑 Set AWS_PROFILE={aws_env.value}")

    if wandb_model_path:
        loaded_classifier = load_classifier_from_wandb(wandb_model_path)
    else:
        console.log("Loading local classifier...")
        loaded_classifier = load_classifier_local(wikibase_id)
        console.log(f"🤖 Loaded classifier {loaded_classifier} from {classifier_dir}")

    console.log(f'📚 Loading concept "{wikibase_id}" from Wikibase and Argilla')
    concept = asyncio.run(
        get_concept_async(
            wikibase_id=wikibase_id,
            include_labels_from_subconcepts=True,
            include_recursive_has_subconcept=True,
        )
    )

    console.log(
        f"📊 Found {len(concept.labelled_passages)} labelled passages for evaluation"
    )

    if not concept.labelled_passages:
        console.log(
            "[yellow]⚠️  Warning: No labelled passages found. Cannot evaluate.[/yellow]"
        )
        raise typer.Exit(code=1)

    wandb_run = None
    if track:
        entity = "climatepolicyradar"
        project = wikibase_id
        namespace = Namespace(project=project, entity=entity)

        wandb_run = wandb.init(
            entity=namespace.entity,
            project=namespace.project,
            job_type="evaluate_model",
            config={
                "classifier_type": loaded_classifier.name,
                "concept_id": concept.id,
            },
        )
        if wandb_model_path:
            wandb_run.config["model_artifact_path"] = wandb_model_path  # type: ignore
        console.log(f"📊 Tracking evaluation in W&B: {wandb_run.url}")

        if wandb_model_path:
            wandb_run.use_artifact(wandb_model_path)

    try:
        df, _, _ = evaluate_classifier(
            classifier=loaded_classifier,
            labelled_passages=concept.labelled_passages,
            wandb_run=wandb_run,
        )

        # Save metrics to local file
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = save_metrics(df, wikibase_id)
        console.log(f"📄 Saved performance metrics to {metrics_path}")

        if track and wandb_run:
            console.log(f"📊 View results at: {wandb_run.url}")
    finally:
        if wandb_run:
            wandb_run.finish()


if __name__ == "__main__":
    app()
