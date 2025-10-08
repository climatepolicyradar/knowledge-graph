import os
from collections import defaultdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated, Any, Optional

import pandas as pd
import typer
import wandb
from rich import box
from rich.console import Console
from rich.table import Table
from wandb.wandb_run import Run

from knowledge_graph.classifier import Classifier
from knowledge_graph.cloud import Namespace
from knowledge_graph.concept import Concept
from knowledge_graph.config import (
    classifier_dir,
    concept_dir,
    equity_columns,
    metrics_dir,
    model_artifact_name,
)
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.metrics import (
    ConfusionMatrix,
    count_passage_level_metrics,
    count_span_level_metrics,
)
from knowledge_graph.span import Span, group_overlapping_spans
from knowledge_graph.version import Version
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
    if not classifier_path.exists() or not list(classifier_path.glob("*.pickle")):
        raise typer.BadParameter(
            f"Classifier for {wikibase_id} not found. \n"
            "If you haven't already, you should run:\n"
            f"  just train {wikibase_id}\n"
        )

    try:
        most_recent_classifier_path = max(
            classifier_path.glob("*.pickle"), key=os.path.getctime
        )
        return Classifier.load(most_recent_classifier_path)
    except (FileNotFoundError, ValueError) as e:
        raise typer.BadParameter(
            f"Classifier for {wikibase_id} not found. \n"
            "If you haven't already, you should run:\n"
            f"  just train {wikibase_id}\n"
        ) from e


def load_classifier_remote(
    run: Run,
    classifier: str,
    version: Version,
    wikibase_id: WikibaseID,
) -> Classifier:
    """
    Load a classifier from W&B artifacts storage.

    NB: This function temporarily sets AWS_PROFILE to enable W&B to download
    artifacts from S3. The profile is restored after download completes.
    """
    artifact_id = f"{wikibase_id}/{classifier}:{version}"
    artifact = run.use_artifact(artifact_id, type="model")

    aws_env = artifact.metadata["aws_env"]

    # Make it easier to know which AWS env this happened in
    run.config["aws_env"] = aws_env

    # Temporarily set AWS_PROFILE for W&B's S3 artifact download
    original_profile = os.environ.get("AWS_PROFILE")
    try:
        os.environ["AWS_PROFILE"] = aws_env
        artifact_dir = artifact.download()
        artifact_path = Path(artifact_dir) / model_artifact_name
        return Classifier.load(artifact_path)
    finally:
        # Restore original AWS_PROFILE
        if original_profile is not None:
            os.environ["AWS_PROFILE"] = original_profile
        else:
            os.environ.pop("AWS_PROFILE", None)


def add_artifact_to_run_lineage_local(
    run: Run,
    classifier: Classifier,
    wikibase_id: WikibaseID,
) -> None:
    """Add a local model artifact to the W&B run lineage."""
    # Attempt to get a version, otherwise, have none for the lineage.
    #
    # The model have been trained and saved without a version.
    if classifier.version is not None:
        # It must have been tracked and uploaded, during training
        artifact_id = f"{wikibase_id}/{classifier.name}:{classifier.version}"
        console.log(f"Using artifact: {artifact_id}")
        run.use_artifact(artifact_id)


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


def label_passages_with_classifier(
    classifier: Classifier,
    gold_standard_labelled_passages: list[LabelledPassage],
) -> list[LabelledPassage]:
    """Label passages using the provided classifier."""
    return [
        labelled_passage.model_copy(
            update={"spans": classifier.predict(labelled_passage.text)},
            deep=True,
        )
        for labelled_passage in gold_standard_labelled_passages
    ]


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


class Source(str, Enum):
    """Source of the classifier model."""

    LOCAL = "local"
    REMOTE = "remote"


def validate_local_args(
    classifier: Optional[str],
    version: Optional[Version],
) -> None:
    """Validate arguments for local source."""
    if classifier is not None or version is not None:
        raise typer.BadParameter(
            "classifier and version should not be specified for local source"
        )


def validate_remote_args(
    track: bool,
    classifier: Optional[str],
    version: Optional[Version],
) -> None:
    """Validate arguments for remote source."""
    if not track:
        if version is not None:
            raise typer.BadParameter(
                f"A remote version ({version}) was specified, but the script "
                "was told not to track. Tracking must be enabled to use a "
                "remote version."
            )
        if classifier is not None:
            raise typer.BadParameter(
                f"A remote classifier ({classifier}) was specified, but the "
                "script was told not to track. Tracking must be enabled to use"
                " a remote classifier."
            )
    else:
        if version is not None and classifier is None:
            raise typer.BadParameter(
                "cannot track a remote model artifact without a classifier name"
            )
        if classifier is not None and version is None:
            raise typer.BadParameter(
                "cannot track a remote model artifact without a version"
            )


def validate_args(
    track: bool,
    classifier: Optional[str],
    version: Optional[Version],
    source: Source,
) -> None:
    """Validate command line arguments for model evaluation."""
    match source:
        case Source.LOCAL:
            validate_local_args(classifier, version)
        case Source.REMOTE:
            validate_remote_args(track, classifier, version)


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


def evaluate_classifier(
    classifier: Classifier,
    labelled_passages: list[LabelledPassage],
    wandb_run: Optional[Run] = None,
) -> tuple[pd.DataFrame, list[LabelledPassage]]:
    """
    Evaluate the performance of a classifier using an evaluation dataset.

    :param Classifier classifier: classifier to evaluate
    :param list[LabelledPassage] labelled_passages: labelled passages, as pulled from
        Argilla, to evaluate the classifier against
    :return tuple[pd.DataFrame, list[LabelledPassage]]: dataframe of metrics, and list
        of passages labelled by the model
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

    if wandb_run:
        log_metrics_to_wandb(wandb_run, df)  # type: ignore

    return df, model_labelled_passages


app = typer.Typer()


@app.command()
async def main(
    wikibase_id: Annotated[
        WikibaseID,
        typer.Option(
            ...,
            help="The Wikibase ID of the concept classifier to train",
            parser=WikibaseID,
        ),
    ],
    track: Annotated[
        bool,
        typer.Option(
            ...,
            help="Whether to track the training run with Weights & Biases",
        ),
    ] = False,
    classifier: Annotated[
        Optional[str],
        typer.Option(
            help="Classifier name that aligns with the Python class name",
        ),
    ] = None,
    version: Annotated[
        Optional[Version],
        typer.Option(
            help="Version of the model (e.g., v3) to download through W&B",
            parser=Version,
        ),
    ] = None,
    source: Annotated[
        Source,
        typer.Option(
            help="Source of the classifier model (local or remote)",
        ),
    ] = Source.LOCAL,
):
    """
    Measure classifier performance.

    This is done against human-labelled gold-standard datasets.
    """
    validate_args(
        track,
        classifier,
        version,
        source,
    )

    # set explicitly to avoid run being possibly unbound later on
    run = None

    if track:
        entity = "climatepolicyradar"
        project = wikibase_id
        namespace = Namespace(project=project, entity=entity)
        job_type = "evaluate_model"

        run = wandb.init(
            entity=namespace.entity,
            project=namespace.project,
            job_type=job_type,
        )

    tracking = "on" if track else "off"
    console.log(
        f"🚀 Starting classifier performance measurement with tracking {tracking}"
    )

    metrics_dir.mkdir(parents=True, exist_ok=True)

    console.log(f'📚 Loading concept with ID "{wikibase_id}" from Wikibase and Argilla')
    concept = await get_concept_async(wikibase_id=wikibase_id)

    if track:
        run.config["preferred_label"] = concept.preferred_label  # type: ignore

    console.log("Loading classifier...")
    match source:
        case Source.LOCAL:
            loaded_classifier = load_classifier_local(wikibase_id)
            console.log(
                f"🤖 Loaded classifier {loaded_classifier} from {classifier_dir}"
            )
            if track:
                add_artifact_to_run_lineage_local(run, classifier, wikibase_id)  # type: ignore
        case Source.REMOTE:
            loaded_classifier = load_classifier_remote(
                run,  # type: ignore
                classifier,  # type: ignore
                version,  # type: ignore
                wikibase_id,
            )
    assert isinstance(classifier, Classifier)

    df, labelled_passages = evaluate_classifier(
        classifier=classifier,
        labelled_passages=concept.labelled_passages,
        wandb_run=run if track else None,
    )

    if track:
        run.finish()  # type: ignore

    metrics_path = save_metrics(df, wikibase_id)
    console.log(f"📄 Saved performance metrics to {metrics_path}")


if __name__ == "__main__":
    app()
