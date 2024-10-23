import os
from collections import defaultdict
from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
import typer
import wandb
from rich import box
from rich.console import Console
from rich.table import Table

from scripts.cloud import Namespace
from scripts.config import (
    EQUAL_COLUMNS,
    STRATIFIED_COLUMNS,
    classifier_dir,
    concept_dir,
    metrics_dir,
    model_artifact_name,
)
from src.classifier import Classifier
from src.concept import Concept
from src.identifiers import WikibaseID
from src.labelled_passage import LabelledPassage
from src.metrics import (
    ConfusionMatrix,
    count_passage_level_metrics,
    count_span_level_metrics,
)
from src.span import Span, group_overlapping_spans
from src.version import Version

console = Console()


def group_passages_by_equity_strata(
    human_labelled_passages: list[LabelledPassage],
    model_labelled_passages: list[LabelledPassage],
    equity_strata: list[str],
) -> list[tuple[str, list[LabelledPassage], list[LabelledPassage]]]:
    groups = [("all", human_labelled_passages, model_labelled_passages)]

    # get the unique values for each equity strata from the labelled passages' metadata
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


app = typer.Typer()


def validate_args(
    track: bool,
    classifier: Optional[str],
    version: Optional[Version],
) -> None:
    match (track, classifier, version):
        # Tracking from a local model artifact
        case (True, None, None):
            pass
        case (True, None, version) if version is not None:
            raise typer.BadParameter(
                "cannot track a remote model artifact without a classifier name"
            )
        case (True, classifier, None) if classifier is not None:
            raise typer.BadParameter(
                "cannot track a remote model artifact without a version"
            )
        # Tracking from a remote model artifact
        case (
            True,
            classifier,
            version,
        ) if classifier is not None and version is not None:
            pass
        case (False, None, version) if version is not None:
            raise typer.BadParameter(
                f"A remote version ({version}) was specified, but the script was told not to track. Tracking must be enabled to use a remote version."
            )
        case (False, classifier, None) if classifier is not None:
            raise typer.BadParameter(
                f"A remote classifier ({classifier}) was specified, but the script was told not to track. Tracking must be enabled to use a remote classifier."
            )
        # No tracking or downloading
        case (False, None, None):
            pass
        case _:
            raise typer.BadParameter("invalid evaluation arguments combination")


@app.command()
def main(
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
):
    validate_args(
        track,
        classifier,
        version,
    )

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

    try:
        concept = Concept.load(concept_dir / f"{wikibase_id}.json")
        console.log(f'📚 Loaded concept "{concept}" from {concept_dir}')
        if track:
            run.config["preferred_label"] = concept.preferred_label
    except FileNotFoundError as e:
        raise typer.BadParameter(
            f"Data for {wikibase_id} not found. \n"
            "If you haven't already, you should run:\n"
            f"  just get-concept {wikibase_id}\n"
        ) from e

    console.log("🥇 Creating a list of gold-standard labelled passages")
    gold_standard_labelled_passages: list[LabelledPassage] = []
    for labelled_passage in concept.labelled_passages:
        merged_spans = []
        for group in group_overlapping_spans(
            spans=labelled_passage.spans, jaccard_threshold=0
        ):
            merged_span = Span.union(spans=group)
            merged_span.labellers = ["gold standard"]
            merged_spans.append(merged_span)

        gold_standard_labelled_passages.append(
            labelled_passage.model_copy(update={"spans": merged_spans}, deep=True)
        )
    n_annotations = sum([len(entry.spans) for entry in gold_standard_labelled_passages])
    console.log(
        f"🚚 Loaded {len(gold_standard_labelled_passages)} labelled passages "
        f"with {n_annotations} individual annotations"
    )
    if track:
        run.config["n_gold_standard_labelled_passages"] = len(
            gold_standard_labelled_passages
        )
        run.config["n_annotations"] = n_annotations

    # No (remote) version was specified, so use the local artifact.
    #
    # Load the classifier, regardless of tracking or not.
    if version is None and classifier is None:
        try:
            classifier = Classifier.load(classifier_dir / wikibase_id)
            console.log(f"🤖 Loaded classifier {classifier} from {classifier_dir}")
            if track:
                # Attempt to get a version, otherwise, have none for the lineage.
                #
                # The model have been trained and saved without a version.
                if classifier.version is not None:
                    # It must have been tracked and uploaded, during training
                    artifact_id = (
                        f"{wikibase_id}/{classifier.name}:{classifier.version}"
                    )
                    console.log(f"Using artifact: {artifact_id}")
                    run.use_artifact(artifact_id)
        except FileNotFoundError as e:
            raise typer.BadParameter(
                f"Classifier for {wikibase_id} not found. \n"
                "If you haven't already, you should run:\n"
                f"  just train {wikibase_id}\n"
            ) from e
    # Otherwise, pull through W&B, if still tracking
    elif track:
        artifact_id = f"{wikibase_id}/{classifier}:{version}"
        artifact = run.use_artifact(artifact_id, type="model")

        aws_env = artifact.metadata["aws_env"]

        # Make it easier to know which AWS env this happened in
        run.config["aws_env"] = aws_env

        # Set this for W&B to pickup
        os.environ["AWS_PROFILE"] = aws_env

        artifact_dir = artifact.download()
        artifiact_path = Path(artifact_dir) / model_artifact_name
        classifier = Classifier.load(artifiact_path)
    else:
        raise ValueError("impossible to reach here")

    console.log("🤖 Labelling passages with the classifier")
    model_labelled_passages = [
        labelled_passage.model_copy(
            update={"spans": classifier.predict(labelled_passage.text)}, deep=True
        )
        for labelled_passage in gold_standard_labelled_passages
    ]
    n_annotations = sum([len(entry.spans) for entry in model_labelled_passages])
    console.log(
        f"✅ Labelled {len(model_labelled_passages)} passages "
        f"with {n_annotations} individual annotations"
    )
    if track:
        run.config["n_model_labelled_passages"] = len(model_labelled_passages)

    console.log(f"📊 Calculating performance metrics for {concept}")
    confusion_matrices: dict[str, dict[str, ConfusionMatrix]] = defaultdict(dict)
    for (
        group,
        gold_standard_labelled_passages,
        model_labelled_passages,
    ) in group_passages_by_equity_strata(
        human_labelled_passages=gold_standard_labelled_passages,
        model_labelled_passages=model_labelled_passages,
        equity_strata=EQUAL_COLUMNS + STRATIFIED_COLUMNS,
    ):
        confusion_matrices[group]["Passage level"] = count_passage_level_metrics(
            gold_standard_labelled_passages, model_labelled_passages
        )

        # calculate span-level metrics at different thresholds. The thresholds define the
        # minimum Jaccard similarity required for two spans to be considered a match. We
        # take a range of thresholds to get a sense of how the model performs at
        # different levels of agreement (with 0 allowing for any overlap between model
        # and human, and 1 setting a requirement for an exact match)
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
    for group, results in confusion_matrices.items():
        for agreement_level, confusion_matrix in results.items():
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

    df = pd.DataFrame(metrics)

    table = Table(box=box.SIMPLE, show_header=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        formatted_row = [
            f"{value:.2f}" if isinstance(value, float) else str(value) for value in row
        ]
        table.add_row(*formatted_row)

    console.log(table)

    metrics_path = metrics_dir / f"{wikibase_id}.json"
    df.to_json(metrics_path, orient="records", indent=2)
    console.log(f"📄 Saved performance metrics to {metrics_path}")

    if track:
        # Attempt to get a version, otherwise, have none for the lineage.
        #
        # The model have been trained and saved without a version.
        #
        # This is only if a remote model artifact isn't being used, as
        # it would've been downloaded, and thus included in the
        # lineage for the run already.
        if classifier.version is not None and (version is None and classifier is None):
            # It must have been tracked and uploaded, during training
            artifact_id = f"{wikibase_id}/{classifier.name}:{classifier.version}"
            console.log(f"Using artifact: {artifact_id}")
            run.use_artifact(artifact_id)

        table = wandb.Table(data=df.values.tolist(), columns=df.columns.tolist())

        run.log({"performance": table})
        run.finish()


if __name__ == "__main__":
    app()
