from collections import defaultdict
from typing import Annotated

import pandas as pd
import typer
import wandb
from pydantic import BaseModel, Field
from rich import box
from rich.console import Console
from rich.table import Table

from scripts.cloud import AwsEnv
from scripts.config import (
    EQUAL_COLUMNS,
    STRATIFIED_COLUMNS,
    classifier_dir,
    concept_dir,
    metrics_dir,
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

console = Console()
app = typer.Typer()


class Namespace(BaseModel):
    """Hierarchy we use: CPR / {concept} / {classifier}"""

    project: WikibaseID = Field(
        ...,
        description="The name of the W&B project, which is the concept ID",
    )
    entity: str = Field(
        ...,
        description="The name of the W&B entity",
    )


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
    aws_env: Annotated[
        AwsEnv,
        typer.Option(
            ...,
            help="AWS environment to use for metadata",
        ),
    ] = AwsEnv.labs,
    # TODO Accept version, and DL it from W&B
):
    if track:
        entity = "climatepolicyradar"
        project = wikibase_id
        namespace = Namespace(project=project, entity=entity)
        job_type = "evaluate_model"
        config = {"aws_env": aws_env.value}

        run = wandb.init(
            entity=namespace.entity,
            project=namespace.project,
            job_type=job_type,
            config=config,
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

    try:
        classifier = Classifier.load(classifier_dir / wikibase_id)
        console.log(f"🤖 Loaded classifier {classifier} from {classifier_dir}")
    except FileNotFoundError as e:
        raise typer.BadParameter(
            f"Classifier for {wikibase_id} not found. \n"
            "If you haven't already, you should run:\n"
            f"  just train {wikibase_id}\n"
        ) from e

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
        if classifier.version is not None:
            # It must have been tracked and uploaded, during training
            version = classifier.version
            artifact_id = f"{wikibase_id}/{classifier.name}:{version}"
            console.log(f"Using artifact: {artifact_id}")
            run.use_artifact(artifact_id)

        table = wandb.Table(data=df.values.tolist(), columns=df.columns.tolist())

        run.log({"performance": table})
        run.finish()


if __name__ == "__main__":
    app()
