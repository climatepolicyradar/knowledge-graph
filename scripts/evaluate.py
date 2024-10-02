from collections import defaultdict
from typing import Annotated

import pandas as pd
import typer
from rich import box
from rich.console import Console
from rich.table import Table

from scripts.config import (
    EQUAL_COLUMNS,
    STRATIFIED_COLUMNS,
    classifier_dir,
    concept_dir,
    processed_data_dir,
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
):
    """Measure classifier performance against human-labelled gold-standard datasets"""
    console.log("🚀 Starting classifier performance measurement")

    try:
        concept = Concept.load(concept_dir / f"{wikibase_id}.json")
        console.log(f'📚 Loaded concept "{concept}" from {concept_dir}')
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
                    "Precision": f"{confusion_matrix.precision():.2f}",
                    "Recall": f"{confusion_matrix.recall():.2f}",
                    "Accuracy": f"{confusion_matrix.accuracy():.2f}",
                    "F1 score": f"{confusion_matrix.f1_score():.2f}",
                    "Support": str(confusion_matrix.support()),
                },
            )

    df = pd.DataFrame(metrics)

    table = Table(box=box.SIMPLE, show_header=True)
    for column in df.columns:
        table.add_column(column)
    for _, row in df.iterrows():
        table.add_row(*row)

    console.log(table)

    metrics_path = processed_data_dir / "classifier_performance" / f"{wikibase_id}.json"
    df.to_json(metrics_path, orient="records", indent=2)
    console.log(f"📄 Saved performance metrics to {metrics_path}")


if __name__ == "__main__":
    app()
