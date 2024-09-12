from pathlib import Path
from typing import Annotated

import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table
from typer import Option, Typer

from scripts.config import classifier_dir, processed_data_dir
from src.classifier import Classifier
from src.labelled_passage import LabelledPassage
from src.metrics import (
    ConfusionMatrix,
    count_passage_level_metrics,
    count_span_level_metrics,
)
from src.sampling import SamplingConfig

console = Console(highlight=False)


app = Typer()


def group_passages_by_equity_strata(
    human_labelled_passages: list[LabelledPassage],
    model_labelled_passages: list[LabelledPassage],
    equity_strata: list[str],
) -> list[tuple[str, list[LabelledPassage], list[LabelledPassage]]]:
    groups = [("all", human_labelled_passages, model_labelled_passages)]
    return groups


@app.command()
def main(
    config_path: Annotated[Path, Option(..., help="Path to the sampling config")],
    verbose: Annotated[
        bool,
        Option(
            ...,
            help="Show the comparison between human and model labels in the console",
        ),
    ] = False,
    equity: Annotated[
        bool,
        Option(
            ...,
            help="Calculate metrics across equity strata",
        ),
    ] = False,
):
    """Measure classifier performance against human-labelled evaluation datasets"""
    console.log("🚀 Starting classifier performance measurement")

    console.log(f"⚙️ Loading config from {config_path}")
    config = SamplingConfig.load(config_path)
    console.log("✅ Config loaded")

    equity_strata = config.equal_columns + config.stratified_columns

    labelled_passages_dir = processed_data_dir / "labelled_passages"

    if not labelled_passages_dir.exists():
        raise FileNotFoundError(
            "Labelled passages data doesn't exist. Run save_labelled_passages_from_argilla.py first"
        )

    for wikibase_id in config.wikibase_ids:
        results: dict[str, ConfusionMatrix] = {}

        # Load the human-labelled passages for the concept
        labelled_passages_path = labelled_passages_dir / wikibase_id / "agreements.json"
        human_labelled_passages = [
            LabelledPassage.model_validate_json(line)
            for line in labelled_passages_path.read_text(encoding="utf-8").splitlines()
        ]
        classifier = Classifier.load(classifier_dir / wikibase_id)

        # create a new set of labelled passages, labelled by the classifier
        model_labelled_passages: list[LabelledPassage] = []
        for labelled_passage in human_labelled_passages:
            model_labelled_passages.append(
                LabelledPassage(
                    text=labelled_passage.text,
                    spans=classifier.predict(labelled_passage.text),
                )
            )

        df = pd.DataFrame(
            columns=[
                "Group",
                "Agreement at",
                "Precision",
                "Recall",
                "Accuracy",
                "F1 score",
                "Cohen's kappa",
            ]
        )

        for (
            group,
            human_labelled_passages,
            model_labelled_passages,
        ) in group_passages_by_equity_strata(
            human_labelled_passages, model_labelled_passages, equity_strata
        ):
            results["Passage level"] = count_passage_level_metrics(
                human_labelled_passages, model_labelled_passages
            )

            for threshold in [0.001, 0.5, 0.9, 1]:
                results[f"Span level ({threshold})"] = count_span_level_metrics(
                    human_labelled_passages,
                    model_labelled_passages,
                    threshold=threshold,
                )

            df = pd.concat(
                [
                    df,
                    pd.DataFrame(
                        [
                            {
                                "Group": group,
                                "Agreement at": agreement_level,
                                "Precision": f"{confusion_matrix.precision():.2f}",
                                "Recall": f"{confusion_matrix.recall():.2f}",
                                "Accuracy": f"{confusion_matrix.accuracy():.2f}",
                                "F1 score": f"{confusion_matrix.f1_score():.2f}",
                                "Cohen's kappa": f"{confusion_matrix.cohens_kappa():.2f}",
                            }
                            for agreement_level, confusion_matrix in results.items()
                        ]
                    ),
                ]
            )

        if verbose:
            table = Table(
                title=f"Performance metrics for {wikibase_id}",
                title_justify="left",
                title_style="bold",
                box=box.SIMPLE,
                show_header=True,
            )
            for column in df.columns:
                table.add_column(column)
            for _, row in df.iterrows():
                table.add_row(*row)

            console.print(table)

        df.to_json(
            processed_data_dir / "classifier_performance" / f"{wikibase_id}.json",
            orient="records",
        )


if __name__ == "__main__":
    app()
