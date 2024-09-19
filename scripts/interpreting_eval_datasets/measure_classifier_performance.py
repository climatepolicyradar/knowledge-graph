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

    output_dir = Path(processed_data_dir / "classifier_performance")
    output_dir.mkdir(parents=True, exist_ok=True)

    console.log("🚀 Starting classifier performance measurement")

    console.log(f"⚙️ Loading config from {config_path}")
    config = SamplingConfig.load(config_path)
    console.log("✅ Config loaded")

    equity_strata = config.equal_columns + config.stratified_columns

    labelled_passages_dir = processed_data_dir / "labelled_passages"
    gold_standard_labelled_passages_paths = [
        labelled_passages_dir / wikibase_id / "gold_standard.jsonl"
        for wikibase_id in config.wikibase_ids
    ]
    missing_gold_standard_paths = [
        path for path in gold_standard_labelled_passages_paths if not path.exists()
    ]
    if missing_gold_standard_paths:
        raise FileNotFoundError(
            "Some gold standard labelled passages don't exist. Make sure you've run "
            "save_labelled_passages_from_argilla.py and "
            "create_gold_standard_labels.py with the same config before running this "
            "script."
            f"Missing paths: {missing_gold_standard_paths}"
        )

    classifier_paths = [
        classifier_dir / wikibase_id for wikibase_id in config.wikibase_ids
    ]
    missing_classifier_paths = [path for path in classifier_paths if not path.exists()]
    if missing_classifier_paths:
        raise FileNotFoundError(
            "Some classifiers don't exist. Make sure you've run train_classifier.py "
            "with the same config before running this script."
            f"Missing paths: {missing_classifier_paths}"
        )

    for labelled_passages_path, classifier_path in zip(
        gold_standard_labelled_passages_paths, classifier_paths
    ):
        results: dict[str, ConfusionMatrix] = {}
        wikibase_id = labelled_passages_path.parent.name

        # load the gold-standard passages and the classifier
        gold_standard_labelled_passages = [
            LabelledPassage.model_validate_json(line)
            for line in labelled_passages_path.read_text(encoding="utf-8").splitlines()
        ]
        classifier = Classifier.load(classifier_path)

        # create a new set of labelled passages, labelled by the classifier
        model_labelled_passages = [
            labelled_passage.model_copy(
                update={"spans": classifier.predict(labelled_passage.text)}, deep=True
            )
            for labelled_passage in gold_standard_labelled_passages
        ]

        df = pd.DataFrame(
            columns=[
                "Group",
                "Agreement at",
                "Precision",
                "Recall",
                "Accuracy",
                "F1 score",
            ]
        )

        for (
            group,
            gold_standard_labelled_passages,
            model_labelled_passages,
        ) in group_passages_by_equity_strata(
            gold_standard_labelled_passages, model_labelled_passages, equity_strata
        ):
            results["Passage level"] = count_passage_level_metrics(
                gold_standard_labelled_passages, model_labelled_passages
            )

            for threshold in [0, 0.5, 0.9, 1]:
                results[f"Span level ({threshold})"] = count_span_level_metrics(
                    gold_standard_labelled_passages,
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
                                "Support": str(confusion_matrix.support()),
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
