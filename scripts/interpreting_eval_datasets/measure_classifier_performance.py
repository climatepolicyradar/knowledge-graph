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
from src.sampling import SamplingConfig
from src.span import jaccard_similarity

console = Console(highlight=False)


app = Typer()
thresholds = [0.001, 0.1, 0.5, 0.9, 1]


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
):
    """Measure classifier performance against human-labelled evaluation datasets"""
    console.log("🚀 Starting classifier performance measurement")

    console.log(f"⚙️ Loading config from {config_path}")
    config = SamplingConfig.load(config_path)
    console.log("✅ Config loaded")

    labelled_passages_dir = processed_data_dir / "labelled_passages"

    if not labelled_passages_dir.exists():
        raise FileNotFoundError(
            "Labelled passages data doesn't exist. Run save_labelled_passages_from_argilla.py first"
        )

    for wikibase_id in config.wikibase_ids:
        results = []
        # Load the labelled passages for the concept
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
        for threshold in thresholds:
            true_positives = 0
            false_positives = 0
            false_negatives = 0
            for human_labelled_passage, model_labelled_passage in zip(
                human_labelled_passages, model_labelled_passages
            ):
                for human_span in human_labelled_passage.spans:
                    found = False
                    for model_span in model_labelled_passage.spans:
                        if jaccard_similarity(human_span, model_span) >= threshold:
                            found = True
                            true_positives += 1
                            break
                    if not found:
                        false_negatives += 1
                        break

                for model_span in model_labelled_passage.spans:
                    found = False
                    for human_span in human_labelled_passage.spans:
                        if jaccard_similarity(model_span, human_span) >= threshold:
                            found = True
                            break
                    if not found:
                        false_positives += 1
                        break

            try:
                precision = true_positives / (true_positives + false_positives)
            except ZeroDivisionError:
                precision = 0
            try:
                recall = true_positives / (true_positives + false_negatives)
            except ZeroDivisionError:
                recall = 0
            try:
                f1_score = 2 * (precision * recall) / (precision + recall)
            except ZeroDivisionError:
                f1_score = 0

            results.append(
                {
                    "wikibase_id": wikibase_id,
                    "threshold": threshold,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score,
                }
            )

        results_path = (
            processed_data_dir / "classifier_performance" / f"{wikibase_id}.json"
        )
        results_path.parent.mkdir(parents=True, exist_ok=True)

        results = pd.DataFrame(results)
        results.to_json(results_path, orient="records")
        if verbose:
            table = Table(
                box=box.SIMPLE,
                show_header=True,
                title=f"{classifier.concept.preferred_label} ({wikibase_id})",
                title_justify="left",
                title_style="bold",
            )
            table.add_column("threshold", style="magenta", width=12)
            table.add_column("precision", style="magenta", width=12)
            table.add_column("recall", style="magenta", width=12)
            table.add_column("f1_score", style="magenta", width=12)
            for _, row in results.iterrows():
                table.add_row(
                    str(row["threshold"]),
                    f"{row['precision']:.2f}",
                    f"{row['recall']:.2f}",
                    f"{row['f1_score']:.2f}",
                )
            console.print(table)

    console.log(f"📝 Wrote results to {results_path.parent}")


if __name__ == "__main__":
    app()
