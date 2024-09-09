from pathlib import Path
from typing import Annotated

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


@app.command()
def main(
    config_path: Annotated[Path, Option(..., help="Path to the sampling config")],
    threshold: Annotated[
        float,
        Option(
            ...,
            help=(
                "Jaccard similarity threshold for filtering. A value of 0.5 means that "
                "two spans are considered similar if one span shares at least half of "
                "its tokens with the other span. 1 corresponds to an exact match, "
                "while 0 corresponds to no overlap. Use a very small value to allow "
                "for matches with very little overlap."
            ),
            min=0,
            max=1,
        ),
    ] = 0.5,
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

    console.log(f"🔍 Using a jaccard similarity threshold of {threshold} for equality")

    labelled_passages_dir = processed_data_dir / "labelled_passages"

    if not labelled_passages_dir.exists():
        raise FileNotFoundError(
            "Labelled passages data doesn't exist. Run save_labelled_passages_from_argilla.py first"
        )

    table = Table(box=box.SIMPLE_HEAVY)
    table.add_column("Preferred label", style="cyan", justify="right")
    table.add_column("Wikibase ID", style="cyan")
    table.add_column("Precision", style="magenta")
    table.add_column("Recall", style="magenta")
    table.add_column("F1 Score", style="magenta")

    for wikibase_id in config.wikibase_ids:
        if verbose:
            console.log(
                f"📚 Evaluating classifier performance for concept [bold]{wikibase_id}[/bold]"
            )
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

        true_positives = 0
        false_positives = 0
        false_negatives = 0
        for human_labelled_passage, model_labelled_passage in zip(
            human_labelled_passages, model_labelled_passages
        ):
            passage_level_agreement = True
            for human_span in human_labelled_passage.spans:
                found = False
                for model_span in model_labelled_passage.spans:
                    if jaccard_similarity(human_span, model_span) >= threshold:
                        found = True
                        true_positives += 1
                        break
                if not found:
                    passage_level_agreement = False
                    false_negatives += 1
                    break

            for model_span in model_labelled_passage.spans:
                found = False
                for human_span in human_labelled_passage.spans:
                    if jaccard_similarity(model_span, human_span) >= threshold:
                        found = True
                        break
                if not found:
                    passage_level_agreement = False
                    false_positives += 1
                    break

            if verbose:
                console.log("Human-labelled passage:", style="white bold")
                console.log(human_labelled_passage.get_highlighted_text())
                console.log("Model-labelled passage:", style="white bold")
                console.log(model_labelled_passage.get_highlighted_text())
                console.log(
                    f"Passage-level agreement: {'✅' if passage_level_agreement else '❌'}",
                    style="white bold",
                    end="\n\n",
                )

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

        table.add_row(
            f"[link=https://climatepolicyradar.wikibase.cloud/wiki/Item:{wikibase_id}]{classifier.concept.preferred_label}[/link]",
            f"[link=https://climatepolicyradar.wikibase.cloud/wiki/Item:{wikibase_id}]{wikibase_id}[/link]",
            f"{precision:.2f}",
            f"{recall:.2f}",
            f"{f1_score:.2f}",
        )

    console.print(table)


if __name__ == "__main__":
    app()
