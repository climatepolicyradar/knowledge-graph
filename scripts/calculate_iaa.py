"""Calculate inter-annotator agreement for a labelled dataset."""

import typer
from rich.console import Console

from scripts.config import concept_dir
from src.concept import Concept
from src.identifiers import WikibaseID
from src.labelled_passage import LabelledPassage
from src.metrics import count_passage_level_metrics, count_span_level_metrics

console = Console()
app = typer.Typer()


def validate_wikibase_id(value: str) -> str:
    try:
        WikibaseID(value)
    except ValueError:
        raise typer.BadParameter(value)
    return value


@app.command()
def calculate_iaa(
    wikibase_id: str = typer.Argument(
        ..., help="The Wikibase ID to calculate IAA for", callback=validate_wikibase_id
    ),
):
    """Calculate inter-annotator agreement for a labelled dataset."""

    try:
        concept = Concept.load(concept_dir / f"{wikibase_id}.json")
    except FileNotFoundError:
        console.print(
            f"Concept {wikibase_id} not found. Have you run `just get-concept {wikibase_id}` ?",
            style="red",
        )
        raise typer.Exit(1)

    labeller_names = set()
    for passage in concept.labelled_passages:
        for span in passage.spans:
            labeller_names.update(span.labellers)
    assert (
        len(labeller_names) == 2
    ), f"There should be two labellers, found {labeller_names}"
    labeller_1_name, labeller_2_name = labeller_names

    labeller_1_passages: list[LabelledPassage] = []
    labeller_2_passages: list[LabelledPassage] = []

    for passage in concept.labelled_passages:
        labeller_1_passage = LabelledPassage(
            text=passage.text,
            spans=[],
        )
        labeller_2_passage = LabelledPassage(
            text=passage.text,
            spans=[],
        )
        for span in passage.spans:
            if labeller_1_name in span.labellers:
                labeller_1_passage.spans.append(span)
            elif labeller_2_name in span.labellers:
                labeller_2_passage.spans.append(span)
        labeller_1_passages.append(labeller_1_passage)
        labeller_2_passages.append(labeller_2_passage)

    confusion_matrix = count_passage_level_metrics(
        labeller_1_passages, labeller_2_passages
    )
    console.print("Passage-level metrics:")
    console.print(confusion_matrix)
    console.print(f"Kappa score: {confusion_matrix.cohens_kappa()}", end="\n\n")

    console.print("Span-level metrics:")
    for threshold in [0, 0.5, 0.9]:
        confusion_matrix = count_span_level_metrics(
            labeller_1_passages, labeller_2_passages, threshold=threshold
        )
        console.print(f"Threshold: {threshold}")
        console.print(confusion_matrix)
        console.print(f"Kappa score: {confusion_matrix.cohens_kappa()}", end="\n\n")


if __name__ == "__main__":
    app()
