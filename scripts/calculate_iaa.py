"""Calculate inter-annotator agreement for a labelled dataset."""

import itertools
from collections import defaultdict

import typer
from rich.console import Console
from rich.table import Table

from src.concept import Concept
from src.config import concept_dir
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
    thresholds: list[float] = typer.Option(
        [0, 0.5, 0.9, 0.99], help="The span-level thresholds to calculate IAA for"
    ),
):
    """
    Calculate inter-annotator agreement for a labelled dataset.

    This script calculates the inter-annotator agreement for a labelled dataset
    using the Cohen's Kappa and F1 Score metrics. The script will produce metrics for
    all labeller pairs, at both the passage-level and span-level (with different
    overlap thresholds).
    """

    try:
        concept = Concept.load(concept_dir / f"{wikibase_id}.json")
    except FileNotFoundError:
        console.print(
            f"Concept {wikibase_id} not found. Have you run `just get-concept {wikibase_id}` ?",
            style="red",
        )
        raise typer.Exit(1)

    # Get all unique labeller names
    labeller_names = set()
    for passage in concept.labelled_passages:
        for span in passage.spans:
            labeller_names.update(span.labellers)

    # Make sure that there are at least 2 labellers
    if len(labeller_names) < 2:
        console.print(
            f"Found {len(labeller_names)} labeller(s). "
            "At least 2 labellers are required for IAA calculation.",
            style="red",
        )
        raise typer.Exit(1)

    labeller_passages = defaultdict(list)
    for passage in concept.labelled_passages:
        for labeller in labeller_names:
            labeller_passage = LabelledPassage(
                text=passage.text,
                spans=[],
            )

            for span in passage.spans:
                if labeller in span.labellers:
                    labeller_passage.spans.append(span)
            labeller_passages[labeller].append(labeller_passage)

    # Calculate a list of all pairwise labeller combinations
    labeller_pairs = list(itertools.combinations(labeller_names, 2))

    # Create a table for each level of agreement
    tables = {}

    tables["passage"] = Table(title="Passage-level Agreement", title_justify="left")
    tables["passage"].add_column("Labeller Pair")
    tables["passage"].add_column("Cohen's Kappa")
    tables["passage"].add_column("F1 Score")

    for threshold in thresholds:
        tables[threshold] = Table(
            title=f"Span-level Agreement (threshold={threshold})", title_justify="left"
        )
        tables[threshold].add_column("Labeller Pair")
        tables[threshold].add_column("Cohen's Kappa")
        tables[threshold].add_column("F1 Score")

    # Calculate the metrics for each labeller pair
    for labeller_1, labeller_2 in labeller_pairs:
        pair_name = f"{labeller_1} & {labeller_2}"

        # Passage-level
        confusion_matrix = count_passage_level_metrics(
            labeller_passages[labeller_1], labeller_passages[labeller_2]
        )
        tables["passage"].add_row(
            pair_name,
            f"{confusion_matrix.cohens_kappa():.3f}",
            f"{confusion_matrix.f1_score():.3f}",
        )

        # Span-level at specifiedthresholds
        for threshold in thresholds:
            confusion_matrix = count_span_level_metrics(
                labeller_passages[labeller_1],
                labeller_passages[labeller_2],
                threshold=threshold,
            )
            tables[threshold].add_row(
                pair_name,
                f"{confusion_matrix.cohens_kappa():.3f}",
                f"{confusion_matrix.f1_score():.3f}",
            )

    # Display the results
    for table in tables.values():
        console.print(table)
        console.print("\n")


if __name__ == "__main__":
    app()
