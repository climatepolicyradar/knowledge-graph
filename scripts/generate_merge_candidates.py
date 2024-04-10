import json
import re
from collections import defaultdict
from pathlib import Path
from string import punctuation

import typer
from rich.console import Console
from rich.table import Table
from typing_extensions import Annotated

from src.wikibase import WikibaseSession

console = Console(highlight=False)
wikibase = WikibaseSession()


def normalise(label: str) -> str:
    """
    Normalise a label

    Removes punctuation, casing, stopwords, pluralisation, and whitespace.
    """
    # lowercase
    label = label.lower()

    # replace any punctuation with whitespace
    label = re.sub(rf"[{punctuation}]", " ", label)

    # remove multiple whitespace characters
    label = re.sub(r"\s+", " ", label)

    # get rid of stopwords
    stopwords = [
        "the",
        "of",
        "and",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "from",
        "as",
        "or",
        "an",
        "a",
    ]
    label = " ".join([word for word in label.split() if word not in stopwords])

    # remove pluralisation
    words = label.split()
    for i, word in enumerate(words):
        if word.endswith("s"):
            words[i] = word[:-1]
        if word.endswith("es"):
            words[i] = word[:-2]
    label = " ".join(words)

    return label


def main(
    output_path: Annotated[
        Path,
        typer.Option(
            exists=False,
            file_okay=True,
            dir_okay=False,
            writable=False,
            resolve_path=True,
            help="Path to the output file",
        ),
    ],
):
    """
    Queries the wikibase instance for groups of items which could plausibly be merged

    Outputs a json file with the following structure:
    [
        [Q123, Q456, Q789],
        [Q234, Q567],
        ...
    ]

    The output file can be used as input to the merge_concepts.py script
    """
    assert output_path.suffix == ".json", "Output file must be a json file"
    merge_candidates = []

    console.print("Fetching all items from wikibase...", style="dim")
    all_item_ids = [item["q_id"] for item in wikibase.get_all_items()]
    concepts = wikibase.get_concepts(all_item_ids)

    console.print("Normalising labels...", style="dim")
    normalised_labels = [
        list(set([normalise(label) for label in concept.all_labels]))
        for concept in concepts
    ]

    console.print("Finding merge candidates...", style="dim")
    normalised_label_to_concept = defaultdict(list)
    for i, concept in enumerate(concepts):
        for label in normalised_labels[i]:
            normalised_label_to_concept[label].append(concept)

    for label, concepts in normalised_label_to_concept.items():
        if len(concepts) > 1:
            merge_candidates.append(concepts)

    console.print(f"Found {len(merge_candidates)} merge candidates", style="bold green")

    # for each candidate, prompt the user to select them as a merge candidate
    for i, candidate in enumerate(merge_candidates):
        console.print(
            f"\nMerge candidate group {i + 1}/{len(merge_candidates)}",
            style="bold white",
        )

        table = Table()
        table.add_column("ID")
        table.add_column("Preferred Label")
        table.add_column("Alternative Labels")

        for concept in candidate:
            table.add_row(
                concept.wikibase_id,
                concept.preferred_label,
                ", ".join(concept.alternative_labels),
            )

        console.print(table)

        merge = typer.confirm("Would you like to merge these concepts?")
        if not merge:
            merge_candidates.remove(candidate)

    merge_candidate_ids = [
        [concept.wikibase_id for concept in candidate] for candidate in merge_candidates
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(merge_candidate_ids, f, indent=4)

    console.print(f"Saved merge candidates to {output_path}", style="bold green")


if __name__ == "__main__":
    typer.run(main)
