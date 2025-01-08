"""
A script to fish acronyms out of the concept store for search purposes.

Exports a JSONL file with all acronyms per concept, and a CSV with one acronym per row.
"""

import json
from pathlib import Path

import pandas as pd
from rich.console import Console

from scripts.config import get_git_root
from src.concept import Concept
from src.wikibase import WikibaseSession

console = Console()


def get_acronyms():
    wikibase = WikibaseSession()

    console.print("Getting concepts...")
    concepts: list[Concept] = wikibase.get_concepts()

    console.print("Finding acronyms...")
    acronyms_by_concept = []
    for concept in concepts:
        acronym_labels = [label for label in concept.all_labels if label.isupper()]
        if not acronym_labels:
            continue

        acronyms_by_concept.append(
            {
                "wikibase_id": concept.wikibase_id,
                "preferred_label": concept.preferred_label,
                "acronyms": acronym_labels,
                "non_acronym_labels": [
                    label for label in concept.all_labels if label not in acronym_labels
                ],
            }
        )

    total_acronyms = sum(
        len(acronyms_by_concept["acronyms"])
        for acronyms_by_concept in acronyms_by_concept
    )
    console.print(
        f"Found {total_acronyms} acronyms in {len(acronyms_by_concept)} concepts with acronyms."
    )

    acronyms_path = (
        (get_git_root() or Path(__file__).parent.parent)
        / "data"
        / "processed"
        / "acronyms"
        / "acronyms.jsonl"
    )
    if not acronyms_path.parent.exists():
        acronyms_path.parent.mkdir(parents=True)

    acronyms_path.write_text(
        "\n".join(json.dumps(acronym) for acronym in acronyms_by_concept)
    )
    console.print(f"Acronyms saved to {acronyms_path}")

    # Create DataFrame with one acronym per row
    df_rows = []
    for concept in acronyms_by_concept:
        for acronym in concept["acronyms"]:
            df_rows.append(
                {
                    "wikibase_id": concept["wikibase_id"],
                    "preferred_label": concept["preferred_label"],
                    "acronym": acronym,
                    "non_acronym_labels": ", ".join(concept["non_acronym_labels"]),
                }
            )

    df = pd.DataFrame(df_rows)
    df["for_search"] = ""

    # Save to CSV
    csv_path = acronyms_path.parent / "acronyms.csv"
    df.to_csv(csv_path, index=False)
    console.print(f"Acronyms CSV saved to {csv_path}")


if __name__ == "__main__":
    get_acronyms()
