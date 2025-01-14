"""Methods for working with CSVs of external keywords."""

import csv
from pathlib import Path

from scripts.config import csv_keywords_dir
from src.concept import Concept


def load_keywords_from_csv(csv_path: Path) -> list[str]:
    """
    Load keywords from a CSV file.

    Extracts values from the 'itemLabel' column and any columns starting with 'Alias'.
    Empty values are filtered out.
    """
    keywords = set()

    if not csv_path.suffix == ".csv" or not csv_path.is_file():
        raise ValueError(f"File must be a CSV: {csv_path}")

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "itemLabel" in row and row["itemLabel"]:
                keywords.add(row["itemLabel"])

            alias_cols = [col for col in row.keys() if col.startswith("Alias")]
            for col in alias_cols:
                if row[col]:
                    keywords.add(row[col])

    keywords = list(set(keywords))

    return list(keywords)


def load_concept_keywords_from_csv(concept: Concept) -> list[str]:
    """
    Load external keywords for a concept.

    :raises FileNotFoundError: if the CSV file for the concept does not exist
    """

    csv_path = csv_keywords_dir / f"{concept.wikibase_id}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    return load_keywords_from_csv(csv_path)
