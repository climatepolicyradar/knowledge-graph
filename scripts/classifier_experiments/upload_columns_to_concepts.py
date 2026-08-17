"""
Upload spreadsheet columns to concept store as alternative labels.

Supports two independent modes -- pick whichever fits the CSV you have:

1. Single column -> one concept. Provide a column name and its corresponding
   concept ID; all unique values from that column become alternative labels
   for that concept, verbatim. Use `column_to_concept_id` +
   `upload_columns_to_concepts`.
2. Cross-product of several columns -> one concept. E.g. a verb column
   ("reduce", "reducing", ...) crossed with a noun-phrase column
   ("emissions", "carbon", ...) produces "reduce emissions", "reducing
   carbon", etc. Use `combo_specs` + `upload_combined_columns_to_concepts`.
   A concept ID can appear in more than one combo (add_alternative_labels is
   additive and de-duplicates, so repeated uploads to the same concept just
   accumulate labels).

Both modes read each column as-is, only stripping whitespace & empty values.
Note especially that capitalization and punctuation is preserved, which
matters bc capitalised entries will be treated as case-sensitive by
classifiers, whereas uncapitalised entries will be treated as case-insensitive.

Use UTF-8 encoding for csv files by default.

Workflow
--------
This script has no CLI args -- all configuration lives in the `__main__`
block below. Each time you have a new upload task:
1. Edit `__main__`: set `csv_path` to your CSV, and replace
   `column_to_concept_id` (mode 1) or `combo_specs` (mode 2) with your own
   column-name -> concept-ID mapping. Comment out whichever mode you're not
   using, and its matching upload_*_to_concepts() call below it.
2. Save the file.
3. Run it from the repo root:
    uv run python scripts/classifier_experiments/upload_columns_to_concepts.py
There's no dry-run mode, so double check column names and concept IDs before
running -- though uploads are additive/de-duplicating, so re-running after a
partial failure is safe.
"""

import logging
import time
from itertools import product
from pathlib import Path

import pandas as pd

from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.wikibase import WikibaseSession

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default encoding for spreadsheets with symbols (e.g. currency)
DEFAULT_ENCODING = "utf-8"


def load_column_values(
    file_path: str | Path,
    columns: list[str],
    *,
    encoding: str = DEFAULT_ENCODING,
    dtype: type = str,
) -> dict[str, list[str]]:
    """
    Load a CSV/spreadsheet and return unique non-empty values per column.

    Args:
        file_path: Path to CSV (or Excel if you use read_excel below).
        columns: Column names to read.
        encoding: Text encoding (default UTF-8 for currency symbols etc.).
        dtype: Type for string columns (default str to preserve symbols).

    Returns:
        Dict mapping each column name to list of unique, stripped, non-empty values.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path, dtype=dtype, encoding=encoding)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path, dtype=dtype)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .csv, .xlsx, or .xls.")

    for col in columns:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

    result: dict[str, list[str]] = {}
    for col in columns:
        series = df[col].astype(str).str.strip()
        series = series.replace("nan", "").replace("NaN", "")
        unique = series.dropna().loc[series != ""].unique().tolist()
        result[col] = unique
        logger.info(f"Column '{col}': {len(result[col])} unique values")

    return result


def upload_columns_to_concepts(
    file_path: str | Path,
    column_to_concept_id: list[tuple[str, str]],
    *,
    encoding: str = DEFAULT_ENCODING,
) -> None:
    """
    Upload each column's unique values as alternative labels to the concept store.

    For each (column_name, concept_id) pair, all unique values from that
    column become alternative labels for the corresponding concept.

    Args:
        file_path: Path to CSV or Excel file.
        column_to_concept_id: List of (column_name, concept_id) in upload order.
        encoding: File encoding (default UTF-8).
    """
    if not column_to_concept_id:
        logger.warning("No column -> concept_id mappings given.")
        return

    columns = [c for c, _ in column_to_concept_id]
    concept_ids = [cid for _, cid in column_to_concept_id]

    column_values = load_column_values(file_path, columns, encoding=encoding)
    keyword_lists = [column_values[col] for col in columns]

    logger.info("Uploading to Wikibase...")
    upload_keyword_lists_to_wikibase(keyword_lists, concept_ids)


def combine_columns(
    column_values: dict[str, list[str]],
    columns: list[str],
    separator: str = " ",
) -> list[str]:
    """
    Cross-product the unique values of several columns into phrases.

    Every value in the first column is combined with every value in every
    other column, e.g. a verb column x a noun-phrase column ->
    "reduce" x "emissions" -> "reduce emissions". Values are assumed already
    stripped/deduped per column (see load_column_values).

    Args:
        column_values: Dict mapping column name to its unique values, as
            returned by load_column_values.
        columns: Column names to cross, in the order they should be joined.
        separator: String to join the parts of each phrase with.

    Returns:
        List of combined phrases (not deduped or filtered further).
    """
    lists = [column_values[col] for col in columns]
    combined = [separator.join(combo) for combo in product(*lists)]
    logger.info(f"Combined {' x '.join(columns)}: {len(combined)} phrases")
    return combined


def upload_combined_columns_to_concepts(
    file_path: str | Path,
    combo_specs: list[tuple[list[str], str]],
    *,
    encoding: str = DEFAULT_ENCODING,
) -> None:
    """
    Cross-product each combo's columns and upload the phrases to its concept.

    For each (columns, concept_id) combo, cross-product the given columns'
    unique values into phrases and upload them as alternative labels for
    that concept.

    A concept_id can appear in more than one combo (e.g. one concept gets
    both a "reduce x emissions" combo and an "increase x sinks" combo) --
    each combo is uploaded separately; add_alternative_labels is additive
    and de-duplicates, so repeated concept_ids just accumulate labels.

    Args:
        file_path: Path to CSV or Excel file.
        combo_specs: List of (columns_to_combine, concept_id), in upload order.
        encoding: File encoding (default UTF-8).
    """
    if not combo_specs:
        logger.warning("No column combinations given.")
        return

    all_columns = sorted({col for cols, _ in combo_specs for col in cols})
    column_values = load_column_values(file_path, all_columns, encoding=encoding)

    concept_ids: list[str] = []
    keyword_lists: list[list[str]] = []
    for columns, concept_id in combo_specs:
        concept_ids.append(concept_id)
        keyword_lists.append(combine_columns(column_values, columns))

    logger.info("Uploading combined phrases to Wikibase...")
    upload_keyword_lists_to_wikibase(keyword_lists, concept_ids)


def upload_keyword_lists_to_wikibase(
    keyword_lists: list[list[str]], concept_ids: list[str]
) -> None:
    """
    Upload keyword lists to Wikibase, one list per concept ID.

    Each keyword list is uploaded as alternative labels for the corresponding
    concept ID (parallel lists, matched by position). Uses a fresh session per
    upload and a short delay between calls to avoid overwhelming the server.
    add_alternative_labels is additive and de-duplicates server-side, so the
    same concept_id can safely appear more than once across the two lists
    (e.g. when a concept has multiple combos).
    """
    assert len(keyword_lists) == len(concept_ids)

    for i, (keywords, concept_id) in enumerate(zip(keyword_lists, concept_ids)):
        try:
            logger.info(
                f"\n---\nUploading {len(keywords)} keywords to concept {concept_id}"
            )

            valid_keywords = [kw for kw in keywords if kw and kw.strip()]
            if not valid_keywords:
                logger.warning(f"No valid keywords to upload for {concept_id}")
                continue

            try:
                session = WikibaseSession()
                logger.info("Successfully connected to Wikibase")
            except Exception as e:
                logger.error(f"Failed to connect to Wikibase: {e}")
                continue

            session.add_alternative_labels(
                wikibase_id=WikibaseID(concept_id),
                alternative_labels=valid_keywords,
            )
            logger.info(f"Successfully uploaded {concept_id}")

            # Small delay between uploads to avoid overwhelming the server
            if i < len(keyword_lists) - 1:
                time.sleep(3)

        except Exception as e:
            logger.error(f"\n!!!\nError uploading {concept_id} to Wikibase: {e}")
            # Continue with next upload even if one fails
            continue


def print_concept_label_count(concept_id: str) -> None:
    """Fetch concept from Wikibase and print its alternative label count."""
    session = WikibaseSession()
    concept = session.get_concept(WikibaseID(concept_id))
    n = len(concept.alternative_labels)
    logger.info(
        "Concept %s (%s): %d alternative labels", concept_id, concept.preferred_label, n
    )
    print(f"Concept {concept_id} ({concept.preferred_label}): {n} alternative labels")


if __name__ == "__main__":
    # --- Mode 1: single column -> one concept (uncomment to use instead) ---
    # csv_path = Path("data/Currency Data - ISO 4217 - Enhanced for Search - Full.csv")
    # column_to_concept_id: list[tuple[str, str]] = [
    #     ("Code", "Q2033"),
    #     ("Common Name", "Q2033"),
    #     ("Plural", "Q2033"),
    #     ("Symbol", "Q2033"),
    # ]
    # if csv_path.exists() and column_to_concept_id:
    #     upload_columns_to_concepts(csv_path, column_to_concept_id)
    #     for cid in sorted(set(cid for _, cid in column_to_concept_id)):
    #         print_concept_label_count(cid)

    # --- Mode 2: cross-product of columns -> one concept (currently active) ---
    # Example from the energy mitigation keywords task: cross-product a
    # verb/modifier column with one or more noun-phrase columns, then upload
    # the resulting phrases. Replace csv_path and combo_specs for your own task.
    csv_path = Path("data/Energy_mitigation_keywords.csv")

    # Each entry: (columns to cross-product, concept ID to upload the result to).
    # A concept ID may appear more than once (Q560 combines two separate
    # verb x noun-phrase pairs) -- both combos get uploaded to the same concept.
    combo_specs: list[tuple[list[str], str]] = [
        (["reduce_long", "Q560 mitigation - emissions"], "Q560"),
        (["increase", "Q560 mitigation - sinks"], "Q560"),
        (["mitigate", "Q2409 energy mitigation"], "Q2409"),
        (
            ["mitigate", "Q2444 reduce emissions from fossil fuel energy generation"],
            "Q2444",
        ),
        (["mitigate", "Q2483 emissions reduction from coal mining"], "Q2483"),
        (["mitigate", "Q2482 emissions reduction from oil and gas mining"], "Q2482"),
        (["mitigate", "Q2413 transport mitigation"], "Q2413"),
        (["update", "Q2423 modernisation of grids"], "Q2423"),
        (["electrify", "Q2450 electric cooker"], "Q2450"),
        (["electrify", "Q2451 electric heating"], "Q2451"),
        (["electrify", "Q2453 electric cooling"], "Q2453"),
        (["electrify", "Q1496 transport electrification"], "Q1496"),
        (["electrify", "Q2448 industry electrification"], "Q2448"),
    ]

    if csv_path.exists() and combo_specs:
        upload_combined_columns_to_concepts(csv_path, combo_specs)
        # Check length of concept(s) after upload
        for cid in sorted(set(cid for _, cid in combo_specs)):
            print_concept_label_count(cid)
    else:
        if not csv_path.exists():
            logger.error("CSV not found: %s", csv_path)
        if not combo_specs:
            logger.info(
                "Edit combo_specs in this script with (columns, concept_id) pairs, then run again."
            )
