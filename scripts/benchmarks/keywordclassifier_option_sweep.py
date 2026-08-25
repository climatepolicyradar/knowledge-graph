r"""
Sweep the optional KeywordClassifier matching relaxations across many concepts.

KeywordClassifier gained two optional relaxations - subscript folding and plural word
forms - both off by default. This script measures what each one actually buys, per
concept, against the human-labelled passages in Argilla.

For every concept listed in ``vibe-checker/config.yml`` it builds one classifier per
option combination, evaluates it against that concept's gold passages, and appends the
results to a CSV as it goes, so an interrupted run keeps everything it had finished.

Nothing is logged to Weights & Biases: ``evaluate_classifier`` is called with
``wandb_run=None``, which skips all of its W&B logging, and this script never creates a
run. (``knowledge_graph.operations.evaluate`` imports the wandb library at module
scope, so the import itself is unavoidable - it is simply never used.)

Only precision, recall and F1 are reported, at passage level and at span level (0.5).

Example::

    uv run python -m scripts.benchmarks.keyword_option_sweep
    uv run python -m scripts.benchmarks.keyword_option_sweep --concepts Q787,Q368
"""

import asyncio
import csv
import time
from pathlib import Path
from typing import Annotated, Any, cast

import pandas as pd
import typer
import yaml
from rich.console import Console
from rich.table import Table

from knowledge_graph.classifier.keyword import KeywordClassifier
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.operations.evaluate import evaluate_classifier
from knowledge_graph.operations.get_concept import get_concept_async

console = Console()

DEFAULT_CONCEPTS_FILE = Path("vibe-checker/config.yml")
DEFAULT_OUTPUT_DIR = Path("scripts/benchmarks/keywordclassifier_option_sweep_results")

# The agreement levels we care about, out of the several that evaluate_classifier
# reports. Passage level answers "did we find the concept at all", span level (0.5)
# answers "did we find it in roughly the right place".
AGREEMENT_LEVELS = ["Passage level", "Span level (0.5)"]

# The option combinations under test. "default" is the control, and must stay first so
# that every other variant can be compared against it.
VARIANTS: dict[str, dict[str, Any]] = {
    "default": {},
    "fold_subscripts": {"fold_subscripts": True},
    "match_word_forms": {"match_word_forms": True},
    "all_on": {"fold_subscripts": True, "match_word_forms": True},
}

CSV_COLUMNS = [
    "wikibase_id",
    "preferred_label",
    "variant",
    "fold_subscripts",
    "match_word_forms",
    "classifier_id",
    "agreement_level",
    "precision",
    "recall",
    "f1",
    "support",
    "n_passages",
    "predict_seconds",
]

app = typer.Typer()


def load_wikibase_ids(path: Path) -> list[WikibaseID]:
    """Read the flat list of Wikibase IDs from a concepts config file."""
    with open(path) as file:
        raw_ids = yaml.safe_load(file)

    return [WikibaseID(str(raw_id)) for raw_id in raw_ids]


class ResultWriter:
    """Appends result rows to a CSV, writing the header only for a fresh file."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._wrote_header = self.path.exists() and self.path.stat().st_size > 0

    def write(self, rows: list[dict[str, Any]]) -> None:
        """Append rows and flush immediately, so nothing is lost if the run dies."""
        with open(self.path, "a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
            if not self._wrote_header:
                writer.writeheader()
                self._wrote_header = True
            writer.writerows(rows)


def evaluate_variant(
    concept: Concept, variant: str, kwargs: dict[str, Any]
) -> list[dict[str, Any]]:
    """
    Evaluate one option combination for one concept.

    :return list[dict[str, Any]]: One row per agreement level of interest
    """
    classifier = KeywordClassifier(concept, **kwargs)

    # Time a bare predict pass over the same texts, so that the cost of each option
    # (fuzzy matching in particular) is directly comparable across variants.
    texts = [passage.text for passage in concept.labelled_passages]
    started_at = time.perf_counter()
    classifier.predict(texts)
    predict_seconds = time.perf_counter() - started_at

    metrics, _, _ = evaluate_classifier(
        classifier, concept.labelled_passages, wandb_run=None
    )

    overall = cast(pd.DataFrame, metrics[metrics["Group"] == "all"])
    by_agreement_level = {row["Agreement at"]: row for _, row in overall.iterrows()}
    rows = []
    for agreement_level in AGREEMENT_LEVELS:
        row = by_agreement_level.get(agreement_level)
        if row is None:
            console.log(
                f"[yellow]No '{agreement_level}' metrics for {concept.wikibase_id} "
                f"({variant})[/yellow]"
            )
            continue

        rows.append(
            {
                "wikibase_id": str(concept.wikibase_id),
                "preferred_label": concept.preferred_label,
                "variant": variant,
                "fold_subscripts": classifier.fold_subscripts,
                "match_word_forms": classifier.match_word_forms,
                "classifier_id": str(classifier.id),
                "agreement_level": agreement_level,
                "precision": row["Precision"],
                "recall": row["Recall"],
                "f1": row["F1 score"],
                "support": row["Support"],
                "n_passages": len(concept.labelled_passages),
                "predict_seconds": round(predict_seconds, 3),
            }
        )

    return rows


def print_summary(rows: list[dict[str, Any]], skipped: list[tuple[str, str]]) -> None:
    """
    Print per-variant F1 deltas against each concept's own default.

    The per-concept view matters more than the mean: an option which lifts F1 on
    average can still wreck a handful of concepts, and those are the ones that would
    need to opt out.
    """
    for agreement_level in AGREEMENT_LEVELS:
        at_level = [row for row in rows if row["agreement_level"] == agreement_level]
        baselines = {
            row["wikibase_id"]: row["f1"]
            for row in at_level
            if row["variant"] == "default"
        }

        table = Table(title=f"F1 change vs default - {agreement_level}", box=None)
        for column in ["Variant", "Mean ΔF1", "Better", "Same", "Worse", "Mean s"]:
            table.add_column(column, justify="right" if column != "Variant" else "left")

        for variant in VARIANTS:
            if variant == "default":
                continue

            deltas = [
                row["f1"] - baselines[row["wikibase_id"]]
                for row in at_level
                if row["variant"] == variant and row["wikibase_id"] in baselines
            ]
            seconds = [
                row["predict_seconds"] for row in at_level if row["variant"] == variant
            ]
            if not deltas:
                continue

            table.add_row(
                variant,
                f"{sum(deltas) / len(deltas):+.4f}",
                str(sum(1 for delta in deltas if delta > 1e-9)),
                str(sum(1 for delta in deltas if abs(delta) <= 1e-9)),
                str(sum(1 for delta in deltas if delta < -1e-9)),
                f"{sum(seconds) / len(seconds):.2f}" if seconds else "-",
            )

        console.print(table)
        console.print()

        regressions = sorted(
            (
                (row["f1"] - baselines[row["wikibase_id"]], row)
                for row in at_level
                if row["variant"] != "default" and row["wikibase_id"] in baselines
            ),
            key=lambda pair: pair[0],
        )[:5]

        if regressions and regressions[0][0] < -1e-9:
            console.print(f"[bold]Largest regressions ({agreement_level})[/bold]")
            for delta, row in regressions:
                if delta >= -1e-9:
                    continue
                console.print(
                    f"  {delta:+.4f}  {row['wikibase_id']} "
                    f"({row['preferred_label']}) via {row['variant']}"
                )
            console.print()

    if skipped:
        console.print(f"[bold yellow]Skipped {len(skipped)} concepts[/bold yellow]")
        for wikibase_id, reason in skipped:
            console.print(f"  {wikibase_id}: {reason}")


async def run_sweep(
    wikibase_ids: list[WikibaseID], output_path: Path
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Evaluate every variant for every concept, writing results out as they land."""
    writer = ResultWriter(output_path)
    all_rows: list[dict[str, Any]] = []
    skipped: list[tuple[str, str]] = []

    for index, wikibase_id in enumerate(wikibase_ids, start=1):
        console.rule(f"[{index}/{len(wikibase_ids)}] {wikibase_id}")

        try:
            concept = await get_concept_async(wikibase_id)
        except Exception as error:
            console.log(f"[red]Could not fetch {wikibase_id}: {error}[/red]")
            skipped.append((str(wikibase_id), f"fetch failed: {error}"))
            continue

        if not concept.labelled_passages:
            console.log(f"[yellow]{wikibase_id} has no labelled passages[/yellow]")
            skipped.append((str(wikibase_id), "no labelled passages"))
            continue

        for variant, kwargs in VARIANTS.items():
            try:
                rows = evaluate_variant(concept, variant, kwargs)
            except Exception as error:
                console.log(f"[red]{wikibase_id} / {variant} failed: {error}[/red]")
                skipped.append((str(wikibase_id), f"{variant} failed: {error}"))
                continue

            writer.write(rows)
            all_rows.extend(rows)

            for row in rows:
                if row["agreement_level"] == "Passage level":
                    console.log(
                        f"{variant:<18} P={row['precision']:.3f} "
                        f"R={row['recall']:.3f} F1={row['f1']:.3f} "
                        f"({row['predict_seconds']}s)"
                    )

    return all_rows, skipped


@app.command()
def main(
    concepts_file: Annotated[
        Path,
        typer.Option(help="YAML file holding a flat list of Wikibase IDs"),
    ] = DEFAULT_CONCEPTS_FILE,
    concepts: Annotated[
        str | None,
        typer.Option(help="Comma-separated Wikibase IDs, overriding --concepts-file"),
    ] = None,
    output_dir: Annotated[
        Path, typer.Option(help="Directory to write results.csv into")
    ] = DEFAULT_OUTPUT_DIR,
    limit: Annotated[
        int | None, typer.Option(help="Only sweep the first N concepts")
    ] = None,
):
    """Sweep the KeywordClassifier match options across a set of concepts."""
    if concepts:
        wikibase_ids = [
            WikibaseID(value.strip()) for value in concepts.split(",") if value.strip()
        ]
    else:
        wikibase_ids = load_wikibase_ids(concepts_file)

    if limit is not None:
        console.log(f"Limiting the sweep to the first {limit} of {len(wikibase_ids)}")
        wikibase_ids = wikibase_ids[:limit]

    output_path = output_dir / "results.csv"
    console.log(
        f"Sweeping {len(VARIANTS)} variants across {len(wikibase_ids)} concepts "
        f"into {output_path}"
    )

    rows, skipped = asyncio.run(run_sweep(wikibase_ids, output_path))

    console.print()
    print_summary(rows, skipped)


if __name__ == "__main__":
    app()
