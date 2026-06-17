"""
Aggregate a multi-seed BERT classifier benchmark CSV and run paired significance tests.

Reads the per-(model x concept x seed) CSV produced by
``scripts/benchmarks/modernbert_models.py`` and:

1. Builds a per-(model, concept) summary table of mean +/- std (and n valid seeds)
   for ``passage_level_f1``, ``passage_level_pr_auc`` and ``passage_level_roc_auc``.
2. Pairs the reference model (``kdutia/cpr-ModernBERT``) against each baseline
   (``answerdotai/ModernBERT-base`` = "base",
   ``climatebert/distilroberta-base-climate-f`` = "CB") *by seed* and runs
   ``scipy.stats.wilcoxon`` and ``scipy.stats.ttest_rel`` per concept and pooled
   across all concepts. The pooled test is the headline / primary result; per-concept
   tests are indicative only because n (seeds) is small.

Only rows whose ``status`` is ``trained`` or ``cached`` have valid metrics; all other
rows are dropped before computing statistics.

Run with::

    uv run python scripts/benchmarks/seed_significance.py --input <csv>
    uv run python scripts/benchmarks/seed_significance.py --input results.csv --output out.csv
"""

from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from scipy import stats

from knowledge_graph.config import processed_data_dir
from knowledge_graph.utils import get_logger

console = Console()
app = typer.Typer()
logger = get_logger()

# --- Configuration -----------------------------------------------------------------

REFERENCE_MODEL = "kdutia/cpr-ModernBERT"

# Baselines to compare the reference model against, as {model_name: short_label}.
BASELINES: dict[str, str] = {
    "answerdotai/ModernBERT-base": "base",
    "climatebert/distilroberta-base-climate-f": "CB",
}

# Metrics to analyse (all higher-is-better).
METRICS: list[str] = [
    "passage_level_f1",
    "passage_level_pr_auc",
    "passage_level_roc_auc",
]

# Rows only carry valid metrics when training succeeded or was cached.
VALID_STATUSES: set[str] = {"trained", "cached"}

CAVEAT = (
    "Per-concept tests are INDICATIVE ONLY: n (seeds) is too small (typically 3) for "
    "Wilcoxon/t-test to reach significance. Trust the POOLED (primary) rows."
)


# --- Loading -----------------------------------------------------------------------


def load_valid(input_path: Path) -> pd.DataFrame:
    """Load the benchmark CSV and keep only rows with valid metrics."""
    df = pd.read_csv(input_path)
    before = len(df)
    df = df[df["status"].isin(VALID_STATUSES)].copy()
    logger.info(
        f"Loaded {before} rows from {input_path}; "
        f"{len(df)} have valid metrics (status in {sorted(VALID_STATUSES)})."
    )
    # Coerce metric columns to numeric so empty/NaN cells don't break stats.
    for metric in METRICS:
        if metric not in df.columns:
            logger.warning(f"Metric column '{metric}' missing from input; filling NaN.")
            df[metric] = np.nan
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce")
    return df


# --- Per-(model, concept) summary --------------------------------------------------


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Mean / std / n across seeds for each (model, concept) and metric."""
    records: list[dict] = []
    for (model_name, wikibase_id), group in df.groupby(["model_name", "wikibase_id"]):
        record: dict = {"model_name": model_name, "wikibase_id": wikibase_id}
        for metric in METRICS:
            values = group[metric].dropna()
            record[f"{metric}_mean"] = values.mean() if len(values) else np.nan
            # std over a single seed is NaN; ddof=1 (sample std) is fine here.
            record[f"{metric}_std"] = values.std(ddof=1) if len(values) > 1 else np.nan
            record[f"{metric}_n"] = int(len(values))
        records.append(record)
    summary = pd.DataFrame.from_records(records)
    if not summary.empty:
        summary = summary.sort_values(["wikibase_id", "model_name"]).reset_index(
            drop=True
        )
    return summary


def print_summary(summary: pd.DataFrame) -> None:
    """Print the per-(model, concept) mean +/- std table."""
    table = Table(
        title="Per-(model, concept) metrics: mean +/- std (n seeds)",
        show_lines=False,
    )
    table.add_column("Concept")
    table.add_column("Model")
    for metric in METRICS:
        table.add_column(metric.replace("passage_level_", ""), justify="right")

    def _fmt(mean: float, std: float, n: int) -> str:
        if pd.isna(mean):
            return "-"
        std_str = f"+/-{std:.3f}" if not pd.isna(std) else "+/-  -"
        return f"{mean:.3f} {std_str} (n={n})"

    for _, row in summary.iterrows():
        cells = [str(row["wikibase_id"]), str(row["model_name"])]
        for metric in METRICS:
            cells.append(
                _fmt(
                    row[f"{metric}_mean"],
                    row[f"{metric}_std"],
                    int(row[f"{metric}_n"]),
                )
            )
        table.add_row(*cells)

    console.print(table)


# --- Paired significance tests -----------------------------------------------------


def _paired_diffs(
    ref: pd.DataFrame, base: pd.DataFrame, metric: str
) -> tuple[np.ndarray, list[int]]:
    """
    Paired (reference - baseline) differences by seed for one metric.

    Returns the difference vector and the list of seeds used (those where BOTH models
    have a valid value for the metric).
    """
    ref_by_seed = ref.dropna(subset=[metric, "seed"]).set_index("seed")[metric]
    base_by_seed = base.dropna(subset=[metric, "seed"]).set_index("seed")[metric]
    # Average defensively in case of duplicate (model, concept, seed) rows.
    ref_by_seed = ref_by_seed.groupby(level=0).mean()
    base_by_seed = base_by_seed.groupby(level=0).mean()
    common = sorted(set(ref_by_seed.index) & set(base_by_seed.index))
    diffs = np.array(
        [ref_by_seed[seed] - base_by_seed[seed] for seed in common], dtype=float
    )
    return diffs, [int(s) for s in common]


def _run_tests(diffs: np.ndarray) -> tuple[float, float, float, str]:
    """
    Run wilcoxon + ttest_rel on a difference vector.

    Returns (mean_diff, wilcoxon_p, ttest_p, note). p-values are NaN when a test cannot
    be computed (too few samples, all-zero diffs, etc.), with the reason in ``note``.
    """
    n = len(diffs)
    if n == 0:
        return (np.nan, np.nan, np.nan, "no paired seeds")

    mean_diff = float(np.mean(diffs))
    note_parts: list[str] = []

    # Wilcoxon signed-rank, two-sided.
    try:
        if np.allclose(diffs, 0.0):
            wilcoxon_p = np.nan
            note_parts.append("wilcoxon: all diffs ~0")
        else:
            wilcoxon_p = float(stats.wilcoxon(diffs, alternative="two-sided").pvalue)
    except (ValueError, ZeroDivisionError) as exc:
        wilcoxon_p = np.nan
        note_parts.append(f"wilcoxon: {type(exc).__name__}")

    # Paired t-test. Needs >= 2 samples and non-zero variance.
    try:
        if n < 2:
            ttest_p = np.nan
            note_parts.append("ttest: n<2")
        else:
            ttest_p = float(stats.ttest_rel(diffs, np.zeros_like(diffs)).pvalue)
            if np.isnan(ttest_p):
                note_parts.append("ttest: NaN (zero variance?)")
    except (ValueError, ZeroDivisionError) as exc:
        ttest_p = np.nan
        note_parts.append(f"ttest: {type(exc).__name__}")

    return (mean_diff, wilcoxon_p, ttest_p, "; ".join(note_parts))


def paired_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run per-concept and pooled paired tests for every baseline x metric.

    Reference = REFERENCE_MODEL. For each baseline, pairs differences by seed within a
    concept, then also pools the per-seed diffs across all concepts.
    """
    records: list[dict] = []
    ref_all = df[df["model_name"] == REFERENCE_MODEL]
    if ref_all.empty:
        logger.warning(
            f"Reference model '{REFERENCE_MODEL}' has no valid rows; "
            "skipping all paired tests."
        )
        return pd.DataFrame(
            columns=[
                "baseline",
                "baseline_model",
                "metric",
                "scope",
                "mean_diff",
                "wilcoxon_p",
                "ttest_p",
                "n",
                "note",
            ]
        )

    concepts = sorted(df["wikibase_id"].unique())

    for base_model, base_label in BASELINES.items():
        base_all = df[df["model_name"] == base_model]
        if base_all.empty:
            logger.warning(
                f"Baseline '{base_model}' ({base_label}) has no valid rows; skipping."
            )
            continue

        for metric in METRICS:
            pooled_diffs: list[float] = []

            for wikibase_id in concepts:
                ref = ref_all[ref_all["wikibase_id"] == wikibase_id]
                base = base_all[base_all["wikibase_id"] == wikibase_id]
                if ref.empty or base.empty:
                    continue
                diffs, _seeds = _paired_diffs(ref, base, metric)
                if len(diffs) == 0:
                    continue
                pooled_diffs.extend(diffs.tolist())
                mean_diff, wilcoxon_p, ttest_p, note = _run_tests(diffs)
                records.append(
                    {
                        "baseline": base_label,
                        "baseline_model": base_model,
                        "metric": metric,
                        "scope": f"per-concept:{wikibase_id}",
                        "mean_diff": mean_diff,
                        "wilcoxon_p": wilcoxon_p,
                        "ttest_p": ttest_p,
                        "n": len(diffs),
                        "note": (
                            "indicative only (n is small)"
                            + (f"; {note}" if note else "")
                        ),
                    }
                )

            # Pooled across concepts (the primary headline test).
            pooled = np.array(pooled_diffs, dtype=float)
            mean_diff, wilcoxon_p, ttest_p, note = _run_tests(pooled)
            records.append(
                {
                    "baseline": base_label,
                    "baseline_model": base_model,
                    "metric": metric,
                    "scope": "pooled",
                    "mean_diff": mean_diff,
                    "wilcoxon_p": wilcoxon_p,
                    "ttest_p": ttest_p,
                    "n": len(pooled),
                    "note": ("pooled (primary)" + (f"; {note}" if note else "")),
                }
            )

    return pd.DataFrame.from_records(records)


def print_tests(tests: pd.DataFrame) -> None:
    """Print the paired-test results, with pooled rows visually distinct."""
    table = Table(
        title=f"Paired tests: {REFERENCE_MODEL} vs baselines (reference - baseline)",
        show_lines=False,
    )
    table.add_column("Baseline")
    table.add_column("Metric")
    table.add_column("Scope")
    table.add_column("mean_diff", justify="right")
    table.add_column("wilcoxon_p", justify="right")
    table.add_column("ttest_p", justify="right")
    table.add_column("n", justify="right")

    def _fmt_p(value: float) -> str:
        return "NaN" if pd.isna(value) else f"{value:.4f}"

    def _fmt_diff(value: float) -> str:
        return "-" if pd.isna(value) else f"{value:+.4f}"

    # Order so each (baseline, metric) block ends with its pooled row.
    if not tests.empty:
        tests = tests.assign(
            _scope_order=tests["scope"].eq("pooled").astype(int)
        ).sort_values(["baseline", "metric", "_scope_order", "scope"])

    for _, row in tests.iterrows():
        is_pooled = row["scope"] == "pooled"
        style = "bold green" if is_pooled else "dim"
        scope_label = "POOLED (primary)" if is_pooled else row["scope"]
        table.add_row(
            str(row["baseline"]),
            row["metric"].replace("passage_level_", ""),
            scope_label,
            _fmt_diff(row["mean_diff"]),
            _fmt_p(row["wilcoxon_p"]),
            _fmt_p(row["ttest_p"]),
            str(int(row["n"])),
            style=style,
        )

    console.print(table)
    console.print(f"[yellow]Note:[/yellow] {CAVEAT}")


# --- Output ------------------------------------------------------------------------


def write_outputs(
    summary: pd.DataFrame, tests: pd.DataFrame, output_path: Path
) -> Path:
    """
    Write the paired-test results to ``output_path`` and the summary alongside it.

    Returns the summary path. Writes both files immediately (flushed by pandas) so
    partial output isn't lost.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Prepend the caveat as a comment line so it travels with the results CSV.
    with open(output_path, "w", newline="") as fh:
        fh.write(f"# {CAVEAT}\n")
        tests.to_csv(fh, index=False)

    summary_path = output_path.with_name(
        output_path.stem + "_summary" + output_path.suffix
    )
    summary.to_csv(summary_path, index=False)

    logger.info(f"Wrote paired tests to {output_path}")
    logger.info(f"Wrote per-(model, concept) summary to {summary_path}")
    return summary_path


# --- Main --------------------------------------------------------------------------


@app.command()
def main(
    input: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Benchmark CSV from modernbert_models.py (one row per model x concept x seed).",
        ),
    ],
    output: Annotated[
        Optional[Path],
        typer.Option(help="Path to write the paired-test results CSV."),
    ] = None,
) -> None:
    """Aggregate seeds and run paired significance tests on a benchmark CSV."""
    output_path = output or (
        processed_data_dir / "benchmarks" / "seed_significance.csv"
    )

    df = load_valid(input)
    if df.empty:
        console.print(
            "[red]No rows with valid metrics found "
            f"(status in {sorted(VALID_STATUSES)}). Nothing to do.[/red]"
        )
        raise typer.Exit(code=1)

    summary = summarise(df)
    print_summary(summary)

    tests = paired_tests(df)
    if tests.empty:
        console.print(
            "[red]No paired comparisons could be computed "
            "(missing reference or baseline rows).[/red]"
        )
    else:
        print_tests(tests)

    summary_path = write_outputs(summary, tests, output_path)
    console.log(f"Done. Tests -> {output_path}; summary -> {summary_path}")


if __name__ == "__main__":
    app()
