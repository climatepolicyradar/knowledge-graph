r"""
Benchmark several LLMs on custom-classifier YAML configs against gold passages.

For every config it builds one LLMClassifier per model, using that config's own system
prompt, labelling guidelines and related definitions, with only ``model_name`` swapped.
The config's own model is always included as the baseline. Each classifier is evaluated
against the concept's human-labelled passages in Argilla, and results are appended to a
CSV as they land, so an interrupted run resumes by skipping finished (concept, model)
pairs.

Nothing is logged to Weights & Biases and nothing is uploaded: ``evaluate_classifier`` is
called with ``wandb_run=None``.

Two extra columns guard against over-reading small gold sets:

- ``f1_ci_low`` / ``f1_ci_high``: a bootstrap 95% interval (over passages) on the
  passage-level F1. Gaps whose intervals overlap should be treated as ties.
- ``n_failed_predictions``: passages where the LLM's response couldn't be turned into
  spans (bad XML, text not reproduced exactly, retries exhausted). These silently
  become "no spans", so a model that can't follow the output format looks like a
  low-recall model rather than a broken one.

Only precision, recall and F1 are reported, at passage level and span level (0.5).

Example::

    uv run python -m scripts.benchmarks.llm_model_sweep \
        --configs scripts/custom_concept_training/configs/q32.yaml \
        --models openrouter:z-ai/glm-5.3,openrouter:anthropic/claude-opus-5.5
"""

import asyncio
import csv
import logging
import time
from pathlib import Path
from typing import Annotated, Any, cast

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from knowledge_graph.classifier.large_language_model import LLMClassifier
from knowledge_graph.concept import Concept
from knowledge_graph.custom_classifier_config import CustomClassifierConfig
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.openrouter_pricing import get_openrouter_pricing
from knowledge_graph.operations.evaluate import (
    create_gold_standard_labelled_passages,
    evaluate_classifier,
)
from knowledge_graph.operations.get_concept import get_concept_async
from knowledge_graph.wikibase import WikibaseSession

console = Console()

DEFAULT_CONFIGS = [
    Path(f"scripts/custom_concept_training/configs/{name}.yaml")
    for name in ["q32", "q911", "q912", "q1829"]
]
DEFAULT_MODELS = [
    "openrouter:moonshotai/kimi-k3",
    "openrouter:deepseek/deepseek-v4-pro-0813",
    "openrouter:anthropic/claude-opus-5.5",
]
DEFAULT_OUTPUT_DIR = Path("scripts/benchmarks/llm_model_sweep_results")

AGREEMENT_LEVELS = ["Passage level", "Span level (0.5)"]
N_BOOTSTRAP = 1000

# Above this share of failed predictions a run isn't written out, so a resumed sweep
# retries it. Failures (API errors like exhausted credit included) silently become "no
# spans", so without this they'd be recorded as a genuinely low-recall model.
MAX_FAILED_SHARE = 0.05

# Models that reject the forced tool call pydantic-ai uses for structured output
# ("tool_choice: type tool and any are not supported"), so they return plain marked-up
# text instead. This is a confound against the other models, so it is recorded per row.
PLAIN_TEXT_MODELS = {"openrouter:anthropic/claude-opus-5.5"}

CSV_COLUMNS = [
    "wikibase_id",
    "preferred_label",
    "model_name",
    "is_baseline",
    "structured_output",
    "agreement_level",
    "precision",
    "recall",
    "f1",
    "f1_ci_low",
    "f1_ci_high",
    "support",
    "n_passages",
    "n_failed_predictions",
    "predict_seconds",
    "prompt_usd_per_mtok",
    "completion_usd_per_mtok",
]

app = typer.Typer()


class FailureCounter(logging.Handler):
    """
    Counts the warnings LLMClassifier logs when a response can't be turned into spans.

    Every failure path in its predict methods logs a warning and returns no spans, so
    counting warnings on its logger counts failed predictions.
    """

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        """Count the record."""
        if "failed" in record.getMessage():
            self.count += 1


class ResultWriter:
    """Appends result rows to a CSV, writing the header only for a fresh file."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._wrote_header = self.path.exists() and self.path.stat().st_size > 0

    def done(self) -> set[tuple[str, str]]:
        """The (wikibase_id, model_name) pairs already in the CSV."""
        if not self._wrote_header:
            return set()
        existing = pd.read_csv(self.path)
        return set(zip(existing["wikibase_id"], existing["model_name"]))

    def write(self, rows: list[dict[str, Any]]) -> None:
        """Append rows and flush immediately, so nothing is lost if the run dies."""
        with open(self.path, "a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
            if not self._wrote_header:
                writer.writeheader()
                self._wrote_header = True
            writer.writerows(rows)


def bootstrap_passage_f1(
    gold: list[LabelledPassage], predicted: list[LabelledPassage], seed: int = 0
) -> tuple[float, float]:
    """Bootstrap 95% interval on passage-level F1, resampling passages."""
    predicted_by_id = {passage.id: passage for passage in predicted}
    y_true = np.array([len(passage.spans) > 0 for passage in gold])
    y_pred = np.array([len(predicted_by_id[passage.id].spans) > 0 for passage in gold])

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(gold), size=(N_BOOTSTRAP, len(gold)))
    t, p = y_true[indices], y_pred[indices]
    tp = (t & p).sum(axis=1)
    fp = (~t & p).sum(axis=1)
    fn = (t & ~p).sum(axis=1)
    denominator = 2 * tp + fp + fn
    f1 = np.where(denominator > 0, 2 * tp / np.maximum(denominator, 1), 0.0)

    low, high = np.percentile(f1, [2.5, 97.5])
    return float(low), float(high)


async def evaluate_model(
    concept: Concept,
    base_kwargs: dict[str, Any],
    model_name: str,
    is_baseline: bool,
) -> list[dict[str, Any]]:
    """
    Evaluate one model for one concept.

    :return list[dict[str, Any]]: One row per agreement level of interest
    """
    structured_output = model_name not in PLAIN_TEXT_MODELS
    classifier = LLMClassifier(
        concept,
        **{
            **base_kwargs,
            "model_name": model_name,
            "structured_output": structured_output,
        },
    )
    pricing = await get_openrouter_pricing(model_name)

    counter = FailureCounter()
    llm_logger = logging.getLogger("knowledge_graph.classifier.large_language_model")
    llm_logger.addHandler(counter)
    try:
        started_at = time.perf_counter()
        metrics, model_labelled_passages, _ = evaluate_classifier(
            classifier, concept.labelled_passages, wandb_run=None, batch_size=50
        )
        predict_seconds = time.perf_counter() - started_at
    finally:
        llm_logger.removeHandler(counter)

    gold = create_gold_standard_labelled_passages(concept.labelled_passages)
    f1_ci_low, f1_ci_high = bootstrap_passage_f1(gold, model_labelled_passages)

    overall = cast(pd.DataFrame, metrics[metrics["Group"] == "all"])
    by_agreement_level = {row["Agreement at"]: row for _, row in overall.iterrows()}
    rows = []
    for agreement_level in AGREEMENT_LEVELS:
        row = by_agreement_level.get(agreement_level)
        if row is None:
            console.log(
                f"[yellow]No '{agreement_level}' metrics for {concept.wikibase_id} "
                f"({model_name})[/yellow]"
            )
            continue

        is_passage_level = agreement_level == "Passage level"
        rows.append(
            {
                "wikibase_id": str(concept.wikibase_id),
                "preferred_label": concept.preferred_label,
                "model_name": model_name,
                "is_baseline": is_baseline,
                "structured_output": structured_output,
                "agreement_level": agreement_level,
                "precision": row["Precision"],
                "recall": row["Recall"],
                "f1": row["F1 score"],
                "f1_ci_low": round(f1_ci_low, 4) if is_passage_level else None,
                "f1_ci_high": round(f1_ci_high, 4) if is_passage_level else None,
                "support": row["Support"],
                "n_passages": len(concept.labelled_passages),
                "n_failed_predictions": counter.count,
                "predict_seconds": round(predict_seconds, 1),
                "prompt_usd_per_mtok": (
                    round(pricing.prompt_price * 1e6, 3) if pricing else None
                ),
                "completion_usd_per_mtok": (
                    round(pricing.completion_price * 1e6, 3) if pricing else None
                ),
            }
        )

    return rows


def print_summary(rows: list[dict[str, Any]], skipped: list[tuple[str, str]]) -> None:
    """Print passage-level P/R/F1 (with CI) per concept and model, plus a macro F1."""
    passage_rows = [row for row in rows if row["agreement_level"] == "Passage level"]
    span_f1 = {
        (row["wikibase_id"], row["model_name"]): row["f1"]
        for row in rows
        if row["agreement_level"] == "Span level (0.5)"
    }

    table = Table(title="Passage-level results", box=None)
    for column in [
        "Concept",
        "Model",
        "P",
        "R",
        "F1",
        "F1 95% CI",
        "Span F1 (0.5)",
        "Failed",
    ]:
        table.add_column(
            column,
            justify="left" if column in ("Concept", "Model") else "right",
            no_wrap=True,
        )

    for row in sorted(passage_rows, key=lambda r: (r["wikibase_id"], -r["f1"])):
        model = row["model_name"].removeprefix("openrouter:")
        model += " (baseline)" if row["is_baseline"] else ""
        span = span_f1.get((row["wikibase_id"], row["model_name"]))
        table.add_row(
            row["wikibase_id"],
            model,
            f"{row['precision']:.3f}",
            f"{row['recall']:.3f}",
            f"{row['f1']:.3f}",
            f"[{row['f1_ci_low']:.3f}, {row['f1_ci_high']:.3f}]",
            f"{span:.3f}" if span is not None else "-",
            str(row["n_failed_predictions"]),
        )
    console.print(table)
    console.print()

    models = sorted({row["model_name"] for row in passage_rows})
    concepts = {row["wikibase_id"] for row in passage_rows}
    macro = Table(title="Macro-average passage-level F1", box=None)
    for column in ["Model", "Macro F1", "Concepts"]:
        macro.add_column(column, justify="left" if column == "Model" else "right")
    for model in models:
        f1s = [row["f1"] for row in passage_rows if row["model_name"] == model]
        macro.add_row(
            model, f"{sum(f1s) / len(f1s):.3f}", f"{len(f1s)}/{len(concepts)}"
        )
    console.print(macro)

    if skipped:
        console.print(f"\n[bold yellow]Skipped {len(skipped)} runs[/bold yellow]")
        for key, reason in skipped:
            console.print(f"  {key}: {reason}")


async def run_sweep(
    configs: list[Path], models: list[str], output_path: Path
) -> tuple[list[dict[str, Any]], list[tuple[str, str]]]:
    """Evaluate every model for every config, writing results out as they land."""
    writer = ResultWriter(output_path)
    done = writer.done()
    skipped: list[tuple[str, str]] = []
    session = WikibaseSession()

    for index, config_path in enumerate(configs, start=1):
        cfg = CustomClassifierConfig.from_yaml(config_path)
        wikibase_id = str(cfg.wikibase_id)
        console.rule(f"[{index}/{len(configs)}] {wikibase_id} ({config_path.name})")

        if cfg.llm is None:
            skipped.append((wikibase_id, "config has no 'llm' section"))
            continue

        definitions = {
            wid: ((await session.get_concept_async(wid)).definition or "")
            for wid in set(cfg.llm.related_definitions)
        }
        base_kwargs = cfg.llm.to_classifier_kwargs(definitions=definitions)

        concept = await get_concept_async(cfg.wikibase_id)
        # Applied the same way run_training does, so the prompt matches training
        for key, value in cfg.concept_overrides.as_overrides().items():
            setattr(concept, key, value)
        if not concept.labelled_passages:
            skipped.append((wikibase_id, "no labelled passages"))
            continue

        # The config's own model goes first, as the baseline
        baseline = cfg.llm.model_name
        for model_name in [baseline] + [m for m in models if m != baseline]:
            if (wikibase_id, model_name) in done:
                console.log(f"Already done: {model_name}")
                continue

            console.log(f"Evaluating {model_name}")
            try:
                rows = await evaluate_model(
                    concept, base_kwargs, model_name, model_name == baseline
                )
            except Exception as error:
                console.log(f"[red]{wikibase_id} / {model_name} failed: {error}[/red]")
                skipped.append((f"{wikibase_id} / {model_name}", str(error)))
                continue

            n_failed = rows[0]["n_failed_predictions"] if rows else 0
            if n_failed > MAX_FAILED_SHARE * len(concept.labelled_passages):
                console.log(
                    f"[red]{wikibase_id} / {model_name}: {n_failed} failed "
                    "predictions, not recording it so a rerun retries it[/red]"
                )
                skipped.append(
                    (f"{wikibase_id} / {model_name}", f"{n_failed} failed predictions")
                )
                continue

            writer.write(rows)
            for row in rows:
                if row["agreement_level"] == "Passage level":
                    console.log(
                        f"{model_name}: P={row['precision']:.3f} "
                        f"R={row['recall']:.3f} F1={row['f1']:.3f} "
                        f"[{row['f1_ci_low']:.3f}, {row['f1_ci_high']:.3f}] "
                        f"failed={row['n_failed_predictions']}"
                    )

    # Summarise everything in the CSV, including pairs finished by earlier runs
    all_rows = pd.read_csv(output_path).to_dict("records") if writer.done() else []
    return cast(list[dict[str, Any]], all_rows), skipped


def split_list(value: str) -> list[str]:
    """Split a comma-separated CLI value."""
    return [item.strip() for item in value.split(",") if item.strip()]


@app.command()
def main(
    configs: Annotated[
        str | None,
        typer.Option(help="Comma-separated custom-classifier YAML config paths"),
    ] = None,
    models: Annotated[
        str | None,
        typer.Option(
            help="Comma-separated pydantic-ai model names to compare against each "
            "config's own model"
        ),
    ] = None,
    output_dir: Annotated[
        Path, typer.Option(help="Directory to write results.csv into")
    ] = DEFAULT_OUTPUT_DIR,
):
    """Benchmark several LLMs on custom-classifier YAML configs."""
    config_paths = (
        [Path(p) for p in split_list(configs)] if configs else DEFAULT_CONFIGS
    )
    model_names = split_list(models) if models else DEFAULT_MODELS

    output_path = output_dir / "results.csv"
    console.log(
        f"Benchmarking {len(model_names)} models (+ each config's baseline) across "
        f"{len(config_paths)} configs into {output_path}"
    )

    rows, skipped = asyncio.run(run_sweep(config_paths, model_names, output_path))

    console.print()
    print_summary(rows, skipped)


if __name__ == "__main__":
    app()
