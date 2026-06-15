"""
Benchmark HuggingFace transformer backbones for the ``BertBasedClassifier``.

Trains a ``BertBasedClassifier`` for every (model x concept) combination on Coiled GPU
(via the existing ``train-on-gpu`` Prefect deployment), then pulls the evaluation
metrics that training logs to W&B into a CSV + console table.

By default, any (model x concept) configuration that already has a finished training run
in W&B is *skipped* and its metrics are pulled instead of re-training. Pass ``--force`` to
re-train everything.

Run with::

    uv run python scripts/benchmarks/modernbert_models.py
    uv run python scripts/benchmarks/modernbert_models.py --force
    uv run python scripts/benchmarks/modernbert_models.py --dry-run

Shared classifier config (e.g. ``unfreeze_layers``) and ``limit_training_samples`` are
defined once below and applied to every concept/model combination.
"""

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Optional

import typer
import wandb
from prefect.client.schemas.objects import FlowRun
from prefect.deployments import run_deployment
from rich.console import Console
from rich.table import Table

from flows.utils import get_flow_run_ui_url
from knowledge_graph.cloud import AwsEnv, generate_deployment_name
from knowledge_graph.config import WANDB_ENTITY, processed_data_dir
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.utils import get_logger

console = Console()
app = typer.Typer()
logger = get_logger()

# --- Benchmark configuration -------------------------------------------------------

# HuggingFace model backbones to compare.
MODELS: list[str] = [
    "kdutia/cpr-ModernBERT",
    "kdutia/cpr-ModernBERT-b",
    "kdutia/cpr-ModernBERT-c",
    "answerdotai/ModernBERT-base",
    "climatebert/distilroberta-base-climate-f",
]


@dataclass
class ConceptConfig:
    """
    Training config for one concept, shared across all model variants.

    ``classifier_kwargs`` are passed to BertBasedClassifier (``model_name`` is added
    per-model). ``limit_training_samples`` caps the training set (None = all samples).
    ``training_data_wandb_path`` is the W&B artifact path to train from (shared across
    models so every backbone trains on the same data).
    """

    classifier_kwargs: dict[str, Any] = field(default_factory=dict)
    limit_training_samples: Optional[int] = None
    training_data_wandb_path: Optional[str] = None


# Per-concept config, shared across every model in MODELS. Edit e.g. unfreeze_layers or
# limit_training_samples per concept here. Concepts are the keys of this mapping.
# training_data_wandb_path values were taken from the succeeded cpr-ModernBERT runs.
CONCEPT_CONFIGS: dict[WikibaseID, ConceptConfig] = {
    WikibaseID("Q32"): ConceptConfig(
        classifier_kwargs={"unfreeze_layers": 2, "limit_training_samples": 5600},
        training_data_wandb_path="climatepolicyradar/Q32/labelled-passages:v10",
    ),
    WikibaseID("Q911"): ConceptConfig(
        classifier_kwargs={"unfreeze_layers": 1, "limit_training_samples": 1850},
        training_data_wandb_path="climatepolicyradar/Q911/updated_bert_training_data:v5",
    ),
    WikibaseID("Q912"): ConceptConfig(
        classifier_kwargs={"unfreeze_layers": 1, "limit_training_samples": 4006},
        training_data_wandb_path="climatepolicyradar/Q912/labelled-passages:v20",
    ),
    WikibaseID("Q1829"): ConceptConfig(
        classifier_kwargs={"unfreeze_layers": 0, "limit_training_samples": 5500},
        training_data_wandb_path="climatepolicyradar/Q1829/labelled-passages:v6",
    ),
}


def config_for(
    wikibase_id: WikibaseID, model_name: str
) -> tuple[dict[str, Any], Any, Optional[int], Optional[str]]:
    """Resolve (classifier_kwargs, unfreeze_layers, limit, training_data_path) for a config."""
    concept_config = CONCEPT_CONFIGS[wikibase_id]
    classifier_kwargs = {**concept_config.classifier_kwargs}
    # limit_training_samples is a run_training param, not a BertBasedClassifier kwarg, so
    # keep it out of classifier_kwargs. Accept it either on the ConceptConfig field or,
    # for convenience, nested inside classifier_kwargs.
    limit = classifier_kwargs.pop("limit_training_samples", None)
    if limit is None:
        limit = concept_config.limit_training_samples
    classifier_kwargs["model_name"] = model_name
    unfreeze_layers = classifier_kwargs.get("unfreeze_layers")
    return (
        classifier_kwargs,
        unfreeze_layers,
        limit,
        concept_config.training_data_wandb_path,
    )


# All configs use the BERT classifier explicitly: most of these concepts do not map to
# BertBasedClassifier in ClassifierFactory by default.
CLASSIFIER_TYPE = "BertBasedClassifier"

# Train-model run summary keys (logged by scripts/evaluate.py) that we collect.
METRIC_KEYS: list[str] = [
    "passage_level_f1",
    "passage_level_precision",
    "passage_level_recall",
    "passage_level_accuracy",
    "passage_level_support",
    "optimal_f1_threshold",
]

CSV_COLUMNS: list[str] = [
    "wikibase_id",
    "model_name",
    "unfreeze_layers",
    "limit_training_samples",
    "status",
    *METRIC_KEYS,
    "wandb_run_url",
]


# --- W&B helpers -------------------------------------------------------------------


def _matches_config(
    run: Any,
    model_name: str,
    unfreeze_layers: Any,
    limit_training_samples: Optional[int],
    training_data_wandb_path: Optional[str],
) -> bool:
    """Return whether a W&B run's config matches a benchmark config."""
    config = run.config
    if config.get("classifier_type") != CLASSIFIER_TYPE:
        return False
    classifier_kwargs = config.get("classifier_kwargs") or {}
    # HuggingFace repo ids are case-insensitive (e.g. the trained run stored
    # "kdutia/cpr-ModernBERT" while MODELS may use "kdutia/cpr-modernBERT").
    stored_model = classifier_kwargs.get("model_name") or ""
    if stored_model.lower() != model_name.lower():
        return False
    # Treat unfreeze_layers None and 0 as equivalent: runs trained before
    # unfreeze_layers was introduced have a null value, which is functionally
    # the same as head-only training (0).
    if (classifier_kwargs.get("unfreeze_layers") or 0) != (unfreeze_layers or 0):
        return False
    if config.get("limit_training_samples") != limit_training_samples:
        return False
    if config.get("training_data_wandb_path") != training_data_wandb_path:
        return False
    return True


def get_matching_runs(
    wikibase_id: WikibaseID,
    model_name: str,
    unfreeze_layers: Any,
    limit_training_samples: Optional[int],
    training_data_wandb_path: Optional[str],
) -> list[Any]:
    """
    Return all train_model W&B runs matching a benchmark config.

    Nested-dict W&B server-side filters are unreliable, so we filter the project's
    train_model runs in Python (same approach as embeddingclassifier.py).
    """
    api = wandb.Api()
    runs = api.runs(
        f"{WANDB_ENTITY}/{wikibase_id}",
        filters={"jobType": "train_model"},
        order="-created_at",
    )
    return [
        run
        for run in runs
        if _matches_config(
            run,
            model_name,
            unfreeze_layers,
            limit_training_samples,
            training_data_wandb_path,
        )
    ]


def extract_metrics(run: Any) -> dict[str, Any]:
    """Pull the benchmark metric keys and run URL from a finished W&B run."""
    metrics: dict[str, Any] = {key: run.summary.get(key) for key in METRIC_KEYS}
    metrics["wandb_run_url"] = run.url
    return metrics


# --- Dispatch ----------------------------------------------------------------------


def dispatch_training(
    wikibase_id: WikibaseID,
    classifier_kwargs: dict[str, Any],
    limit_training_samples: Optional[int],
    training_data_wandb_path: Optional[str],
    aws_env: AwsEnv,
) -> FlowRun:
    """Start a train-on-gpu Coiled flow run for a single config (fire-and-forget)."""
    flow_name = "train-on-gpu"
    deployment_name = generate_deployment_name(flow_name=flow_name, aws_env=aws_env)
    flow_run: FlowRun = run_deployment(  # type: ignore[assignment]
        name=f"{flow_name}/{deployment_name}",
        parameters={
            "wikibase_id": wikibase_id,
            "track_and_upload": True,
            "aws_env": aws_env,
            "evaluate": True,
            "classifier_type": CLASSIFIER_TYPE,
            "classifier_kwargs": classifier_kwargs,
            "concept_overrides": None,
            "training_data_wandb_path": training_data_wandb_path,
            "limit_training_samples": limit_training_samples,
        },
        timeout=0,  # don't wait for the flow to finish
    )
    return flow_run


# --- CSV output --------------------------------------------------------------------


def _make_row(
    wikibase_id: WikibaseID,
    model_name: str,
    unfreeze_layers: Any,
    limit_training_samples: Optional[int],
    status: str,
    metrics: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "wikibase_id": wikibase_id,
        "model_name": model_name,
        "unfreeze_layers": unfreeze_layers,
        "limit_training_samples": limit_training_samples,
        "status": status,
    }
    for key in METRIC_KEYS:
        row[key] = (metrics or {}).get(key)
    row["wandb_run_url"] = (metrics or {}).get("wandb_run_url")
    return row


class ResultsWriter:
    """Append benchmark result rows to a CSV incrementally so nothing is lost."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "w", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=CSV_COLUMNS)
        self._writer.writeheader()
        self._fh.flush()
        self.rows: list[dict[str, Any]] = []

    def write(self, row: dict[str, Any]) -> None:
        self._writer.writerow(row)
        self._fh.flush()
        self.rows.append(row)

    def close(self) -> None:
        self._fh.close()


def print_summary(rows: list[dict[str, Any]]) -> None:
    """Print a console table of results, sorted by concept then F1 (desc)."""
    table = Table(title="ModernBERT model benchmark", show_lines=False)
    table.add_column("Concept")
    table.add_column("Model")
    table.add_column("Status")
    table.add_column("F1", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("Run", overflow="fold")

    def _sort_key(row: dict[str, Any]) -> tuple[str, float]:
        f1 = row.get("passage_level_f1")
        return (str(row["wikibase_id"]), -(f1 if isinstance(f1, (int, float)) else -1))

    def _fmt(value: Any) -> str:
        return f"{value:.3f}" if isinstance(value, (int, float)) else "-"

    for row in sorted(rows, key=_sort_key):
        table.add_row(
            str(row["wikibase_id"]),
            row["model_name"],
            row["status"],
            _fmt(row.get("passage_level_f1")),
            _fmt(row.get("passage_level_precision")),
            _fmt(row.get("passage_level_recall")),
            row.get("wandb_run_url") or "-",
        )

    console.print(table)


# --- Main --------------------------------------------------------------------------


@app.command()
def main(
    force: Annotated[
        bool,
        typer.Option(
            help="Re-train every config even if a finished W&B run already exists.",
        ),
    ] = False,
    aws_env: Annotated[
        AwsEnv,
        typer.Option(help="AWS environment for the train-on-gpu deployment."),
    ] = AwsEnv.production,
    output: Annotated[
        Optional[Path],
        typer.Option(help="Path to write the results CSV."),
    ] = None,
    poll_interval: Annotated[
        int,
        typer.Option(help="Seconds between W&B polls while waiting for runs."),
    ] = 300,
    timeout: Annotated[
        int,
        typer.Option(help="Per-config timeout (seconds) when waiting for a run."),
    ] = 5400,
    dry_run: Annotated[
        bool,
        typer.Option(help="Look up existing runs but do not dispatch any training."),
    ] = False,
) -> None:
    """Benchmark BERT backbones across concepts, pulling metrics from W&B."""
    output_path = output or (
        processed_data_dir / "benchmarks" / "modernbert_models.csv"
    )

    writer = ResultsWriter(output_path)
    console.log(f"Writing results to {output_path}")

    # (wikibase_id, model, set of pre-existing run ids)
    pending: list[tuple[WikibaseID, str, set[str]]] = []

    try:
        # Phase 1: resolve cached configs, dispatch the rest.
        for wikibase_id in CONCEPT_CONFIGS:
            for model_name in MODELS:
                classifier_kwargs, unfreeze_layers, limit, train_data = config_for(
                    wikibase_id, model_name
                )

                existing = get_matching_runs(
                    wikibase_id, model_name, unfreeze_layers, limit, train_data
                )
                finished = [r for r in existing if r.state == "finished"]

                if finished and not force:
                    metrics = extract_metrics(finished[0])
                    writer.write(
                        _make_row(
                            wikibase_id,
                            model_name,
                            unfreeze_layers,
                            limit,
                            "cached",
                            metrics,
                        )
                    )
                    console.log(
                        f"✅ cached: {wikibase_id} / {model_name} "
                        f"(F1={metrics.get('passage_level_f1')})"
                    )
                    continue

                if dry_run:
                    console.log(f"🔸 would dispatch: {wikibase_id} / {model_name}")
                    writer.write(
                        _make_row(
                            wikibase_id,
                            model_name,
                            unfreeze_layers,
                            limit,
                            "would-train",
                        )
                    )
                    continue

                pre_existing_ids = {r.id for r in existing}
                flow_run = dispatch_training(
                    wikibase_id, classifier_kwargs, limit, train_data, aws_env
                )
                console.log(
                    f"🚀 dispatched: {wikibase_id} / {model_name} — "
                    f"{get_flow_run_ui_url(flow_run)}"
                )
                pending.append((wikibase_id, model_name, pre_existing_ids))

        # Phase 2: poll W&B for each dispatched config to finish.
        for wikibase_id, model_name, pre_existing_ids in pending:
            _, unfreeze_layers, limit, train_data = config_for(wikibase_id, model_name)
            console.log(f"⏳ waiting for {wikibase_id} / {model_name} ...")
            deadline = time.monotonic() + timeout
            status = "timeout"
            metrics: Optional[dict[str, Any]] = None

            while time.monotonic() < deadline:
                runs = get_matching_runs(
                    wikibase_id, model_name, unfreeze_layers, limit, train_data
                )
                # Only consider runs created by this benchmark invocation.
                new_runs = [r for r in runs if r.id not in pre_existing_ids]
                done = next(
                    (
                        r
                        for r in new_runs
                        if r.state in ("finished", "failed", "crashed")
                    ),
                    None,
                )
                if done is not None:
                    if done.state == "finished":
                        status = "trained"
                        metrics = extract_metrics(done)
                    else:
                        status = f"failed ({done.state})"
                        metrics = {"wandb_run_url": done.url}
                    break
                time.sleep(poll_interval)

            writer.write(
                _make_row(
                    wikibase_id, model_name, unfreeze_layers, limit, status, metrics
                )
            )
            console.log(
                f"{'✅' if status == 'trained' else '⚠️'} {status}: "
                f"{wikibase_id} / {model_name}"
            )
    finally:
        writer.close()

    print_summary(writer.rows)
    console.log(f"Done. Results written to {output_path}")


if __name__ == "__main__":
    app()
