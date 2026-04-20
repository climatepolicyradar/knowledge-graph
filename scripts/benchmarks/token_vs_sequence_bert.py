"""
Benchmark BERT sequence-based classifiers against:

- a token-based classifier, with no layers unfrozen
- a token-based classifier with two layers unfrozen
"""

import asyncio
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from knowledge_graph.classifier import BertBasedClassifier, BertTokenClassifier
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.wandb_helpers import load_labelled_passages_from_wandb
from scripts.evaluate import evaluate_classifier
from scripts.get_concept import get_concept_async
from scripts.train import deduplicate_training_data

console = Console()
app = typer.Typer()


async def fetch_concept(wikibase_id: WikibaseID) -> Concept:
    concept = await get_concept_async(
        wikibase_id=wikibase_id,
        include_recursive_has_subconcept=True,
        include_labels_from_subconcepts=True,
    )
    if not concept.labelled_passages:
        console.log(f"{wikibase_id} has no labelled passages for evaluation")
        raise typer.Exit(1)
    return concept


def run_benchmark_for_concept(
    concept: Concept,
    training_passages: list[LabelledPassage],
    enable_wandb: bool = False,
) -> pd.DataFrame | None:
    """Train and evaluate both classifiers on a single concept."""
    console.rule(f"{concept.wikibase_id} ({concept.preferred_label})")

    # Evaluation data comes from the concept's labelled passages (Argilla)
    eval_passages = concept.labelled_passages

    # Deduplicate training data against evaluation set
    train_passages = deduplicate_training_data(
        training_data=training_passages,
        evaluation_data=eval_passages,
    )

    console.log(f"Train: {len(train_passages)}, Eval: {len(eval_passages)}")

    if len(train_passages) < 10:
        console.log(
            "Not enough training passages after deduplication (need >= 10). Skipping."
        )
        return None

    results = []

    classifiers_to_run = [
        (
            "BertBasedClassifier (sequence)",
            lambda: BertBasedClassifier(concept=concept),
        ),
        ("BertTokenClassifier (token)", lambda: BertTokenClassifier(concept=concept)),
        (
            "BertTokenClassifier (token, 2 layers unfrozen)",
            lambda: BertTokenClassifier(concept=concept, unfreeze_layers=2),
        ),
    ]

    for name, make_classifier in classifiers_to_run:
        console.log(f"\nTraining {name}...")
        classifier = make_classifier()
        classifier.fit(train_passages, enable_wandb=enable_wandb)

        console.log(f"Evaluating {name}...")
        metrics_df, _, _ = evaluate_classifier(
            classifier=classifier,
            labelled_passages=eval_passages,
        )

        metrics_df["classifier"] = name
        metrics_df["concept_id"] = str(concept.wikibase_id)
        results.append(metrics_df)

    if results:
        return pd.concat(results, ignore_index=True)
    return None


def print_comparison_table(df: pd.DataFrame) -> None:
    """Print a side-by-side comparison of classifier metrics."""
    table = Table(title="Classifier Comparison", show_lines=True)
    table.add_column("Metric")
    table.add_column("Jaccard threshold")

    classifiers = df["classifier"].unique()
    for c in classifiers:
        table.add_column(c, justify="right")

    # Group by metric and agreement level
    metric_cols = [
        c for c in df.columns if c not in ("classifier", "concept_id", "group")
    ]

    for col in metric_cols:
        if col in ("agreement",):
            continue
        for agreement in (
            df["agreement"].unique() if "agreement" in df.columns else [""]
        ):
            row = [col, str(agreement)]
            for c in classifiers:
                mask = df["classifier"] == c
                if "agreement" in df.columns and agreement:
                    mask = mask & (df["agreement"] == agreement)
                vals = df.loc[mask, col]
                if not vals.empty and pd.api.types.is_numeric_dtype(vals):
                    row.append(f"{vals.mean():.4f}")
                else:
                    row.append("-")
            if any(v not in ("-", col, str(agreement)) for v in row[2:]):
                table.add_row(*row)

    console.print(table)


@app.command()
def main(
    concept: Annotated[
        str,
        typer.Option(
            "--concept",
            help="Wikibase ID to evaluate (e.g., 'Q1016')",
        ),
    ],
    training_data_wandb_path: Annotated[
        str,
        typer.Option(
            "--training-data-wandb-path",
            help="W&B artifact path for training data (e.g., 'climatepolicyradar/Q1016/training-data:latest')",
        ),
    ],
    wandb: Annotated[
        bool,
        typer.Option("--wandb", help="Enable W&B logging during training"),
    ] = False,
):
    """
    Benchmark token-level vs sequence-level BERT classifiers.

    Trains both classifiers on the same training data (from W&B) for a
    concept, evaluates against the concept's labelled passages from Argilla,
    and compares span-level and passage-level metrics.
    """
    console.log("Token vs Sequence BERT Classifier Benchmark")

    concept_id = WikibaseID(concept)

    # Load training data from W&B
    console.log(f"Loading training data from W&B: {training_data_wandb_path}")
    training_passages = load_labelled_passages_from_wandb(
        wandb_path=training_data_wandb_path
    )
    console.log(f"Loaded {len(training_passages)} training passages from W&B")

    # Fetch concept (evaluation data comes from Argilla via concept)
    fetched_concept = asyncio.run(fetch_concept(concept_id))

    # Run benchmark
    result = run_benchmark_for_concept(
        fetched_concept,
        training_passages=training_passages,
        enable_wandb=wandb,
    )

    if result is not None:
        print_comparison_table(result)

        # Save raw results
        output_path = Path(__file__).parent / "token_vs_sequence_results.csv"
        result.to_csv(output_path, index=False)
        console.log(f"Results saved to {output_path}")
    else:
        console.log("No results to compare.")


if __name__ == "__main__":
    app()
