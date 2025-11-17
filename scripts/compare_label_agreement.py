"""
Compare label agreement between ensemble and best model predictions.

This script loads ensemble predictions from W&B and runs the same
passages through the best model to measure label agreement.
"""

import asyncio
import random
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

import scripts.get_concept
from knowledge_graph.classifier.large_language_model import LLMClassifier
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.wandb_helpers import load_labelled_passages_from_wandb

console = Console()
app = typer.Typer()


def calculate_agreement_metrics(
    ensemble_labels: list[bool], best_model_labels: list[bool]
) -> dict:
    """
    Calculate agreement metrics between two sets of labels.

    :param ensemble_labels: Labels from ensemble model
    :param best_model_labels: Labels from best model
    :return: Dictionary with agreement metrics
    """
    assert len(ensemble_labels) == len(best_model_labels), "Label lists must be same length"

    n = len(ensemble_labels)
    agreements = sum(a == b for a, b in zip(ensemble_labels, best_model_labels))
    disagreements = n - agreements

    # Calculate Cohen's Kappa
    # P(agreement observed)
    p_o = agreements / n

    # P(agreement expected by chance)
    ensemble_pos = sum(ensemble_labels) / n
    best_model_pos = sum(best_model_labels) / n
    p_e = (ensemble_pos * best_model_pos) + ((1 - ensemble_pos) * (1 - best_model_pos))

    # Cohen's Kappa
    kappa = (p_o - p_e) / (1 - p_e) if p_e != 1 else 1.0

    # Confusion matrix
    true_pos = sum(a and b for a, b in zip(ensemble_labels, best_model_labels))
    false_pos = sum(a and not b for a, b in zip(ensemble_labels, best_model_labels))
    false_neg = sum(not a and b for a, b in zip(ensemble_labels, best_model_labels))
    true_neg = sum(not a and not b for a, b in zip(ensemble_labels, best_model_labels))

    return {
        "total_samples": n,
        "agreements": agreements,
        "disagreements": disagreements,
        "agreement_rate": agreements / n,
        "cohens_kappa": kappa,
        "confusion_matrix": {
            "true_positive": true_pos,
            "false_positive": false_pos,
            "false_negative": false_neg,
            "true_negative": true_neg,
        },
        "ensemble_positive_rate": ensemble_pos,
        "best_model_positive_rate": best_model_pos,
    }


@app.command()
def main(
    wikibase_id: Annotated[
        WikibaseID,
        typer.Option(
            ...,
            help="The Wikibase ID of the concept",
            parser=WikibaseID,
        ),
    ],
    ensemble_predictions_path: Annotated[
        str,
        typer.Option(
            ...,
            help="W&B artifact path for ensemble predictions (e.g., 'climatepolicyradar/Q1829/t8hn3c3t-labelled-passages:v3')",
        ),
    ],
    best_model_type: Annotated[
        str,
        typer.Option(
            ...,
            help="Model type to use as best model (e.g., 'gpt-4.5')",
        ),
    ],
    sample_size: Annotated[
        int,
        typer.Option(
            help="Number of passages to compare (default: all)",
        ),
    ] = None,
):
    """
    Compare label agreement between ensemble and best model.

    This loads ensemble predictions from W&B and runs the same passages
    through the best model to measure agreement.
    """
    asyncio.run(
        async_main(
            wikibase_id=wikibase_id,
            ensemble_predictions_path=ensemble_predictions_path,
            best_model_type=best_model_type,
            sample_size=sample_size,
        )
    )


async def async_main(
    wikibase_id: WikibaseID,
    ensemble_predictions_path: str,
    best_model_type: str,
    sample_size: int | None,
):
    """Async main function to handle concept loading."""
    console.log(f"[bold]Comparing ensemble vs {best_model_type} for {wikibase_id}")

    # Load ensemble predictions from W&B
    console.log(f"Loading ensemble predictions from {ensemble_predictions_path}...")
    ensemble_passages = load_labelled_passages_from_wandb(wandb_path=ensemble_predictions_path)
    console.log(f"Loaded {len(ensemble_passages)} passages from ensemble")

    # Sample if requested
    if sample_size and sample_size < len(ensemble_passages):
        random.seed(42)
        ensemble_passages = random.sample(ensemble_passages, sample_size)
        console.log(f"Sampled {sample_size} passages for comparison")

    # Load concept using async method like train.py does
    console.log(f"Loading concept {wikibase_id}...")
    concept = await scripts.get_concept.get_concept_async(
        wikibase_id=wikibase_id,
        include_labels_from_subconcepts=True,
        include_recursive_has_subconcept=True,
    )
    console.log(f"Loaded concept: {concept.preferred_label}")

    # Initialize best model classifier
    console.log(f"Initializing {best_model_type} classifier...")
    best_model_classifier = LLMClassifier(
        concept=concept,
        model_name=best_model_type,
    )

    # Extract ensemble labels and texts
    console.log("Extracting ensemble labels...")
    ensemble_labels = [
        len(passage.spans) > 0 and passage.spans[0].prediction_probability > 0.5
        for passage in ensemble_passages
    ]
    passage_texts = [passage.text for passage in ensemble_passages]

    # Run best model predictions (batching is handled internally by LLMClassifier)
    console.log(
        f"Running {best_model_type} predictions on {len(ensemble_passages)} passages..."
    )
    best_model_predictions = best_model_classifier.predict(passage_texts, batch_size=50, show_progress=True)

    # Convert predictions to boolean labels
    best_model_labels = [len(spans) > 0 for spans in best_model_predictions]

    # Calculate agreement metrics
    console.log("\nCalculating agreement metrics...")
    metrics = calculate_agreement_metrics(ensemble_labels, best_model_labels)

    # Display results
    console.print("\n[bold cyan]═══ Label Agreement Analysis ═══[/bold cyan]\n")

    # Summary table
    summary_table = Table(title="Summary Statistics", show_header=True)
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green", justify="right")

    summary_table.add_row("Total Samples", str(metrics["total_samples"]))
    summary_table.add_row("Agreements", str(metrics["agreements"]))
    summary_table.add_row("Disagreements", str(metrics["disagreements"]))
    summary_table.add_row(
        "Agreement Rate", f"{metrics['agreement_rate']:.1%}"
    )
    summary_table.add_row("Cohen's Kappa", f"{metrics['cohens_kappa']:.3f}")

    console.print(summary_table)
    console.print()

    # Confusion matrix
    cm = metrics["confusion_matrix"]
    cm_table = Table(title="Confusion Matrix", show_header=True)
    cm_table.add_column("", style="cyan")
    cm_table.add_column("Best Model: Positive", justify="center")
    cm_table.add_column("Best Model: Negative", justify="center")

    cm_table.add_row(
        "Ensemble: Positive",
        f"[green]{cm['true_positive']}[/green]",
        f"[red]{cm['false_positive']}[/red]",
    )
    cm_table.add_row(
        "Ensemble: Negative",
        f"[red]{cm['false_negative']}[/red]",
        f"[green]{cm['true_negative']}[/green]",
    )

    console.print(cm_table)
    console.print()

    # Label rates
    rates_table = Table(title="Label Distribution", show_header=True)
    rates_table.add_column("Model", style="cyan")
    rates_table.add_column("Positive Rate", justify="right")
    rates_table.add_column("Negative Rate", justify="right")

    rates_table.add_row(
        "Ensemble",
        f"{metrics['ensemble_positive_rate']:.1%}",
        f"{1 - metrics['ensemble_positive_rate']:.1%}",
    )
    rates_table.add_row(
        "Best Model",
        f"{metrics['best_model_positive_rate']:.1%}",
        f"{1 - metrics['best_model_positive_rate']:.1%}",
    )

    console.print(rates_table)

    # Interpretation
    console.print("\n[bold cyan]═══ Interpretation ═══[/bold cyan]\n")

    kappa = metrics["cohens_kappa"]
    if kappa > 0.8:
        interpretation = "[green]Almost perfect agreement[/green]"
    elif kappa > 0.6:
        interpretation = "[green]Substantial agreement[/green]"
    elif kappa > 0.4:
        interpretation = "[yellow]Moderate agreement[/yellow]"
    elif kappa > 0.2:
        interpretation = "[orange1]Fair agreement[/orange1]"
    else:
        interpretation = "[red]Poor agreement[/red]"

    console.print(f"Cohen's Kappa = {kappa:.3f}: {interpretation}")
    console.print(
        f"\nAgreement rate: {metrics['agreement_rate']:.1%} of labels match between models"
    )

    if metrics["disagreements"] > 0:
        console.print(
            f"\n[yellow]⚠️  {metrics['disagreements']} disagreements found "
            f"({metrics['disagreements'] / metrics['total_samples']:.1%} of samples)[/yellow]"
        )
        console.print(
            "\nConsider manually reviewing disagreements to understand label quality issues."
        )


if __name__ == "__main__":
    app()
