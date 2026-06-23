"""
CLI wrapper for the evaluate operation.

The reusable logic lives in `knowledge_graph.operations.evaluate` (imported directly by
`knowledge_graph.classifier.autollm`). This module only adds the Typer command used by
`just evaluate`, which loads the classifier and concept, sets up W&B tracking, runs the
evaluation and saves the metrics.
"""

import asyncio
import os
from typing import Annotated, Optional

import typer
import wandb
from rich.console import Console

from knowledge_graph.cloud import AwsEnv, Namespace, parse_aws_env
from knowledge_graph.config import classifier_dir, metrics_dir
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.operations.evaluate import (
    evaluate_classifier,
    load_classifier_local,
    save_metrics,
)
from knowledge_graph.operations.get_concept import get_concept_async
from knowledge_graph.wandb_helpers import load_classifier_from_wandb

console = Console()
app = typer.Typer()


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
    wandb_model_path: Annotated[
        Optional[str],
        typer.Option(
            help="W&B model artifact path (e.g. 'climatepolicyradar/Q1829/abcdefg:v0'). "
            "If not provided, loads the most recent local classifier.",
        ),
    ] = None,
    aws_env: Annotated[
        AwsEnv,
        typer.Option(
            help="AWS environment to evaluate the model artifact within",
            parser=parse_aws_env,
        ),
    ] = AwsEnv.labs,
    track: Annotated[
        bool,
        typer.Option(
            help="Whether to track the evaluation run in Weights & Biases",
        ),
    ] = True,
):
    """Evaluate a classifier against its validation dataset."""
    console.log("🚀 Starting model evaluation")

    if os.environ.get("USE_AWS_PROFILES", "true").lower() == "true":
        os.environ["AWS_PROFILE"] = aws_env.value
        console.log(f"🔑 Set AWS_PROFILE={aws_env.value}")

    if wandb_model_path:
        loaded_classifier = load_classifier_from_wandb(wandb_model_path)
    else:
        console.log("Loading local classifier...")
        loaded_classifier = load_classifier_local(wikibase_id)
        console.log(f"🤖 Loaded classifier {loaded_classifier} from {classifier_dir}")

    console.log(f'📚 Loading concept "{wikibase_id}" from Wikibase and Argilla')
    concept = asyncio.run(
        get_concept_async(
            wikibase_id=wikibase_id,
            include_labels_from_subconcepts=True,
            include_recursive_has_subconcept=True,
        )
    )

    console.log(
        f"📊 Found {len(concept.labelled_passages)} labelled passages for evaluation"
    )

    if not concept.labelled_passages:
        console.log(
            "[yellow]⚠️  Warning: No labelled passages found. Cannot evaluate.[/yellow]"
        )
        raise typer.Exit(code=1)

    wandb_run = None
    if track:
        entity = "climatepolicyradar"
        project = wikibase_id
        namespace = Namespace(project=project, entity=entity)

        wandb_run = wandb.init(
            entity=namespace.entity,
            project=namespace.project,
            job_type="evaluate_model",
            config={
                "classifier_type": loaded_classifier.name,
                "concept_id": concept.id,
            },
        )
        if wandb_model_path:
            wandb_run.config["model_artifact_path"] = wandb_model_path  # type: ignore
        console.log(f"📊 Tracking evaluation in W&B: {wandb_run.url}")

        if wandb_model_path:
            wandb_run.use_artifact(wandb_model_path)

    try:
        df, _, _ = evaluate_classifier(
            classifier=loaded_classifier,
            labelled_passages=concept.labelled_passages,
            wandb_run=wandb_run,
        )

        # Save metrics to local file
        metrics_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = save_metrics(df, wikibase_id)
        console.log(f"📄 Saved performance metrics to {metrics_path}")

        if track and wandb_run:
            console.log(f"📊 View results at: {wandb_run.url}")
    finally:
        if wandb_run:
            wandb_run.finish()


if __name__ == "__main__":
    app()
