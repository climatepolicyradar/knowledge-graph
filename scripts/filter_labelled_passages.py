"""
Filters labelled passages by excluding corpus types.

A labelled passages JSONL file is uploaded to a new W&B run.
"""

from contextlib import nullcontext
from typing import Annotated, Optional

import click
import typer
import wandb
from rich.console import Console

from knowledge_graph.config import WANDB_ENTITY
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.wandb_helpers import (
    load_labelled_passages_from_wandb,
    log_labelled_passages_artifact_to_wandb_run,
)
from knowledge_graph.wikibase import WikibaseSession

app = typer.Typer()
console = Console()

CORPUS_TYPES = [
    "Litigation",
    "Laws and Policies",
    "Intl. agreements",
    "Reports",
    "AF",
    "GEF",
    "CIF",
    "GCF",
]

CORPUS_TYPE_KEY = "document_metadata.corpus_type_name"


@app.command()
def main(
    wikibase_id: Annotated[
        WikibaseID,
        typer.Option(
            ...,
            help="The Wikibase ID of the concept associated with the labelled passages",
            parser=WikibaseID,
        ),
    ],
    labelled_passages_wandb_path: Annotated[
        str,
        typer.Option(
            help="W&B artifact path to load labelled passages from. E.g. 'climatepolicyradar/Q913/labelled-passages:v0'"
        ),
    ],
    corpus_types_include: Annotated[
        Optional[list[str]],
        typer.Option(
            help="Corpus types to include. Can be specified multiple times. If not set, all types are included.",
            click_type=click.Choice(CORPUS_TYPES),
        ),
    ] = None,
    corpus_types_exclude: Annotated[
        Optional[list[str]],
        typer.Option(
            help="Corpus types to exclude. Can be specified multiple times.",
            click_type=click.Choice(CORPUS_TYPES),
        ),
    ] = None,
    track_and_upload: bool = typer.Option(
        True,
        help="Whether to track the run and upload the filtered passages to W&B",
    ),
):
    """
    Load labelled passages from W&B, filter by corpus type, and re-upload to a new run.

    The new artifact is linked to the source artifact via W&B lineage.
    """

    if not corpus_types_include and not corpus_types_exclude:
        raise typer.BadParameter(
            "At least one of --corpus-types-include or --corpus-types-exclude must be provided."
        )

    wandb_api = wandb.Api()

    source_artifact = wandb_api.artifact(labelled_passages_wandb_path)
    labelled_passages: list[LabelledPassage] = load_labelled_passages_from_wandb(
        wandb_path=labelled_passages_wandb_path
    )

    original_count = len(labelled_passages)
    console.log(f"Loaded {original_count} passages from W&B")

    # Filter by corpus type
    if corpus_types_include:
        labelled_passages = [
            p
            for p in labelled_passages
            if p.metadata.get(CORPUS_TYPE_KEY) in corpus_types_include
        ]
        console.log(
            f"Filtered to corpus types {corpus_types_include}: "
            f"{len(labelled_passages)} passages remain"
        )

    if corpus_types_exclude:
        labelled_passages = [
            p
            for p in labelled_passages
            if p.metadata.get(CORPUS_TYPE_KEY) not in corpus_types_exclude
        ]
        console.log(
            f"Excluded corpus types {corpus_types_exclude}: "
            f"{len(labelled_passages)} passages remain"
        )

    console.log(
        f"Filtered {original_count} → {len(labelled_passages)} passages "
        f"(removed {original_count - len(labelled_passages)})"
    )

    wikibase = WikibaseSession()
    concept = wikibase.get_concept(wikibase_id, include_labels_from_subconcepts=True)

    wandb_config = {
        "source_artifact": source_artifact.qualified_name,
        "corpus_types_include": corpus_types_include,
        "corpus_types_exclude": corpus_types_exclude,
        "original_count": original_count,
        "filtered_count": len(labelled_passages),
    }

    with (
        wandb.init(
            entity=WANDB_ENTITY,
            project=wikibase_id,
            job_type="filter_labelled_passages",
            config=wandb_config,
        )
        if track_and_upload
        else nullcontext()
    ) as run:
        if track_and_upload and run:
            run.use_artifact(source_artifact)

        if track_and_upload and run:
            console.log("📄 Creating filtered labelled passages artifact")
            log_labelled_passages_artifact_to_wandb_run(
                labelled_passages=labelled_passages,
                run=run,
                concept=concept,
            )
            console.log("✅ Filtered labelled passages uploaded successfully")


if __name__ == "__main__":
    app()
