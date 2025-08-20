"""A script for editing classifier metadata in wandb."""

import logging
from enum import Enum
from typing import Annotated

import typer
import wandb
import wandb.apis.public.api
from rich.console import Console

from scripts.cloud import AwsEnv
from scripts.utils import ModelPath
from src.identifiers import ClassifierID, WikibaseID

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# This magic value was from the W&B webapp.
ORG_ENTITY = "climatepolicyradar_UZODYJSN66HCQ"
REGISTRY_NAME = "model"
ENTITY = "climatepolicyradar"
JOB_TYPE = "configure_model"

app = typer.Typer()
console = Console()


class DontRunOnEnum(Enum):
    """A `source` that will be filtered out in inference."""

    sabin = "sabin"
    cclw = "cclw"
    cpr = "cpr"
    af = "af"
    cif = "cif"
    gcf = "gcf"
    gef = "gef"
    oep = "oep"
    unfccc = "unfccc"

    def __str__(self) -> str:
        """Return a string representation"""
        return self.value


@app.command()
def main(
    wikibase_id: Annotated[
        WikibaseID,
        typer.Option(
            help="Wikibase ID of the concept, e.g. Q123",
            parser=WikibaseID,
        ),
    ],
    classifier_id: Annotated[
        ClassifierID,
        typer.Option(
            help="Classifier ID hash, eg `8np4shsw`",
            parser=ClassifierID,
        ),
    ],
    clear_dont_run_on: Annotated[
        bool, typer.Option(help="Remove all existing items from dont_run_on")
    ] = False,
    add_dont_run_on: Annotated[
        list[DontRunOnEnum] | None,
        typer.Option(help="Adds a single item to the metadata."),
    ] = None,
    aws_env: Annotated[
        AwsEnv,
        typer.Option(help="AWS environment the classifier belongs to"),
    ] = AwsEnv.labs,
):
    """
    Updates the metadata for a classifier to determine behaviour at inference time.

    Set `dont_run_on` key to prevent a classifier from running on documents from a
    specific source. You can either add to existing items by passing
    `--add-dont-run-on <source>` or clear all items with `--clear-dont-run-on`
    use both to reset the list down to a new selection, eg:
    ```
    ... --clear-dont-run-on --add-dont-run-on gef --add-dont-run-on sabin
    ```
    """
    model_path = ModelPath(wikibase_id=wikibase_id, classifier_id=classifier_id)
    artifact_id = f"{model_path}:{aws_env.value}"

    with wandb.init(entity=ENTITY, project=wikibase_id, job_type=JOB_TYPE) as run:
        artifact: wandb.Artifact = run.use_artifact(artifact_id)

        if clear_dont_run_on:
            console.log(f"Clearing existing `dont_run_on` metadata from {artifact_id}")
            artifact.metadata.pop("dont_run_on", None)

        if add_dont_run_on:
            additions: list[str] = [a.value for a in add_dont_run_on]
            console.log(f"Applying {additions=} to {artifact_id}")
            current: list[str] = artifact.metadata.get("dont_run_on", [])
            update: list[str] = list(set(current + additions))
            artifact.metadata["dont_run_on"] = update

        artifact.save()


if __name__ == "__main__":
    app()
