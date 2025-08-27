"""A script for editing classifier metadata in wandb."""

import logging
from typing import Annotated

import typer
import wandb
import wandb.apis.public.api
from rich.console import Console

from flows.classifier_specs.spec_interface import load_classifier_specs
from scripts.cloud import AwsEnv
from scripts.config import WANDB_ENTITY
from scripts.update_classifier_spec import get_all_available_classifiers
from scripts.utils import DontRunOnEnum, ModelPath
from src.identifiers import ClassifierID, WikibaseID

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

REGISTRY_NAME = "model"
JOB_TYPE = "configure_model"

app = typer.Typer()
console = Console()


@app.command()
def update_entire_env(
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
    update_specs: Annotated[
        bool,
        typer.Option(
            help="Also update the classifier specs for the environment following changes"
        ),
    ] = True,
):
    specs = load_classifier_specs(aws_env)
    for spec in specs:
        update(
            wikibase_id=spec.wikibase_id,
            classifier_id=spec.classifier_id,
            clear_dont_run_on=clear_dont_run_on,
            add_dont_run_on=add_dont_run_on,
            aws_env=aws_env,
            update_specs=False,  # since we'll only update once all are done
        )

    if update_specs:
        get_all_available_classifiers([aws_env])


@app.command()
def update(
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
    update_specs: Annotated[
        bool,
        typer.Option(
            help="Also update the classifier specs for the environment following changes"
        ),
    ] = True,
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

    with wandb.init(entity=WANDB_ENTITY, project=wikibase_id, job_type=JOB_TYPE) as run:
        artifact: wandb.Artifact = run.use_artifact(artifact_id)

        if clear_dont_run_on:
            console.log(f"Clearing existing `dont_run_on` metadata from {artifact_id}")
            if add_dont_run_on:
                console.log(
                    "`add_dont_run_on` is set, so values will be fully refreshed"
                )

            artifact.metadata.pop("dont_run_on", None)

        if add_dont_run_on:
            additions: list[str] = [a.value for a in add_dont_run_on]
            console.log(f"Applying {additions=} to {artifact_id}")
            current: list[str] = artifact.metadata.get("dont_run_on", [])
            update: list[str] = list(set(current + additions))
            artifact.metadata["dont_run_on"] = update

        artifact.save()

    if update_specs:
        get_all_available_classifiers([aws_env])


if __name__ == "__main__":
    app()
