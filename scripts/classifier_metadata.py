"""
CLI wrapper for editing classifier metadata in wandb.

The reusable logic lives in `flows.classifier_metadata`; this module only adds the Typer
commands used by `just classifier-metadata` and the `classifier-metadata` console script.
"""

from typing import Annotated

import typer

from flows.classifier_metadata import update as run_update
from flows.classifier_metadata import update_entire_env as run_update_entire_env
from flows.classifier_specs.spec_interface import DontRunOnEnum
from knowledge_graph.cloud import AwsEnv
from knowledge_graph.identifiers import ClassifierID, WikibaseID

app = typer.Typer()


@app.command()
def update_entire_env(
    clear_dont_run_on: Annotated[
        bool, typer.Option(help="Remove all existing items from dont_run_on")
    ] = False,
    add_dont_run_on: Annotated[
        list[DontRunOnEnum] | None,
        typer.Option(help="Adds a single item to the metadata."),
    ] = None,
    clear_require_gpu: Annotated[
        bool, typer.Option(help="updates `compute_environment.gpu` to remove the field")
    ] = False,
    add_require_gpu: Annotated[
        bool, typer.Option(help="updates `compute_environment.gpu` to True")
    ] = False,
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
    max_workers: Annotated[
        int,
        typer.Option(help="Max number of threads to use, one classifier per thread."),
    ] = 8,
):
    """Update classifier metadata for every classifier in an envs spec."""
    try:
        run_update_entire_env(
            clear_dont_run_on=clear_dont_run_on,
            add_dont_run_on=add_dont_run_on,
            clear_require_gpu=clear_require_gpu,
            add_require_gpu=add_require_gpu,
            aws_env=aws_env,
            update_specs=update_specs,
            max_workers=max_workers,
        )
    except ValueError as e:
        raise typer.BadParameter(str(e))


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
        typer.Option(help="Adds 1 or more items to the metadata."),
    ] = None,
    clear_require_gpu: Annotated[
        bool, typer.Option(help="updates `compute_environment.gpu` to remove the field")
    ] = False,
    add_require_gpu: Annotated[
        bool, typer.Option(help="updates `compute_environment.gpu` to True")
    ] = False,
    aws_env: Annotated[
        AwsEnv,
        typer.Option(help="AWS environment the classifier belongs to"),
    ] = AwsEnv.labs,
    add_classifiers_profiles: Annotated[
        list[str] | None,
        typer.Option(help="Adds 1 classifiers profile to the metadata."),
    ] = None,
    remove_classifiers_profiles: Annotated[
        list[str] | None,
        typer.Option(help="Removes 1 or more classifiers profiles from the metadata."),
    ] = None,
    update_specs: Annotated[
        bool,
        typer.Option(
            help="Also update the classifier specs for the environment following changes"
        ),
    ] = True,
):
    """Update the metadata for a classifier to determine behaviour at inference time."""
    try:
        run_update(
            wikibase_id=wikibase_id,
            classifier_id=classifier_id,
            clear_dont_run_on=clear_dont_run_on,
            add_dont_run_on=add_dont_run_on,
            clear_require_gpu=clear_require_gpu,
            add_require_gpu=add_require_gpu,
            aws_env=aws_env,
            add_classifiers_profiles=add_classifiers_profiles,
            remove_classifiers_profiles=remove_classifiers_profiles,
            update_specs=update_specs,
        )
    except ValueError as e:
        raise typer.BadParameter(str(e))


if __name__ == "__main__":
    app()
