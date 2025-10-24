"""A script for editing classifier metadata in wandb."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Annotated

import typer
import wandb
from rich.console import Console

from flows.classifier_specs.spec_interface import DontRunOnEnum, load_classifier_specs
from knowledge_graph.classifier import ModelPath
from knowledge_graph.cloud import AwsEnv
from knowledge_graph.config import WANDB_ENTITY
from knowledge_graph.identifiers import ClassifierID, WikibaseID
from knowledge_graph.version import get_latest_model_version
from scripts.update_classifier_spec import refresh_all_available_classifiers

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

REGISTRY_NAME = "model"
JOB_TYPE = "configure_model"

type ComputeEnvironment = dict[str, str | int | bool]


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
    specs = load_classifier_specs(aws_env)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for spec in specs:
            future = executor.submit(
                update,
                wikibase_id=spec.wikibase_id,
                classifier_id=spec.classifier_id,
                clear_dont_run_on=clear_dont_run_on,
                add_dont_run_on=add_dont_run_on,
                clear_require_gpu=clear_require_gpu,
                add_require_gpu=add_require_gpu,
                aws_env=aws_env,
                update_specs=False,  # since we'll only update once all are done
            )
            futures.append(future)

        for future in as_completed(futures):
            future.result()

    if update_specs:
        refresh_all_available_classifiers([aws_env])


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
        typer.Option(help="Adds 1 or more items to the metadata."),
    ] = None,
    remove_classifiers_profiles: Annotated[
        list[str] | None,
        typer.Option(help="Removes 1 or more items to the metadata."),
    ] = None,
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

    api = wandb.Api()
    artifacts = api.artifacts(type_name="model", name=f"{model_path}")
    classifier_version = get_latest_model_version(artifacts, aws_env)

    artifact_id = f"{model_path}:{classifier_version}"

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

        if clear_require_gpu:
            if add_require_gpu:
                raise typer.BadParameter(
                    "`clear-require-gpu` and `add-require-gpu` can't both be set"
                )

            if artifact.metadata.get("compute_environment"):
                artifact.metadata["compute_environment"].pop("gpu", None)

                # Remove if now empty
                if not artifact.metadata.get("compute_environment"):
                    artifact.metadata.pop("compute_environment", None)

        elif add_require_gpu:
            compute_environment: ComputeEnvironment = artifact.metadata.get(
                "compute_environment", {}
            )
            compute_environment: ComputeEnvironment = compute_environment | {
                "gpu": True
            }
            artifact.metadata["compute_environment"] = compute_environment

        if add_classifiers_profiles or remove_classifiers_profiles:
            add_class_prof: set[str] = (
                set(add_classifiers_profiles) if add_classifiers_profiles else set()
            )
            remove_class_prof: set[str] = (
                set(remove_classifiers_profiles)
                if remove_classifiers_profiles
                else set()
            )

            if dupes := add_class_prof & remove_class_prof:
                raise typer.BadParameter(
                    f"duplicate values found for adding and removing classifiers profiles: `{','.join(dupes)}`"
                )

            current_class_prof: set[str] = set(
                artifact.metadata.get("classifiers_profiles", [])
            )

            if (
                classifiers_profiles := (current_class_prof | add_class_prof)
                - remove_class_prof
            ):
                if len(classifiers_profiles) > 1:
                    raise typer.BadParameter(
                        f"Artifact must have maximum of one classifiers profile in metadata, or you must specify 1 to remove. Current classifiers profiles `{current_class_prof}`"
                    )
                else:
                    artifact.metadata["classifiers_profiles"] = classifiers_profiles
            else:
                artifact.metadata.pop("classifiers_profiles", None)

        artifact.save()

    if update_specs:
        refresh_all_available_classifiers([aws_env])


if __name__ == "__main__":
    app()
