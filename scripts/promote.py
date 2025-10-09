"""A script for model promotion."""

import logging
import os
from typing import Annotated

import typer
import wandb
import wandb.apis.public.api

from knowledge_graph.classifier import ModelPath
from knowledge_graph.cloud import (
    AwsEnv,
    is_logged_in,
    parse_aws_env,
    throw_not_logged_in,
)
from knowledge_graph.config import WANDB_ENTITY
from knowledge_graph.identifiers import ClassifierID, WikibaseID
from knowledge_graph.version import Version

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# This magic value was from the W&B webapp.
REGISTRY_NAME = "model"
JOB_TYPE = "promote_model"

app = typer.Typer()


@app.command()
def main(
    wikibase_id: Annotated[
        WikibaseID,
        typer.Option(
            help="Wikibase ID of the concept",
            parser=WikibaseID,
        ),
    ],
    classifier_id: Annotated[
        ClassifierID,
        typer.Option(
            help="Classifier ID that aligns with the Python class name",
            parser=ClassifierID,
        ),
    ],
    aws_env: Annotated[
        AwsEnv,
        typer.Option(
            help="AWS environment to promote the model artifact within",
            parser=parse_aws_env,
        ),
    ],
    add_classifiers_profiles: Annotated[
        list[str] | None,
        typer.Option(help="Adds 1 or more classifiers profiles."),
    ] = None,
    remove_classifiers_profiles: Annotated[
        list[str] | None,
        typer.Option(help="Removes 1 or more classifiers profiles."),
    ] = None,
):
    """
    Promote a model to the registry so it can be used downstream.

    The model should already have been trained, meaning it will exist
    as an artifact in W&B, with a linked model in s3 for the environment.

    This script adds a link to the chosen model from the W&B registry as a
    collection, and tags it with the AWS environment.

    If a W&B model registry collection doesn't exist, it'll automatically be
    made as part of this script.

    Note: promoting between environments is not yet supported.
    """
    log.info("Starting model promotion process")

    log.info("Validating classifiers profiles...")
    add_class_prof: set[str] = (
        set(add_classifiers_profiles) if add_classifiers_profiles else set()
    )
    remove_class_prof: set[str] = (
        set(remove_classifiers_profiles) if remove_classifiers_profiles else set()
    )
    if dupes := add_class_prof & remove_class_prof:
        raise typer.BadParameter(
            f"duplicate values found for adding and removing classifiers profiles: `{','.join(dupes)}`"
        )

    log.info("Validating AWS logins...")
    use_aws_profiles = os.environ.get("USE_AWS_PROFILES", "true").lower() == "true"
    if not is_logged_in(aws_env, use_aws_profiles):
        throw_not_logged_in(aws_env)

    collection_name = wikibase_id
    model_path = ModelPath(wikibase_id=wikibase_id, classifier_id=classifier_id)

    # Get all artifacts for the model path to select latest version for aws_env
    log.info(f"Getting latest model version for AWS environment {aws_env.value}...")
    api = wandb.Api()

    artifacts = api.artifacts(type_name="model", name=f"{model_path}")
    current_env_versions = [
        Version(art.version)
        for art in artifacts
        if art.metadata.get("aws_env") == aws_env.value
    ]
    classifier_version = max(current_env_versions)

    # This is the hierarchy we use: CPR / {concept} / {model architecture}(s)
    #
    # The concept aka Wikibase ID is the collection name.
    #
    # > W&B automatically creates a collection with the name you specify
    # > in the target path if you try to link an artifact to a collection
    # > that does not exist. [1]
    #
    # [1] https://docs.wandb.ai/guides/registry/create_collection#programmatically-create-a-collection
    target_path = f"wandb-registry-{REGISTRY_NAME}/{collection_name}"
    log.info(f"Using model collection: {target_path}...")

    log.info("Initialising Weights & Biases run...")
    with wandb.init(entity=WANDB_ENTITY, project=wikibase_id, job_type=JOB_TYPE) as run:
        # Regardless of the promotion, we'll always be using some artifact.
        #
        # This also validates that the classifier exists. It relies on an
        # artifact not existing. That is, when trying to `use_artifact`
        # below, it'll throw an exception.
        artifact_id = f"{model_path}:{classifier_version}"
        log.info(f"Using model artifact: {artifact_id}...")
        artifact: wandb.Artifact = run.use_artifact(artifact_id)

        # Check that classifiers profiles are defined
        current_class_prof = set(artifact.metadata.get("classifiers_profiles", []))
        if not current_class_prof and not add_class_prof:
            raise typer.BadParameter(
                "Artifact must have at least one classifiers profile in metadata, or you must specify at least 1 to add. "
            )

        if (
            classifiers_profiles := (current_class_prof | add_class_prof)
            - remove_class_prof
        ):
            artifact.metadata["classifiers_profiles"] = classifiers_profiles
        else:
            artifact.metadata.pop("classifiers_profiles", None)

        log.info(f"Artifact has classifier profiles: {current_class_prof}")

        # Add AWS environment as a tag
        current_tags = set(artifact.tags or [])
        current_tags.add(aws_env.value)
        artifact.tags = list(current_tags)
        artifact.save()
        log.info(f"Added AWS environment tag: {aws_env.value}")

        # Link the artifact to a collection
        log.info(f"Linking artifact to collection: {target_path}...")
        run.link_artifact(
            artifact=artifact,
            target_path=target_path,
            aliases=None,
        )

        log.info("Model promoted")


if __name__ == "__main__":
    app()
