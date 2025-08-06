"""A script for model promotion."""

import logging
import os
from typing import Annotated, Optional, Union

import typer
import wandb
import wandb.apis.public.api

from scripts.cloud import (
    AwsEnv,
    is_logged_in,
    parse_aws_env,
    throw_not_logged_in,
)
from scripts.utils import ModelPath
from src.identifiers import ClassifierID, WikibaseID

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# This magic value was from the W&B webapp.
ORG_ENTITY = "climatepolicyradar_UZODYJSN66HCQ"
REGISTRY_NAME = "model"
ENTITY = "climatepolicyradar"
JOB_TYPE = "promote_model"

app = typer.Typer()


def find_artifact_in_registry(
    model_collection, classifier_id: ClassifierID, aws_env: AwsEnv
) -> Optional[wandb.Artifact]:
    """
    Find an artifact with the specified alias in the model collection.

    This runs through artifacts in the collection and inspects them, checking if they
    have the id we are looking for, and also inspecting the alias for the aws_env.
    """
    for art in model_collection.artifacts():
        found_classifier_id, _ = art.source_name.split(":")
        aliases = art.aliases
        if found_classifier_id == str(classifier_id) and aws_env.value in aliases:
            return art


def check_existing_artifact_aliases(
    api: wandb.apis.public.api.Api,
    target_path: str,
    classifier_id: ClassifierID,
    aws_env: AwsEnv,
) -> None:
    """
    Review current state of the model in the wandb registry.

    This means first checking wether the collection itself exists (/Q123).
    Then if found, will look if the artifact already exists within the collection.
    If the artifact exists, will check the aliases to see if there are any conflicts.

    Note that the versions for project artifacts are not the same as the versions for
    collection models.
    """
    if not api.artifact_collection_exists(
        type="model",
        name=target_path,
    ):
        log.info("Model collection doesn't already exist")
        return None

    log.info("Model collection does already exist")
    model_collection = api.artifact_collection(
        type_name="model",
        name=target_path,
    )

    target_artifact = find_artifact_in_registry(
        model_collection, classifier_id, aws_env=aws_env
    )

    # It's okay if there isn't yet an registry artifact for this artifact, and
    # if there isn't, then there's nothing to check.
    if not target_artifact:
        log.info(f"Model collection artifact with alias {aws_env.value} not found")
        return None

    log.info(f"Model collection artifact with alias {aws_env.value} found")

    # Get all AWS env values except the one we're promoting to
    other_env_values = {env.value for env in AwsEnv} - {aws_env.value}

    # Check if any other AWS environment values are present as aliases
    if set(target_artifact.aliases) & other_env_values:
        raise typer.BadParameter(
            "Something has gone wrong with the source artifact, multiple AWS "
            f"environments where found in the aliases: {target_artifact.aliases}"
        )


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
    primary: Annotated[
        bool,
        typer.Option(
            help="Whether this will be primary for this AWS environment",
        ),
    ] = False,
):
    """
    Promote a model to the registry so it can be used downstream.

    The model should already have been trained, meaning it will exist
    as an artefact in wandb, with a linked model in s3 for the environment.

    This script adds a link to the chosen model from the wandb registry as a
    collection. Optionally the model can be made primary for the AWS
    environment, this means applying an environment alias to the model in the
    collection.

    If a W&B model registry collection doesn't exist, it'll automatically be
    made as part of this script.

    Note: promoting between environments is not yet supported.
    """
    log.info("Starting model promotion process")

    log.info("Validating AWS logins...")
    use_aws_profiles = os.environ.get("USE_AWS_PROFILES", "true").lower() == "true"
    if not is_logged_in(aws_env, use_aws_profiles):
        throw_not_logged_in(aws_env)

    collection_name = wikibase_id

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

    log.info("Initializing Weights & Biases run...")
    with wandb.init(entity=ENTITY, project=wikibase_id, job_type=JOB_TYPE) as run:
        # Regardless of the promotion, we'll always be using some artifact.
        #
        # This also validates that the classifier exists. It relies on an
        # artifiact not existing. That is, when trying to `use_artifact`
        # below, it'll throw an exception.
        model_path = ModelPath(wikibase_id=wikibase_id, classifier_id=classifier_id)
        artifact_id = f"{model_path}:{aws_env.value}"
        log.info(f"Using model artifact: {artifact_id}...")
        artifact: wandb.Artifact = run.use_artifact(artifact_id)

        api = wandb.Api()

        check_existing_artifact_aliases(
            api,
            target_path,
            classifier_id,
            aws_env,
        )

        aliases: Union[list[str], None] = [aws_env.value] if primary else None

        # Link the artifact to a collection
        log.info(f"Linking artifact to collection: {target_path}...")
        run.link_artifact(
            artifact=artifact,
            target_path=target_path,
            aliases=aliases,
        )

        log.info("Model promoted")


if __name__ == "__main__":
    app()
