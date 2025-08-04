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
)
from src.identifiers import WikibaseID
from src.version import Version

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# This magic value was from the W&B webapp.
ORG_ENTITY = "climatepolicyradar_UZODYJSN66HCQ"
REGISTRY_NAME = "model"
ENTITY = "climatepolicyradar"
JOB_TYPE = "promote_model"

app = typer.Typer()


def throw_not_logged_in(aws_env: AwsEnv):
    """Raise a typer.BadParameter exception for a not logged in AWS environment."""
    raise typer.BadParameter(
        f"you're not logged into {aws_env.value}. Do `aws sso --login {aws_env.value}`"
    )


def find_artifact_by_version(
    model_collection, version: Version
) -> Optional[wandb.Artifact]:
    """Find an artifact with the specified version in the model collection."""
    return next(
        (art for art in model_collection.artifacts() if art.version == str(version)),
        None,
    )


def check_existing_artifact_aliases(
    api: wandb.apis.public.api.Api,
    target_path: str,
    version: Version,
    aws_env: AwsEnv,
) -> None:
    """Check if an artifact exists and has conflicting AWS environment aliases."""
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

    target_artifact = find_artifact_by_version(model_collection, version)

    # It's okay if there isn't yet an artifact for this version, and
    # if there isn't, then there's nothing to check.
    if not target_artifact:
        log.info(f"Model collection artifact with version {version} not found")
        return None

    log.info(f"Model collection artifact with version {version} found")

    # Get all AWS env values except the one we're promoting to
    other_env_values = {env.value for env in AwsEnv} - {aws_env.value}
    # Check if any other AWS environment values are present as aliases

    if existing_env_aliases := set(target_artifact.aliases) & other_env_values:
        raise typer.BadParameter(
            "An artifact already exists with AWS environment aliases "
            f"{existing_env_aliases} in collection {target_path}."
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
    classifier: Annotated[
        str,
        typer.Option(
            help="Classifier name that aligns with the Python class name",
        ),
    ],
    version: Annotated[
        Version,
        typer.Option(
            help="Version of the model (e.g., v3)",
            parser=Version,
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
            help="Whether this will be the primary version for this AWS environment",
        ),
    ] = False,
):
    """
    Promote a model to the registry so it can be used downstream.

    The model should already have been trained, meaning it will exist
    as an artefact in wandb, with a linked model in s3 for the environment.

    This script adds a link to the chosen model from the wandb registry as a
    collection. Optionally the model can be made the primary version for the
    AWS environment.

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
    run = wandb.init(entity=ENTITY, project=wikibase_id, job_type=JOB_TYPE)

    # Regardless of the promotion, we'll always be using some artifact.
    #
    # This also validates that the classifier exists. It relies on an
    # artifiact not existing. That is, when trying to `use_artifact`
    # below, it'll throw an exception.
    artifact_id = f"{wikibase_id}/{classifier}:{version}"
    log.info(f"Using model artifact: {artifact_id}...")
    artifact: wandb.Artifact = run.use_artifact(artifact_id)

    api = wandb.Api()

    check_existing_artifact_aliases(
        api,
        target_path,
        version,
        aws_env,
    )

    aliases: Union[list[str], None] = [aws_env.value] if primary else None

    # Link the artifact to a collection
    #
    # It will either be the Artifact that we originally used, if a
    # _within_, or a newly logged Artifact, if _across_.
    log.info(f"Linking artifact to collection: {target_path}...")
    run.link_artifact(
        artifact=artifact,
        target_path=target_path,
        aliases=aliases,
    )

    log.info("Finishing W&B run...")

    run.finish()

    log.info("Model promoted")


if __name__ == "__main__":
    app()
