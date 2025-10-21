"""A script for model demotion."""

import logging
import os
from typing import Annotated, Optional

import typer
import wandb
from rich.logging import RichHandler

from knowledge_graph.cloud import (
    AwsEnv,
    is_logged_in,
    parse_aws_env,
    throw_not_logged_in,
)
from knowledge_graph.config import WANDB_ENTITY
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.version import Version, get_latest_model_version

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(rich_tracebacks=True),
    ],
)

log = logging.getLogger("rich")

WANDB_MODEL_ORG = "climatepolicyradar_UZODYJSN66HCQ"
WANDB_MODEL_REGISTRY = "wandb-registry-model"
JOB_TYPE = "demote_model"


def validate_login(
    aws_env: AwsEnv,
    use_aws_profiles: bool,
) -> None:
    """Validate that the user is logged in to the necessary AWS environment."""
    if not is_logged_in(aws_env, use_aws_profiles):
        throw_not_logged_in(aws_env)


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
    aws_env: Annotated[
        AwsEnv,
        typer.Option(
            help="AWS environment to demote the model artifact in",
            parser=parse_aws_env,
        ),
    ],
    wandb_registry_version: Annotated[
        Optional[Version],
        typer.Option(
            help="Optional: specific registry version of the model to demote",
            parser=Version,
        ),
    ] = None,
):
    """
    Demote a model within an AWS environment.

    This removes the environment alias from the specified model in the registry,
    effectively making it no longer the primary version for that environment.
    """
    log.info("Starting model demotion process")

    use_aws_profiles = os.environ.get("USE_AWS_PROFILES", "true").lower() == "true"

    log.info("Validating AWS login...")
    validate_login(aws_env, use_aws_profiles)

    if not wandb_registry_version:
        log.info(f"Getting latest model version for AWS environment {aws_env.value}...")
        api = wandb.Api()

        registry_filters = {"name": {"$regex": "model"}}
        collection_filters = {"name": wikibase_id}
        version_filters = {"tag": aws_env.name}

        artifacts = (
            api.registries(filter=registry_filters)
            .collections(collection_filters)
            .versions(filter=version_filters)
        )
        classifier_version = get_latest_model_version(artifacts, aws_env)
        log.info(
            f"Latest model version for AWS environment {aws_env.value} is {classifier_version}"
        )
    else:
        log.info(
            f"Demoting specific registry version {wandb_registry_version} for wikibase id {wikibase_id} with AWS environment {aws_env.value}..."
        )
        classifier_version = wandb_registry_version

    log.info("Initialising Weights & Biases run...")
    with wandb.init(entity=WANDB_ENTITY, project=wikibase_id, job_type=JOB_TYPE) as run:
        target_path = f"{WANDB_MODEL_REGISTRY}/{wikibase_id}"

        artifact_id = f"{target_path}:{classifier_version}"
        log.info(f"Using model artifact: {artifact_id}...")
        model: wandb.Artifact = run.use_artifact(artifact_id)

        # validate tag exists
        if model.tags is None or aws_env.value not in model.tags:
            raise typer.BadParameter(
                f"Model {artifact_id} does not contain tag: {aws_env.value}"
            )
        # validate model trained for this aws env
        if model.metadata.get("aws_env") != aws_env.value:
            raise typer.BadParameter(
                f"Model {artifact_id} is not promoted in AWS environment {aws_env.value}"
            )

        # remove all classifiers profiles
        model.metadata.pop("classifiers_profiles", None)

        # remove aws env tag
        model.tags.remove(aws_env.value)
        model.save()

        log.info(f"Model {artifact_id} demoted")


if __name__ == "__main__":
    app()
