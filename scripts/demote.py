"""A script for model demotion."""

import logging
import os
from typing import Annotated

import typer
import wandb
from rich.logging import RichHandler

from knowledge_graph.cloud import (
    AwsEnv,
    is_logged_in,
    parse_aws_env,
    throw_not_logged_in,
)
from knowledge_graph.identifiers import WikibaseID

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(rich_tracebacks=True),
    ],
)

log = logging.getLogger("rich")

REGISTRY_NAME = "model"
JOB_TYPE = "demote_model"

REGION_NAME = "eu-west-1"


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

    collection_name = wikibase_id

    target_path = f"wandb-registry-{REGISTRY_NAME}/{collection_name}:{aws_env}"

    api = wandb.Api()

    model = api.artifact(target_path)
    model.aliases.remove(aws_env.value)
    model.save()

    log.info("Model demoted")


if __name__ == "__main__":
    app()
