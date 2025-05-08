"""A script for model demotion."""

import logging
import os
from typing import Annotated

import typer
from rich.logging import RichHandler

import wandb
from scripts.cloud import AwsEnv, is_logged_in
from src.identifiers import WikibaseID

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(rich_tracebacks=True),
    ],
)

log = logging.getLogger("rich")

ORG_ENTITY = "climatepolicyradar_UZODYJSN66HCQ"
REGISTRY_NAME = "model"
ENTITY = "climatepolicyradar"
JOB_TYPE = "demote_model"

REGION_NAME = "eu-west-1"


def throw_not_logged_in(aws_env: AwsEnv):
    """Raise a typer.BadParameter exception for a not logged in AWS environment."""
    raise typer.BadParameter(
        f"you're not logged into {aws_env.value}. Do `aws sso --login {aws_env.value}`"
    )


def validate_login(
    aws_env: AwsEnv,
    use_aws_profiles: bool,
) -> None:
    """Validate that the user is logged in to the necessary AWS environment."""
    if not is_logged_in(aws_env, use_aws_profiles):
        throw_not_logged_in(aws_env)


app = typer.Typer()


def parse_aws_env(value: str) -> str:
    """
    Parse a string a string as a possible enum value.

    We rely on a somewhat custom enum, to allow `"dev"`|`"staging"` for
    `staging`.
    """
    try:
        # This would convert `"dev"` to `AwsEnv.staging`.
        #
        # The raw value is returned, since we can't return an `AwsEnv` from
        # this function.
        return AwsEnv(value).value
    except ValueError as e:
        if "is not a valid AwsEnv" in str(e):
            valid = ", ".join([f"'{env.value}'" for env in AwsEnv])
            raise typer.BadParameter(f"'{value}' is not one of {valid}.")
        else:
            raise typer.BadParameter(str(e))


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

    This removes the environment alias from the specified version, effectively
    making it no longer the primary version for that environment.
    """
    log.info("Starting model demotion process")

    use_aws_profiles = os.environ.get("USE_AWS_PROFILES", "true").lower() == "true"

    log.info("Validating AWS login...")
    validate_login(aws_env, use_aws_profiles)

    collection_name = wikibase_id

    target_path = f"wandb-registry-{REGISTRY_NAME}/{collection_name}:{aws_env}"

    api = wandb.Api()  # type: ignore[reportGeneralTypeIssues]

    model = api.artifact(target_path)
    model.aliases.remove(aws_env.value)
    model.save()

    log.info("Model demoted")


if __name__ == "__main__":
    app()
