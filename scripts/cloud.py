import os
from collections.abc import Callable
from enum import Enum
from typing import Any, Optional

import boto3
import boto3.session
import botocore
import botocore.client
import typer
from prefect.blocks.system import JSON
from pydantic import BaseModel, Field

from src.identifiers import WikibaseID

PROJECT_NAME = "knowledge-graph"


async def get_prefect_job_variable(param_name: str) -> str:
    """Get a single variable from the Prefect job variables."""
    aws_env = AwsEnv(os.environ["AWS_ENV"])
    block_name = f"default-job-variables-prefect-mvp-{aws_env}"
    workpool_default_job_variables = await JSON.load(block_name)  # pyright: ignore[reportGeneralTypeIssues]
    return workpool_default_job_variables.value[param_name]


class Namespace(BaseModel):
    """Hierarchy we use: CPR / {concept} / {classifier}"""

    project: WikibaseID = Field(
        ...,
        description="The name of the W&B project, which is the concept ID",
    )
    entity: str = Field(
        ...,
        description="The name of the W&B entity",
    )


class AwsEnv(str, Enum):
    """The only available AWS environments."""

    labs = "labs"
    sandbox = "sandbox"
    staging = "staging"
    production = "prod"

    def __str__(self):
        """Return a string representation"""
        return self.value

    @classmethod
    def _missing_(cls, value):
        if value == "dev":
            return cls.staging
        if value == "production":
            return cls.production


VALID_FROM_TO_TRANSITIONS = [
    (AwsEnv.sandbox, AwsEnv.labs),
    (AwsEnv.sandbox, AwsEnv.staging),
    (AwsEnv.labs, AwsEnv.staging),
    (AwsEnv.staging, AwsEnv.production),
]


def throw_not_logged_in(aws_env: AwsEnv):
    """Raise a typer.BadParameter exception for a not logged in AWS environment."""
    raise typer.BadParameter(
        f"you're not logged into {aws_env.value}. Do `aws sso --login {aws_env.value}`"
    )


def validate_transition(from_aws_env: AwsEnv, to_aws_env: AwsEnv) -> None:
    if (from_aws_env, to_aws_env) not in VALID_FROM_TO_TRANSITIONS:
        raise ValueError(
            f"cannot deploy from {from_aws_env.value} → {to_aws_env.value}"
        )


def generate_deployment_name(flow_name: str, aws_env: AwsEnv):
    project_initials = "".join([i[0] for i in PROJECT_NAME.split("-")])
    return f"{project_initials}-{flow_name}-{aws_env}"


def function_to_flow_name(fn: Callable[..., Any]) -> str:
    return fn.__name__.replace("_", "-")


def get_session(aws_env: AwsEnv) -> boto3.session.Session:
    """Create an AWS session using the specified AWS environment."""
    return boto3.Session(profile_name=aws_env.value)


def get_s3_client(
    aws_env: Optional[AwsEnv],
    region_name: str,
) -> botocore.client.BaseClient:
    """
    Create an AWS S3 client.

    Uses the specified AWS environment and region.
    """
    match aws_env:
        case None:
            return boto3.client("s3", region_name=region_name)

        case aws_env:
            session = get_session(aws_env)
            return session.client("s3", region_name=region_name)


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
            raise ValueError(f"'{value}' is not one of {valid}.")
        else:
            raise ValueError(str(e))


def get_sts_client(
    aws_env: Optional[AwsEnv],
) -> botocore.client.BaseClient:
    """
    Create an AWS STS client.

    Uses the specified AWS environment.
    """
    match aws_env:
        case None:
            return boto3.client("sts")

        case aws_env:
            session = get_session(aws_env)
            return session.client("sts")


def is_logged_in(aws_env: AwsEnv, use_aws_profiles: bool) -> bool:
    """Check if the user is logged in to the specified AWS environment."""
    try:
        aws_env_as_profile = aws_env if use_aws_profiles else None

        sts = get_sts_client(aws_env_as_profile)
        sts.get_caller_identity()  # pyright: ignore[reportAttributeAccessIssue]

        return True
    except (
        botocore.exceptions.ClientError,  # type: ignore
        botocore.exceptions.NoCredentialsError,  # type: ignore
        botocore.exceptions.SSOTokenLoadError,  # type: ignore
    ) as e:
        print(f"determining that not logged in due to exception: {e}")
        return False
