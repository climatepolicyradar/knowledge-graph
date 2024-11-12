from enum import Enum
from typing import Optional

import boto3
import boto3.session
import botocore
import botocore.client
from pydantic import BaseModel, Field

from src.identifiers import WikibaseID

PROJECT_NAME = "knowledge-graph"


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

    @classmethod
    def _missing_(cls, value):
        if value == "dev":
            return cls.staging
        if value == "production":
            return cls.production


def generate_deployment_name(flow_name: str, aws_env: AwsEnv):
    return f"{PROJECT_NAME}-{flow_name}-{aws_env}"


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
        aws_env = aws_env if use_aws_profiles else None

        sts = get_sts_client(aws_env)
        sts.get_caller_identity()

        return True
    except (
        botocore.exceptions.ClientError,
        botocore.exceptions.NoCredentialsError,
        botocore.exceptions.SSOTokenLoadError,
    ) as e:
        print(f"determining that not logged in due to exception: {e}")
        return False
