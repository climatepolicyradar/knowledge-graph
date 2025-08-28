import os
from collections.abc import Callable, Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import boto3
import boto3.session
import botocore
import botocore.client
import typer
import yaml
from prefect.blocks.system import JSON
from pydantic import BaseModel, Field

from src.identifiers import WikibaseID

PROJECT_NAME = "knowledge-graph"
SPEC_DIR = Path("flows") / "classifier_specs"


# Version 1 classifier spec, to be cleaned up and replaced
# with model from `flows/classifier_specs/spec_interface.py`
class ClassifierSpec(BaseModel):
    """Details for a classifier to run."""

    name: str = Field(
        description="The reference of the classifier in wandb. e.g. 'Q992'",
        min_length=1,
    )
    alias: str = Field(
        description=(
            "The alias tag for the version to use for inference. e.g 'latest' or 'v2'"
        ),
        min_length=1,
    )

    def __hash__(self):
        """Make ClassifierSpec hashable for use in sets and as dict keys."""
        return hash((self.name, self.alias))

    def __str__(self):
        """Return a string representation of the classifier spec."""
        return f"{self.name}:{self.alias}"

    def __repr__(self):
        """Return a string representation of the classifier spec."""
        return f"{self.name}:{self.alias}"


def disallow_latest_alias(classifier_specs: Sequence[ClassifierSpec]):
    if any(classifier_spec.alias == "latest" for classifier_spec in classifier_specs):
        raise ValueError("`latest` is not allowed")
    return None


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
    use_aws_profiles: bool = os.environ.get("USE_AWS_PROFILES", "false").lower() == "true"
    if not use_aws_profiles:
        aws_env = None
    
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


# Version 1 classifier spec helper, to be cleaned up and replaced
# with model from `flows/classifier_specs/spec_interface.py`
def build_spec_file_path(aws_env: AwsEnv) -> Path:
    file_path = SPEC_DIR / f"{aws_env}.yaml"
    return file_path


# Version 1 classifier spec helper, to be cleaned up and replaced
# with model from `flows/classifier_specs/spec_interface.py`
def read_spec_file(aws_env: AwsEnv) -> list[str]:
    file_path = build_spec_file_path(aws_env)
    with open(file_path, "r") as file:
        return yaml.load(file, Loader=yaml.FullLoader)


# Version 1 classifier spec helper, to be cleaned up and replaced
# with model from `flows/classifier_specs/spec_interface.py`
def parse_spec_file(aws_env: AwsEnv) -> list[ClassifierSpec]:
    contents = read_spec_file(aws_env)
    classifier_specs: list[ClassifierSpec] = []
    for item in contents:
        try:
            name, alias = item.split(":")
            classifier_specs.append(ClassifierSpec(name=name, alias=alias))
        except ValueError:
            raise ValueError(f"Invalid format in spec file: {item}")

    return classifier_specs


# Version 1 classifier spec helper, to be cleaned up and replaced
# with model from `flows/classifier_specs/spec_interface.py`
def write_spec_file(file_path: Path, data: list[ClassifierSpec]):
    """Save a classifier spec YAML"""
    serialised_data = list(map(lambda spec: str(spec), data))
    with open(file_path, "w") as file:
        yaml.dump(serialised_data, file, explicit_start=True)
