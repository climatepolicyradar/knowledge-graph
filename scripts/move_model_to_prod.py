"""
A script for moving a classifier artifact from one environment to another in W&B.

Uploads a classifier artifact to a new environment (typically from labs to production),
creating a new W&B run with updated metadata and uploading to the target environment's
S3 bucket.

Note: This script only uploads to S3 and creates a W&B artifact. It does NOT promote
to the W&B registry. For registry promotion, use scripts/promote.py.
"""

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Annotated

import boto3
import botocore.exceptions
import typer
import wandb
from rich.console import Console

from knowledge_graph.classifier import (
    ModelPath,
    get_local_classifier_path,
)
from knowledge_graph.cloud import (
    AwsEnv,
    Namespace,
    get_session,
    parse_aws_env,
)
from knowledge_graph.config import WANDB_ENTITY
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.operations.train import (
    StorageUpload,
    get_next_version,
    upload_model_artifact,
)
from knowledge_graph.wandb_helpers import load_classifier_from_wandb

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

JOB_TYPE = "promote_to_prod"

app = typer.Typer()
console = Console()


AWS_CREDENTIAL_ENV_VARS = (
    "AWS_PROFILE",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
)


@contextmanager
def aws_profile(aws_env: AwsEnv, use_aws_profiles: bool) -> Generator[None]:
    """
    Temporarily point the default AWS credential chain at an env's profile.

    W&B builds its own boto3 clients for S3 references (downloads and
    checksums), so it only sees credentials via the environment. Setting
    AWS_PROFILE alone isn't enough, since exported AWS_ACCESS_KEY_ID etc.
    take precedence over it, so we export the profile's resolved credentials.
    """
    if not use_aws_profiles:
        yield
        return

    credentials = get_session(aws_env).get_credentials()
    if credentials is None:
        raise typer.BadParameter(
            f"No credentials for {aws_env.value}. "
            f"Run: aws sso login --profile {aws_env.value}"
        )
    frozen = credentials.get_frozen_credentials()
    assert frozen.access_key and frozen.secret_key

    previous = {name: os.environ.get(name) for name in AWS_CREDENTIAL_ENV_VARS}
    os.environ.pop("AWS_PROFILE", None)
    os.environ["AWS_ACCESS_KEY_ID"] = frozen.access_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = frozen.secret_key
    if frozen.token:
        os.environ["AWS_SESSION_TOKEN"] = frozen.token
    else:
        os.environ.pop("AWS_SESSION_TOKEN", None)
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@app.command()
def main(
    wandb_path: Annotated[
        str,
        typer.Option(
            help="W&B artifact path (e.g., 'climatepolicyradar/Q913/rsgz5ygh:v0')",
        ),
    ],
    source_env: Annotated[
        AwsEnv,
        typer.Option(
            help="Source AWS environment (for validation)",
            parser=parse_aws_env,
        ),
    ] = AwsEnv.labs,
    target_env: Annotated[
        AwsEnv,
        typer.Option(
            help="Target AWS environment to promote to",
            parser=parse_aws_env,
        ),
    ] = AwsEnv.production,
):
    """
    Load a classifier from W&B, and upload to a different environment's S3/W&B.

    This script:
    1. Loads a classifier from the specified W&B artifact path
    2. Validates the source environment matches the artifact metadata
    3. Uploads to the target environment's S3 bucket
    4. Creates a new W&B artifact with updated metadata

    :param wandb_path: W&B artifact path to load the classifier from
    :param source_env: Source AWS environment (for validation)
    :param target_env: Target AWS environment for upload
    """

    console.log(f"Validating AWS login for {target_env.value}...")
    use_aws_profiles = os.environ.get("USE_AWS_PROFILES", "true").lower() == "true"
    # Build the target session explicitly: the shared cloud helpers default
    # USE_AWS_PROFILES to false and would otherwise fall back to the default
    # credential chain (e.g. the source env's profile).
    target_session = get_session(target_env) if use_aws_profiles else boto3.Session()
    try:
        identity = target_session.client("sts").get_caller_identity()
    except (
        botocore.exceptions.ClientError,
        botocore.exceptions.NoCredentialsError,
        botocore.exceptions.SSOTokenLoadError,
    ) as e:
        raise typer.BadParameter(
            f"Not logged into {target_env.value} ({e}). "
            f"Run: aws sso login --profile {target_env.value}"
        )
    console.log(f"Using AWS account {identity['Account']} for {target_env.value}")

    console.log(f"Loading original artifact metadata from {wandb_path}...")
    api = wandb.Api()
    original_artifact = api.artifact(wandb_path)
    original_metadata = dict(original_artifact.metadata)

    # Validate source environment
    artifact_env = original_metadata.get("aws_env")
    if artifact_env != source_env.value:
        console.log(
            f"[yellow]⚠️  Warning: Artifact metadata shows aws_env='{artifact_env}', "
            f"but you specified source_env='{source_env.value}'[/yellow]"
        )
        console.log("[yellow]Continuing anyway...[/yellow]")

    console.log("Loading classifier from W&B...")
    with aws_profile(source_env, use_aws_profiles):
        classifier = load_classifier_from_wandb(wandb_path)

    console.log(
        f"[green]✓[/green] Loaded classifier {classifier.name} (ID: {classifier.id})"
    )

    wikibase_id = classifier.concept.wikibase_id
    assert isinstance(wikibase_id, WikibaseID)

    namespace = Namespace(entity=WANDB_ENTITY, project=wikibase_id)
    model_path = ModelPath(wikibase_id=wikibase_id, classifier_id=classifier.id)

    console.log(f"Determining next version in {target_env.value}...")
    next_version = get_next_version(namespace, model_path, classifier)
    console.log(f"Next version: {next_version}")

    # Save classifier locally
    classifier_path = get_local_classifier_path(
        target_path=model_path, version=next_version
    )
    console.log(f"Saving classifier to {classifier_path}...")
    classifier_path.parent.mkdir(parents=True, exist_ok=True)
    classifier.save(classifier_path)

    # Upload to target environment S3
    console.log(f"Uploading to {target_env.value} S3...")
    s3_client = target_session.client("s3", region_name="eu-west-1")
    storage_upload = StorageUpload(
        target_path=str(model_path),
        next_version=next_version,
        aws_env=target_env,
    )
    bucket, key = upload_model_artifact(
        classifier,
        classifier_path,
        storage_upload,
        s3_client=s3_client,
    )

    # Create W&B run and artifact in target environment
    console.log("Initialising Weights & Biases run...")
    with wandb.init(entity=WANDB_ENTITY, project=wikibase_id, job_type=JOB_TYPE) as run:
        # Update metadata for target environment
        metadata = {
            **original_metadata,
            "aws_env": target_env.name,
            "source_artifact": wandb_path,
            "promoted_from_env": source_env.value,
        }

        artifact = wandb.Artifact(
            name=classifier.id,
            type="model",
            metadata=metadata,
        )
        uri = os.path.join("s3://", bucket, key)
        # W&B reads the object to checksum it, so it needs target env creds
        with aws_profile(target_env, use_aws_profiles):
            artifact.add_reference(uri=uri, checksum=True)

        artifact = run.log_artifact(artifact, aliases=[])
        artifact = artifact.wait()

        console.log(
            f"[green]✓[/green] Successfully promoted classifier from {source_env.value} to {target_env.value}"
        )
        console.log(f"[green]✓[/green] New artifact: {artifact.name}")
        console.log(f"[green]✓[/green] S3 location: s3://{bucket}/{key}")


if __name__ == "__main__":
    app()
