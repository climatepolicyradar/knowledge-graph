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
from typing import Annotated

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
    get_s3_client,
    is_logged_in,
    parse_aws_env,
)
from knowledge_graph.config import WANDB_ENTITY
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.wandb_helpers import load_classifier_from_wandb
from scripts.train import (
    StorageUpload,
    get_next_version,
    upload_model_artifact,
)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

JOB_TYPE = "promote_to_prod"

app = typer.Typer()
console = Console()


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
    if not is_logged_in(target_env, use_aws_profiles):
        raise typer.BadParameter(
            f"Not logged into {target_env.value}. "
            f"Run: aws sso login --profile {target_env.value}"
        )

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
    s3_client = get_s3_client(target_env, region_name="eu-west-1")
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
            "aws_env": target_env.value,
            "source_artifact": wandb_path,
            "promoted_from_env": source_env.value,
        }

        artifact = wandb.Artifact(
            name=classifier.id,
            type="model",
            metadata=metadata,
        )
        uri = os.path.join("s3://", bucket, key)
        artifact.add_reference(uri=uri, checksum=True)

        artifact = run.log_artifact(artifact, aliases=[])
        artifact = artifact.wait()

        console.log(
            f"[green]✓[/green] Successfully promoted classifier from {source_env.value} to {target_env.value}"
        )
        console.log(f"[green]✓[/green] New artifact: {artifact.name}:{next_version}")
        console.log(f"[green]✓[/green] S3 location: s3://{bucket}/{key}")


if __name__ == "__main__":
    app()
