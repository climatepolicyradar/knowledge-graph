"""
A script for modifying the threshold of a classifier from W&B.

Uploads a classifier artifact with the chosen threshold to a new W&B run, and saves the
classifier locally. This is a workaround for the fact that we can't set the threshold
at inference time at the moment.
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
from knowledge_graph.classifier.classifier import ProbabilityCapableClassifier
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

JOB_TYPE = "modify_threshold"

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
    threshold: Annotated[
        float,
        typer.Option(
            help="Prediction threshold to set for the classifier",
        ),
    ],
    aws_env: Annotated[
        AwsEnv,
        typer.Option(
            help="AWS environment to upload the model to",
            parser=parse_aws_env,
        ),
    ] = AwsEnv.production,
):
    """
    Load a classifier from W&B, set a new prediction threshold, and upload to S3/W&B.

    This script:
    1. Loads a classifier from the specified W&B artifact path
    2. Checks if it's a ProbabilityCapableClassifier
    3. Sets the specified threshold
    4. Uploads to S3 and logs to W&B following the standard training workflow

    :param wandb_path: W&B artifact path to load the classifier from
    :param threshold: The new prediction threshold to set
    :param aws_env: AWS environment for S3 upload
    """

    console.log("Validating AWS login...")
    use_aws_profiles = os.environ.get("USE_AWS_PROFILES", "true").lower() == "true"
    if not is_logged_in(aws_env, use_aws_profiles):
        raise typer.BadParameter(
            f"Not logged into {aws_env.value}. "
            f"Run: aws sso login --profile {aws_env.value}"
        )

    console.log(f"Loading original artifact metadata from {wandb_path}...")
    api = wandb.Api()
    original_artifact = api.artifact(wandb_path)
    original_metadata = dict(original_artifact.metadata)

    console.log("Loading classifier from W&B...")
    classifier = load_classifier_from_wandb(wandb_path)

    if not isinstance(classifier, ProbabilityCapableClassifier):
        console.log(
            f"[yellow]⚠️  Classifier {classifier.name} is not a ProbabilityCapableClassifier.[/yellow]"
        )
        console.log("[yellow]Exiting without making changes.[/yellow]")
        return

    console.log(
        f"[green]✓[/green] Classifier {classifier.name} is a ProbabilityCapableClassifier"
    )
    console.log(f"Setting threshold to {threshold}...")

    # Set the prediction threshold
    classifier.set_prediction_threshold(threshold)

    wikibase_id = classifier.concept.wikibase_id
    assert isinstance(wikibase_id, WikibaseID)

    namespace = Namespace(entity=WANDB_ENTITY, project=wikibase_id)
    model_path = ModelPath(wikibase_id=wikibase_id, classifier_id=classifier.id)

    console.log("Determining next version...")
    next_version = get_next_version(namespace, model_path, classifier)
    console.log(f"Next version: {next_version}")

    # Save classifier locally
    classifier_path = get_local_classifier_path(
        target_path=model_path, version=next_version
    )
    console.log(f"Saving classifier to {classifier_path}...")
    classifier_path.parent.mkdir(parents=True, exist_ok=True)
    classifier.save(classifier_path)

    # Upload to S3
    console.log("Uploading to S3...")
    s3_client = get_s3_client(aws_env, region_name="eu-west-1")
    storage_upload = StorageUpload(
        target_path=str(model_path),
        next_version=next_version,
        aws_env=aws_env,
    )
    bucket, key = upload_model_artifact(
        classifier,
        classifier_path,
        storage_upload,
        s3_client=s3_client,
    )

    # Create W&B run and artifact
    console.log("Initialising Weights & Biases run...")
    with wandb.init(entity=WANDB_ENTITY, project=wikibase_id, job_type=JOB_TYPE) as run:
        metadata = {
            **original_metadata,
            "prediction_threshold": threshold,
            "source_artifact": wandb_path,
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

    console.log(f"Successfully modified classifier threshold to {threshold}")


if __name__ == "__main__":
    app()
