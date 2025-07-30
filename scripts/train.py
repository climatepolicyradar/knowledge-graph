import os
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Annotated, Any

import typer
import wandb
from pydantic import BaseModel, Field
from rich.console import Console
from wandb.errors.errors import CommError
from wandb.sdk.wandb_run import Run

from scripts.cloud import AwsEnv, Namespace, get_s3_client, is_logged_in
from scripts.utils import get_local_classifier_path
from src.classifier import Classifier, ClassifierFactory
from src.identifiers import WikibaseID
from src.version import Version
from src.wikibase import WikibaseSession

console = Console()
app = typer.Typer()


def validate_params(track: bool, upload: bool, aws_env: AwsEnv) -> None:
    """Validate parameter dependencies."""
    if (not track) and upload:
        raise ValueError(
            "you can only upload a model artifact, if you're also tracking the run"
        )

    use_aws_profiles = os.environ.get("USE_AWS_PROFILES", "true").lower() == "true"
    if upload and (not is_logged_in(aws_env, use_aws_profiles)):
        raise typer.BadParameter(
            f"you're not logged into {aws_env.value}. "
            f"Do `aws sso login --profile {aws_env.value}`"
        )


class StorageUpload(BaseModel):
    """Represents the storage configuration for model artifacts in S3."""

    target_path: str = Field(
        ...,
        description="The target path in S3 where the model artifact will be stored.",
    )
    next_version: str = Field(
        ...,
        description="The next version used for this artifact.",
    )
    aws_env: AwsEnv = Field(
        ...,
        description="The AWS environment associated with this storage configuration.",
    )


class StorageLink(BaseModel):
    """Represents the storage configuration for model artifacts in S3."""

    bucket: str = Field(
        ...,
        description="The name of the S3 bucket where the model artifact is stored.",
    )
    key: str = Field(
        ...,
        description="The S3 key (path) where the model artifact is located within the bucket.",
    )
    aws_env: AwsEnv = Field(
        ...,
        description="The AWS environment associated with this storage configuration.",
    )


def create_and_link_model_artifact(
    run: Run,
    classifier: Classifier,
    storage_link: StorageLink,
) -> wandb.Artifact:  # type: ignore
    """
    Links a model artifact, stored in S3, to a Weights & Biases run.

    :param run: The W&B run object.
    :type run: Run
    :param classifier: The classifier object.
    :type classifier: Classifier
    :param storage_link: The storage location configuration.
    :type storage_link: StorageLink
    :return: The created W&B artifact.
    :rtype: wandb.Artifact
    """
    metadata = {"aws_env": storage_link.aws_env.value}

    # Set this, so W&B knows where to look for AWS credentials profile
    os.environ["AWS_PROFILE"] = storage_link.aws_env.value

    artifact = wandb.Artifact(  # type: ignore
        name=classifier.id,
        type="model",
        metadata=metadata,
    )
    uri = os.path.join(
        "s3://",
        storage_link.bucket,
        storage_link.key,
    )
    artifact.add_reference(uri=uri, checksum=True)

    # Log the artifact to W&B, creating it within a wandb project
    artifact = run.log_artifact(artifact)
    artifact = artifact.wait()

    return artifact


def get_next_version(
    namespace: Namespace,
    target_path: str,
    classifier: Classifier,
) -> str:
    """
    Retrieves the next version number for a given classifier.

    :param namespace: The W&B configuration containing project and entity.
    :type namespace: WandBConfig
    :param target_path: The path to the classifier in W&B.
    :type target_path: str
    :param classifier: The classifier object.
    :type classifier: Classifier
    :return: The next version string.
    :rtype: str
    """
    try:
        api = wandb.Api()
        artifact = api.artifact(f"{namespace.entity}/{target_path}:latest")
        next_version = Version(artifact._version).increment()  # type: ignore[reportArgumentType]
    except CommError as e:
        error_message = str(e)
        wikibase_id = classifier.concept.wikibase_id
        pattern = rf"artifact '.*?' not found in '{namespace.entity}/{wikibase_id}'"
        if re.search(pattern, error_message):
            console.log(
                f"No previous wandb version found, '{target_path}' will be at v0"
            )
            next_version = Version("v0")
        else:
            raise

    return str(next_version)


def upload_model_artifact(
    classifier: Classifier,
    classifier_path: Path,
    storage_upload: StorageUpload,
    s3_client: Any,
) -> tuple[str, str]:
    """
    Uploads a model artifact to S3.

    :param classifier: The classifier object.
    :type classifier: Classifier
    :param classifier_path: The local path to the classifier file.
    :type classifier_path: Path
    :param storage_upload: The configuration for uploading the artifact.
    :type storage_upload: StorageUpload
    :param s3_client: The S3 client used for uploading.
    :type s3_client: Any
    :return: The bucket name and the key of the uploaded artifact.
    :rtype: tuple[str, str]
    """
    bucket = f"cpr-{storage_upload.aws_env.value}-models"

    key = os.path.join(
        storage_upload.target_path,
        f"{storage_upload.next_version}.pickle",
    )

    console.log(f"Uploading {classifier.name} to {key} in bucket {bucket}")

    s3_client.upload_file(
        classifier_path,
        bucket,
        key,
        ExtraArgs={"ContentType": "application/octet-stream"},
        Callback=lambda bytes_transferred: None,
    )

    console.log(f"Uploaded {classifier.name} to {key} in bucket {bucket}")

    return bucket, key


@app.command()
def main(
    wikibase_id: Annotated[
        WikibaseID,
        typer.Option(
            ...,
            help="The Wikibase ID of the concept classifier to train",
            parser=WikibaseID,
        ),
    ],
    track: Annotated[
        bool,
        typer.Option(
            ...,
            help="Whether to track the training run with Weights & Biases",
        ),
    ] = False,
    upload: Annotated[
        bool,
        typer.Option(
            ...,
            help="Whether to upload the model artifact to S3",
        ),
    ] = False,
    aws_env: Annotated[
        AwsEnv,
        typer.Option(
            ...,
            help="AWS environment to use for S3 uploads",
        ),
    ] = AwsEnv.labs,
) -> Classifier:
    """
    Main function to train the model and optionally upload the artifact.

    :param wikibase_id: The Wikibase ID of the concept classifier to train.
    :type wikibase_id: WikibaseID
    :param track: Whether to track the training run with W&B.
    :type track: bool
    :param upload: Whether to upload the model artifact to S3.
    :type upload: bool
    :param aws_env: The AWS environment to use for S3 uploads.
    :type aws_env: AwsEnv
    """
    entity = "climatepolicyradar"
    project = wikibase_id
    namespace = Namespace(project=project, entity=entity)
    job_type = "train_model"

    # Validate parameter dependencies
    validate_params(track, upload, aws_env)

    with (
        wandb.init(
            entity=namespace.entity, project=namespace.project, job_type=job_type
        )
        if track
        else nullcontext()
    ) as run:
        # Fetch all of its subconcepts recursively
        wikibase = WikibaseSession()
        concept = wikibase.get_concept(
            wikibase_id,
            include_recursive_has_subconcept=True,
            include_labels_from_subconcepts=True,
        )

        # Create and train a classifier instance
        classifier = ClassifierFactory.create(concept=concept)
        classifier.fit()

        target_path = f"{namespace.project}/{classifier.id}"  # e.g. 'Q123/v4prnc54'

        # Lookup the next version (aka the new version) before saving, even if we're
        # not uploading or tracking, so the classifier has the correct version
        # Note that as we use the id and the id changes whenever the model changes,
        # the version would almost always be v0 in practice.
        next_version = get_next_version(
            namespace=namespace,
            target_path=target_path,
            classifier=classifier,
        )

        console.log(f"Using next version {next_version}")

        # Set this _before_ the model is saved to disk
        classifier.version = Version(next_version)

        # Save the classifier to a file locally
        classifier_path = get_local_classifier_path(
            target_path=target_path,
            next_version=next_version,
        )
        classifier_path.parent.mkdir(parents=True, exist_ok=True)
        classifier.save(classifier_path)
        console.log(f"Saved {classifier} to {classifier_path}")

        if upload:
            region_name = "eu-west-1"
            s3_client = get_s3_client(aws_env, region_name)

            storage_upload = StorageUpload(
                target_path=target_path,
                next_version=next_version,
                aws_env=aws_env,
            )

            bucket, key = upload_model_artifact(
                classifier,
                classifier_path,
                storage_upload,
                s3_client=s3_client,
            )

            storage_link = StorageLink(
                bucket=bucket,
                key=key,
                aws_env=aws_env,
            )

            create_and_link_model_artifact(
                run,  # type: ignore
                classifier,
                storage_link,
            )

    return classifier


if __name__ == "__main__":
    app()
