import os
import re
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import BaseModel, Field
from rich.console import Console

import wandb
from scripts.cloud import AwsEnv, Namespace, get_s3_client, is_logged_in
from scripts.config import classifier_dir
from scripts.utils import get_local_classifier_path
from src.classifier import Classifier, ClassifierFactory
from src.identifiers import WikibaseID
from src.version import Version
from src.wikibase import WikibaseSession
from wandb.sdk.wandb_run import Run

console = Console()
app = typer.Typer()


class StorageUpload(BaseModel):
    """Represents the storage configuration for model artifacts in S3."""

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


def link_model_artifact(
    run: Run,
    classifier: Classifier,
    storage_link: StorageLink,
) -> wandb.Artifact:
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

    artifact = wandb.Artifact(
        name=classifier.name,
        type="model",
        metadata=metadata,
    )
    uri = os.path.join(
        "s3://",
        storage_link.bucket,
        storage_link.key,
    )
    artifact.add_reference(uri=uri, checksum=True)

    artifact = run.log_artifact(artifact)
    artifact = artifact.wait()

    return artifact


def get_next_version(
    namespace: Namespace,
    wikibase_id: WikibaseID,
    classifier: Classifier,
) -> str:
    """
    Retrieves the next version number for a given classifier.

    :param namespace: The W&B configuration containing project and entity.
    :type namespace: WandBConfig
    :param wikibase_id: The Wikibase ID.
    :type wikibase_id: WikibaseID
    :param classifier: The classifier object.
    :type classifier: Classifier
    :return: The next version string.
    :rtype: str
    """
    version = 0  # Default value

    try:
        api = wandb.Api()
        latest = api.artifact(f"{namespace.project}/{classifier.name}:latest")._version
        version = int(latest[1:]) + 1  # type: ignore
    except wandb.errors.CommError as e:  # type: ignore
        error_message = str(e)
        pattern = rf"artifact '{classifier.name}:latest' not found in '{namespace.entity}/{wikibase_id}'"

        if not re.search(pattern, error_message):
            raise

    return f"v{version}"


def upload_model_artifact(
    classifier: Classifier,
    classifier_path: Path,
    storage_upload: StorageUpload,
    namespace: Namespace,
    s3_client: Any,
) -> tuple[str, str]:
    """
    Uploads a model artifact to S3.

    :param classifier: The classifier object.
    :type classifier: Classifier
    :param classifier_path: The path to the classifier file.
    :type classifier_path: Path
    :param storage_upload: The configuration for uploading the artifact.
    :type storage_upload: StorageUpload
    :param namespace: The W&B configuration containing project and entity.
    :type namespace: Namespace
    :param s3_client: The S3 client used for uploading.
    :type s3_client: Any
    :return: The bucket name and the key of the uploaded artifact.
    :rtype: tuple[str, str]
    """
    bucket = f"cpr-{storage_upload.aws_env.value}-models"

    key = os.path.join(
        namespace.project,
        classifier.name,
        storage_upload.next_version,
        "model.pickle",
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

    if track:
        run = wandb.init(
            entity=namespace.entity, project=namespace.project, job_type=job_type
        )

    classifier_dir.mkdir(parents=True, exist_ok=True)

    wikibase = WikibaseSession()

    # Fetch all of its subconcepts recursively
    concept = wikibase.get_concept(
        wikibase_id,
        include_recursive_subconcept_of=True,
        include_labels_from_subconcepts=True,
    )

    # Create a classifier instance
    classifier = ClassifierFactory.create(concept=concept)

    # until we have more sophisticated classifier implementations in
    # the factory, this is effectively a no-op
    classifier.fit()

    # In both scenarios, we need the next version aka the new version
    if track or upload:
        next_version = get_next_version(
            namespace,
            wikibase_id,
            classifier,
        )

        console.log(f"Using next version {next_version}")

        # Set this _before_ the model is saved to disk
        classifier.version = Version(next_version)

    # Save the classifier to a file with the concept ID in the name
    classifier_path = get_local_classifier_path(concept, classifier)
    classifier_path.parent.mkdir(parents=True, exist_ok=True)
    classifier.save(classifier_path)
    console.log(f"Saved {classifier} to {classifier_path}")

    if upload:
        region_name = "eu-west-1"
        s3_client = get_s3_client(aws_env, region_name)

        storage_upload = StorageUpload(
            next_version=next_version,  # type: ignore
            aws_env=aws_env,
        )

        bucket, key = upload_model_artifact(
            classifier,
            classifier_path,
            storage_upload,
            namespace,
            s3_client=s3_client,
        )

    if track:
        if upload:
            storage_link = StorageLink(
                bucket=bucket,  # type: ignore
                key=key,  # type: ignore
                aws_env=aws_env,
            )

            link_model_artifact(
                run,  # type: ignore
                classifier,
                storage_link,
            )

        run.finish()  # type: ignore

    return classifier


if __name__ == "__main__":
    app()
