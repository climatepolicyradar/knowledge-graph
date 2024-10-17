import os
import re
from typing import Annotated

import botocore
import botocore.client
import typer
import wandb
from pydantic import BaseModel, Field
from rich.console import Console
from wandb.sdk.wandb_run import Run

from scripts.config import classifier_dir, concept_dir
from scripts.platform import AwsEnv, get_s3_client
from src.classifier import Classifier, ClassifierFactory
from src.concept import Concept
from src.identifiers import WikibaseID
from src.wikibase import WikibaseSession

console = Console()
app = typer.Typer()


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


class StorageUpload(BaseModel):
    """Represents the storage configuration for model artifacts in S3."""

    class Config:
        """Internal config for the model."""

        arbitrary_types_allowed = True  # For the s3_client

    s3_client: botocore.client.BaseClient = Field(
        ...,
        description="The S3 client used for uploading.",
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
    :param bucket: The storage location configuration.
    :type bucket: Storage
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
    # Don't checksum files since that means that W&B will try
    # and be too smart and will think a model artifact file in
    # a different AWS environment is the same, I think.
    artifact.add_reference(uri=uri, checksum=False)

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
        version = int(latest[1:]) + 1
    except wandb.errors.CommError as e:
        error_message = str(e)
        pattern = rf"artifact '{classifier.name}:latest' not found in '{namespace.entity}/{wikibase_id}'"

        if not re.search(pattern, error_message):
            raise

    return f"v{version}"


def upload_model_artifact(
    classifier: Classifier,
    classifier_path: str,
    storage_upload: StorageUpload,
    namespace: Namespace,
) -> tuple[str, str]:
    """
    Uploads a model artifact to S3.

    :param classifier: The classifier object.
    :type classifier: Classifier
    :param classifier_path: The path to the classifier file.
    :type classifier_path: str
    :param storage_upload: The configuration for uploading the artifact..
    :type storage_upload: StorageUpload
    :param namespace: The W&B configuration containing project and entity.
    :type namespace: WandBConfig
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

    storage_upload.s3_client.upload_file(classifier_path, bucket, key)

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
):
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

    if track:
        run = wandb.init(
            entity=namespace.entity, project=namespace.project, job_type=job_type
        )

    classifier_dir.mkdir(parents=True, exist_ok=True)

    console.log(f"Loading concept {wikibase_id} from {concept_dir}")
    try:
        concept = Concept.load(concept_dir / f"{wikibase_id}.json")
    except FileNotFoundError as e:
        raise typer.BadParameter(
            f"Data for {wikibase_id} not found. \n"
            "If you haven't already, you should run:\n"
            f"  just get-concept {wikibase_id}\n"
        ) from e

    wikibase = WikibaseSession()

    # Fetch all of its subconcepts recursively
    subconcepts = wikibase.get_subconcepts(wikibase_id, recursive=True)

    # fetch all of the labels and negative_labels for all of the subconcepts
    # and the concept itself
    all_positive_labels = set(concept.all_labels)
    all_negative_labels = set(concept.negative_labels)
    for subconcept in subconcepts:
        all_positive_labels.update(subconcept.all_labels)
        all_negative_labels.update(subconcept.negative_labels)

    concept.alternative_labels = list(all_positive_labels)
    concept.negative_labels = list(all_negative_labels)

    # Create a classifier instance
    classifier = ClassifierFactory.create(concept=concept)

    # until we have more sophisticated classifier implementations in
    # the factory, this is effectively a no-op
    classifier.fit()

    # Save the classifier to a file with the concept ID in the name
    classifier_path = classifier_dir / wikibase_id
    classifier.save(classifier_path)
    console.log(f"Saved {classifier} to {classifier_path}")

    if upload:
        region_name = "eu-west-1"
        s3_client = get_s3_client(aws_env, region_name)

        next_version = get_next_version(
            namespace,
            wikibase_id,
            classifier,
        )

        storage_upload = StorageUpload(
            s3_client=s3_client,
            next_version=next_version,
            aws_env=aws_env,
        )

        bucket, key = upload_model_artifact(
            classifier,
            classifier_path,
            storage_upload,
            namespace,
        )

    if track:
        if upload:
            storage_link = StorageLink(
                bucket=bucket,
                key=key,
                aws_env=aws_env,
            )

            link_model_artifact(
                run,
                classifier,
                storage_link,
            )

        run.finish()


if __name__ == "__main__":
    app()
