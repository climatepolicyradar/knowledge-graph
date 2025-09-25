import asyncio
import os
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Annotated, Any, Optional

import typer
import wandb
from prefect.client.schemas.objects import FlowRun
from prefect.deployments import run_deployment
from pydantic import BaseModel, Field
from rich.console import Console
from wandb.errors.errors import CommError
from wandb.sdk.wandb_run import Run

import scripts.get_concept
from flows.utils import get_flow_run_ui_url
from knowledge_graph.classifier import (
    Classifier,
    ClassifierFactory,
    GPUBoundClassifier,
    ModelPath,
    get_local_classifier_path,
)
from knowledge_graph.cloud import (
    AwsEnv,
    Namespace,
    generate_deployment_name,
    get_s3_client,
    is_logged_in,
)
from knowledge_graph.config import WANDB_ENTITY
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.version import Version
from knowledge_graph.wikibase import WikibaseConfig
from scripts.classifier_metadata import ComputeEnvironment
from scripts.evaluate import evaluate_classifier

app = typer.Typer()


def parse_classifier_kwargs(classifier_kwarg: Optional[list[str]]) -> dict[str, Any]:
    """Parse classifier kwargs from key=value strings."""
    if not classifier_kwarg:
        return {}

    kwargs = {}
    for kv in classifier_kwarg:
        if "=" not in kv:
            raise typer.BadParameter(
                f"Invalid format for classifier kwarg: '{kv}'. Expected key=value format."
            )

        key, value = kv.split("=", 1)

        # Try to parse as int, then bool, then string
        try:
            kwargs[key] = int(value)
        except ValueError:
            if value.lower() in ("true", "false"):
                kwargs[key] = value.lower() == "true"
            else:
                kwargs[key] = value

    return kwargs


def validate_params(track_and_upload: bool, aws_env: AwsEnv) -> None:
    """Validate parameter dependencies."""

    use_aws_profiles = os.environ.get("USE_AWS_PROFILES", "true").lower() == "true"
    if track_and_upload and (not is_logged_in(aws_env, use_aws_profiles)):
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
    add_classifiers_profiles: list[str] | None = None,
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

    metadata: dict[str, Any] = {
        "aws_env": storage_link.aws_env.value,
        "classifier_name": classifier.name,
        "concept_id": classifier.concept.id,
        "concept_wikibase_revision": classifier.concept.wikibase_revision,
    }
    if add_classifiers_profiles:
        metadata["classifiers_profiles"] = list(add_classifiers_profiles)
    if isinstance(classifier, GPUBoundClassifier):
        Console().log("Adding GPU requirement to metadata")
        compute_environment: ComputeEnvironment = {"gpu": True}
        metadata["compute_environment"] = compute_environment

    artifact = wandb.Artifact(
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
    artifact = run.log_artifact(artifact, aliases=[storage_link.aws_env.value])
    artifact = artifact.wait()

    return artifact


def get_next_version(
    namespace: Namespace,
    target_path: ModelPath,
    classifier: Classifier,
) -> str:
    """
    Retrieves the next version number for a given classifier.

    :param namespace: The W&B configuration containing project and entity.
    :type namespace: WandBConfig
    :param target_path: The path to the classifier in W&B.
    :type target_path: ModelPath
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
        pattern = rf"artifact membership '.*?' not found in '{namespace.entity}/{wikibase_id}'"
        if re.search(pattern, error_message):
            Console().log(
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
        storage_upload.next_version,
        "model.pickle",
    )

    Console().log(f"Uploading {classifier.name} to {key} in bucket {bucket}")

    s3_client.upload_file(
        classifier_path,
        bucket,
        key,
        ExtraArgs={"ContentType": "application/octet-stream"},
        Callback=lambda bytes_transferred: None,
    )

    Console().log(f"Uploaded {classifier.name} to {key} in bucket {bucket}")

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
    track_and_upload: Annotated[
        bool,
        typer.Option(
            ...,
            help="Whether to track the training run with Weights & Biases. Includes uploading the model artifact to S3.",
        ),
    ] = False,
    aws_env: Annotated[
        AwsEnv,
        typer.Option(
            ...,
            help="AWS environment to use for S3 uploads",
        ),
    ] = AwsEnv.labs,
    use_coiled_gpu: Annotated[
        bool,
        typer.Option(
            ...,
            help=(
                "Run on Coiled with a GPU. This uses prefect to start a Coiled cluster. "
                "Note, that the classifier won't be available locally after training."
            ),
        ),
    ] = False,
    evaluate: Annotated[
        bool,
        typer.Option(
            ...,
            help="Whether to evaluate the model after training",
        ),
    ] = True,
    classifier_type: Annotated[
        Optional[str],
        typer.Option(
            help="Classifier type to use (e.g., LLMClassifier, KeywordClassifier). If not specified, uses ClassifierFactory default.",
        ),
    ] = None,
    classifier_kwarg: Annotated[
        Optional[list[str]],
        typer.Option(
            help="Classifier kwargs in key=value format. Can be specified multiple times.",
        ),
    ] = None,
    add_classifiers_profiles: Annotated[
        list[str] | None,
        typer.Option(help="Adds 1 or more classifiers profiles."),
    ] = None,
) -> Classifier | None:
    """
    Main function to train the model and optionally upload the artifact.

    :param wikibase_id: The Wikibase ID of the concept classifier to train.
    :type wikibase_id: WikibaseID
    :param track_and_upload: Whether to track the training run with Weights & Biases. Includes uploading the model artifact to S3.
    :type track_and_upload: bool
    :param aws_env: The AWS environment to use for S3 uploads.
    :type aws_env: AwsEnv
    :param use_coiled_gpu: Whether to run training remotely using a coiled gpu
    :type use_coiled_gpu: bool
    :param evaluate: Whether to evaluate the model after training
    :type evaluate: bool
    :param classifier_type: The classifier type to use, optional. Defaults to the
    classifier chosen by ClassifierFactory otherwise
    :type classifier_type: Optional[str]
    :param classifier_kwarg: List of classifier kwargs in key=value format
    :type classifier_kwarg: Optional[list[str]]
    """
    classifier_kwargs = parse_classifier_kwargs(classifier_kwarg)

    if use_coiled_gpu:
        flow_name = "train-on-gpu"
        deployment_name = generate_deployment_name(flow_name=flow_name, aws_env=aws_env)
        qualified_name = f"{flow_name}/{deployment_name}"

        flow_run: FlowRun = run_deployment(  # type: ignore[misc]
            name=qualified_name,
            parameters={
                "wikibase_id": wikibase_id,
                "track_and_upload": track_and_upload,
                "aws_env": aws_env,
                "evaluate": evaluate,
                "classifier_type": classifier_type,
                "classifier_kwargs": classifier_kwargs,
                "add_classifiers_profiles": add_classifiers_profiles,
            },
            timeout=0,  # Don't wait for the flow to finish before continuing
        )
        Console().print(
            f"Deployment started. [blue][link={get_flow_run_ui_url(flow_run)}]Click here to open flow run on Prefect.[/link][/blue]"
        )

        return None  # Can't return the classifier when running remotely
    else:
        return asyncio.run(
            run_training(
                wikibase_id=wikibase_id,
                track_and_upload=track_and_upload,
                aws_env=aws_env,
                evaluate=evaluate,
                classifier_type=classifier_type,
                classifier_kwargs=classifier_kwargs,
                add_classifiers_profiles=add_classifiers_profiles,
            )
        )


async def run_training(
    wikibase_id: WikibaseID,
    track_and_upload: bool,
    aws_env: AwsEnv,
    wikibase_config: Optional[WikibaseConfig] = None,
    s3_client: Optional[Any] = None,
    evaluate: bool = True,
    classifier_type: Optional[str] = None,
    classifier_kwargs: Optional[dict[str, Any]] = None,
    add_classifiers_profiles: list[str] | None = None,
) -> Classifier:
    """Train the model and optionally track the run, uploading the model."""
    # Create console locally to avoid serialization issues
    console = Console()

    project = wikibase_id
    namespace = Namespace(project=project, entity=WANDB_ENTITY)
    job_type = "train_model"

    # Validate parameter dependencies
    validate_params(track_and_upload=track_and_upload, aws_env=aws_env)

    concept = await scripts.get_concept.get_concept_async(
        wikibase_id=wikibase_id,
        include_labels_from_subconcepts=True,
        include_recursive_has_subconcept=True,
        wikibase_config=wikibase_config,
    )

    classifier = ClassifierFactory.create(
        concept=concept,
        classifier_type=classifier_type,
        classifier_kwargs=classifier_kwargs or {},
    )

    wandb_config = {
        "classifier_type": classifier.name,
        "classifier_kwargs": classifier_kwargs,
        "experimental-model-type": classifier_type is not None,
        "concept_hash": concept.__hash__(),
    }

    with (
        wandb.init(
            entity=namespace.entity,
            project=namespace.project,
            job_type=job_type,
            config=wandb_config,
        )
        if track_and_upload
        else nullcontext()
    ) as run:
        classifier.fit()
        target_path = ModelPath(
            wikibase_id=namespace.project, classifier_id=classifier.id
        )
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
            version=next_version,
        )
        classifier_path.parent.mkdir(parents=True, exist_ok=True)
        classifier.save(classifier_path)
        console.log(f"Saved {classifier} to {classifier_path}")

        if track_and_upload:
            region_name = "eu-west-1"
            # When running in prefect the client is instantiated earlier
            if not s3_client:
                # Set this, so W&B knows where to look for AWS credentials profile
                os.environ["AWS_PROFILE"] = aws_env
                s3_client = get_s3_client(aws_env, region_name)

            storage_upload = StorageUpload(
                target_path=str(target_path),
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

            _ = create_and_link_model_artifact(
                run,  # type: ignore
                classifier,
                storage_link,
                add_classifiers_profiles,
            )

        if evaluate:
            metrics_df, model_labelled_passages = evaluate_classifier(
                classifier=classifier,
                labelled_passages=concept.labelled_passages,
                wandb_run=run,
            )

            if track_and_upload and run:
                console.log("📄 Creating labelled passages artifact")
                labelled_passages_artifact = wandb.Artifact(
                    name=f"{classifier.id}-labelled-passages",
                    type="labelled_passages",
                    metadata={
                        "classifier_id": classifier.id,
                        "concept_wikibase_revision": classifier.concept.wikibase_revision,
                        "passage_count": len(model_labelled_passages),
                    },
                )

                with labelled_passages_artifact.new_file(
                    "labelled_passages.json", mode="w"
                ) as f:
                    data = "\n".join(
                        [entry.model_dump_json() for entry in model_labelled_passages]
                    )
                    f.write(data)

                console.log("📤 Uploading labelled passages to W&B")
                run.log_artifact(labelled_passages_artifact)
                console.log("✅ Labelled passages uploaded successfully")

    return classifier


if __name__ == "__main__":
    app()
