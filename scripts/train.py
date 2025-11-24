import asyncio
import os
import random
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
from knowledge_graph.config import WANDB_ENTITY, wandb_model_artifact_filename
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.labelling import ArgillaConfig
from knowledge_graph.version import Version
from knowledge_graph.wandb_helpers import (
    load_labelled_passages_from_wandb,
    log_labelled_passages_artifact_to_wandb_run,
)
from knowledge_graph.wikibase import WikibaseConfig
from scripts.classifier_metadata import ComputeEnvironment
from scripts.evaluate import evaluate_classifier

app = typer.Typer()


def parse_kwargs_from_strings(key_value_strings: Optional[list[str]]) -> dict[str, Any]:
    """Parse key=value strings into dicts that can be used as kwargs."""
    if not key_value_strings:
        return {}

    kwargs = {}
    for kv in key_value_strings:
        if "=" not in kv:
            raise typer.BadParameter(
                f"Invalid format for kwarg: '{kv}'. Expected key=value format."
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


def deduplicate_training_data(
    training_data: list[LabelledPassage],
    evaluation_data: list[LabelledPassage],
) -> list[LabelledPassage]:
    """Remove passages from training data that appear in evaluation data."""
    console = Console()

    eval_texts = {passage.text for passage in evaluation_data}

    console.log(f"📊 Starting with {len(training_data)} passages for training")

    filtered = [p for p in training_data if p.text not in eval_texts]

    removed_count = len(training_data) - len(filtered)
    console.log(
        f"🔍 Removed {removed_count} duplicate passages, training with {len(filtered)} passages"
    )

    return filtered


def limit_training_samples(
    training_data: list[LabelledPassage],
    max_samples: int,
) -> list[LabelledPassage]:
    """
    Limit the number of training samples, aiming for a balanced set.

    If a perfect split isn't possible, take all available from the smaller group and
    the remainder from the larger group.

    :param training_data: The list of labelled passages to limit.
    :type training_data: list[LabelledPassage]
    :param max_samples: Maximum number of samples to keep in total.
    :type max_samples: int
    :return: A (mostly) balanced subset of the training data.
    :rtype: list[LabelledPassage]
    """
    console = Console()

    positive_passages = [p for p in training_data if p.spans]
    negative_passages = [p for p in training_data if not p.spans]

    console.log(
        f"📊 Starting with {len(positive_passages)} positive and "
        f"{len(negative_passages)} negative passages"
    )

    half = max_samples // 2
    # Take up to half from each group, or as many as you can
    pos_count = min(len(positive_passages), half)
    neg_count = min(len(negative_passages), half)

    # Fill up with remainder from the group that still has samples left
    remainder = max_samples - (pos_count + neg_count)
    if remainder > 0:
        if pos_count < len(positive_passages):
            extra = min(remainder, len(positive_passages) - pos_count)
            pos_count += extra
            remainder -= extra
        if remainder > 0 and neg_count < len(negative_passages):
            extra = min(remainder, len(negative_passages) - neg_count)
            neg_count += extra

    limited_positive = positive_passages[:pos_count]
    limited_negative = negative_passages[:neg_count]

    console.log(
        f"✂️  Limited to {len(limited_positive)} positive and "
        f"{len(limited_negative)} negative passages "
        f"({len(limited_positive) + len(limited_negative)} total)"
    )

    limited_dataset = limited_positive + limited_negative
    random.shuffle(limited_dataset)
    return limited_dataset


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
        "aws_env": storage_link.aws_env.name,
        "classifier_name": classifier.name,
        "concept_id": classifier.concept.id,
        "concept_wikibase_revision": classifier.concept.wikibase_revision,
    }
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

    # Log the artifact to W&B, creating it within a W&B project
    artifact = run.log_artifact(artifact, aliases=[])
    artifact = artifact.wait()

    return artifact


def classifier_exists_in_wandb(
    namespace: Namespace,
    target_path: ModelPath,
) -> bool:
    """
    Check whether a classifier artifact already exists in W&B.

    :param namespace: The W&B configuration containing project and entity.
    :param target_path: The path to the classifier in W&B.
    :return: True if the artifact exists, False otherwise.
    """
    try:
        api = wandb.Api()
        api.artifact(f"{namespace.entity}/{target_path}:latest")
        return True
    except CommError as e:
        error_message = str(e)
        # Check if the error is because the artifact doesn't exist
        # Error format: "artifact membership '...' not found in '{entity}/{wikibase_id}'"
        pattern = rf"artifact membership '.*?' not found in '{namespace.entity}/{target_path.wikibase_id}'"
        if re.search(pattern, error_message):
            return False
        # Re-raise if it's a different error (e.g., network issue)
        raise


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
        wandb_model_artifact_filename,
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
    ] = True,
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
    classifier_override: Annotated[
        Optional[list[str]],
        typer.Option(
            help="Classifier kwarg overrides in key=value format. Can be specified multiple times.",
        ),
    ] = None,
    concept_override: Annotated[
        Optional[list[str]],
        typer.Option(
            help="Concept property overrides in key=value format. Can be specified multiple times.",
        ),
    ] = None,
    training_data_wandb_path: Annotated[
        Optional[str],
        typer.Option(
            help="W&B artifact path (e.g., 'entity/project/artifact:version') to fetch training data from.",
        ),
    ] = None,
    limit_training_samples: Annotated[
        Optional[int],
        typer.Option(
            help="Maximum number of training samples to use. Samples are selected in a way that achieves the best possible class balance. If not specified, all samples are used.",
        ),
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
    :param classifier_override: List of classifier kwargs in key=value format
    :type classifier_override: Optional[list[str]]
    :param concept_override: List of concept property overrides in key=value format (e.g., description, labels)
    :type concept_override: Optional[list[str]]
    :param limit_training_samples: Maximum number of training samples to use
    :type limit_training_samples: Optional[int]
    """
    classifier_kwargs = parse_kwargs_from_strings(classifier_override)
    concept_overrides = parse_kwargs_from_strings(concept_override)

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
                "concept_overrides": concept_overrides,
                "training_data_wandb_path": training_data_wandb_path,
                "limit_training_samples": limit_training_samples,
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
                concept_overrides=concept_overrides,
                training_data_wandb_path=training_data_wandb_path,
                limit_training_samples=limit_training_samples,
            )
        )


async def train_classifier(
    classifier: Classifier,
    wikibase_id: WikibaseID,
    track_and_upload: bool,
    aws_env: AwsEnv,
    s3_client: Optional[Any] = None,
    evaluate: bool = True,
    extra_wandb_config: dict[str, Any] = {},
    train_validation_data: Optional[list[LabelledPassage]] = None,
    max_training_samples: Optional[int] = None,
    force: bool = False,
) -> "Classifier":
    """Train a classifier and optionally track the run, uploading the model."""
    # Create console locally to avoid serialization issues
    console = Console()

    project = wikibase_id
    namespace = Namespace(project=project, entity=WANDB_ENTITY)
    job_type = "train_model"

    # Validate parameter dependencies
    validate_params(track_and_upload=track_and_upload, aws_env=aws_env)

    # Check whether the classifier already exists in W&B, unless the user wants to force re-training
    if track_and_upload and not force:
        target_path = ModelPath(
            wikibase_id=namespace.project, classifier_id=classifier.id
        )
        if classifier_exists_in_wandb(namespace=namespace, target_path=target_path):
            # If the classifier already exists, just log and return the classifier without
            # running the redundant training/uploading process
            console.log(
                f"Classifier {classifier.id} already exists in W&B. Skipping training."
            )
            return classifier

    wandb_config = {
        "classifier_type": classifier.name,
        "concept_id": classifier.concept.id,
    }
    wandb_config |= extra_wandb_config

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
        # Determine training data and deduplicate against evaluation set
        training_data = (
            train_validation_data if train_validation_data is not None else []
        )
        if max_training_samples is not None:
            training_data = limit_training_samples(training_data, max_training_samples)

        if training_data and wandb_config.get("training_data_wandb_path") and run:
            unprocessed_training_data_artifact_path = wandb_config[
                "training_data_wandb_path"
            ]
            run.use_artifact(unprocessed_training_data_artifact_path)

        # Remove any passages from training that appear in evaluation set
        evaluation_data = classifier.concept.labelled_passages
        if training_data:
            deduplicated_training_data = deduplicate_training_data(
                training_data=training_data,
                evaluation_data=evaluation_data,
            )

            train_num_positives = len(
                [p for p in deduplicated_training_data if p.spans]
            )
            train_num_negatives = len(deduplicated_training_data) - train_num_positives
            Console().print(
                f"Training data has length {len(deduplicated_training_data)} with {train_num_positives} positive and {train_num_negatives} negative examples after deduplication."
            )

            if track_and_upload and run and deduplicated_training_data:
                console.log("📄 Creating artifact for deduplicated training data")
                log_labelled_passages_artifact_to_wandb_run(
                    labelled_passages=deduplicated_training_data,
                    run=run,
                    concept=classifier.concept,
                    classifier=classifier,
                    artifact_name="training-data",
                )
                console.log("✅ Training data artifact uploaded successfully")
        else:
            deduplicated_training_data = []

        classifier.fit(
            labelled_passages=deduplicated_training_data,
            enable_wandb=track_and_upload,
        )

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
            )

        if evaluate:
            metrics_df, model_labelled_passages = evaluate_classifier(
                classifier=classifier,
                labelled_passages=classifier.concept.labelled_passages,
                wandb_run=run,
                batch_size=50,
            )

            if track_and_upload and run:
                console.log("📄 Creating labelled passages artifact")
                log_labelled_passages_artifact_to_wandb_run(
                    labelled_passages=model_labelled_passages,
                    run=run,
                    concept=classifier.concept,
                    classifier=classifier,
                )
                console.log("✅ Labelled passages uploaded successfully")

    return classifier


async def run_training(
    wikibase_id: WikibaseID,
    track_and_upload: bool,
    aws_env: AwsEnv,
    wikibase_config: Optional[WikibaseConfig] = None,
    argilla_config: Optional[ArgillaConfig] = None,
    s3_client: Optional[Any] = None,
    evaluate: bool = True,
    classifier_type: Optional[str] = None,
    classifier_kwargs: Optional[dict[str, Any]] = None,
    concept_overrides: Optional[dict[str, Any]] = None,
    training_data_wandb_path: Optional[str] = None,
    limit_training_samples: Optional[int] = None,
    force: bool = False,
) -> Classifier:
    """
    Get a concept and create a classifier, then train the classifier.

    Optionally evaluate, track in W&B and upload the model to S3.
    """
    console = Console()

    # Validate parameter dependencies
    validate_params(track_and_upload=track_and_upload, aws_env=aws_env)

    concept = await scripts.get_concept.get_concept_async(
        wikibase_id=wikibase_id,
        include_labels_from_subconcepts=True,
        include_recursive_has_subconcept=True,
        wikibase_config=wikibase_config,
        argilla_config=argilla_config,
    )

    if concept_overrides:
        console.log(f"🔧 Applying custom concept properties: {concept_overrides}")
        for key, value in concept_overrides.items():
            if hasattr(concept, key):
                setattr(concept, key, value)
                console.log(f"  ✓ Set concept.{key} = {value}")
            else:
                console.log(
                    f"  ⚠️  Warning: concept has no attribute '{key}'", style="yellow"
                )

    # Fetch labelled passages from W&B if specified
    labelled_passages = None
    if training_data_wandb_path:
        console.log(
            f"📥 Fetching training data from W&B artifact path: {training_data_wandb_path}"
        )
        labelled_passages = load_labelled_passages_from_wandb(
            wandb_path=training_data_wandb_path
        )
        console.log(f"✅ Loaded {len(labelled_passages)} labelled passages from W&B")

    classifier = ClassifierFactory.create(
        concept=concept,
        classifier_type=classifier_type,
        classifier_kwargs=classifier_kwargs or {},
    )

    extra_wandb_config: dict[str, object] = {
        "experimental_model_type": classifier_type is not None,
        "experimental_concept": concept_overrides is not None
        and len(concept_overrides) > 0,
        "classifier_kwargs": classifier_kwargs,
        "concept_overrides": concept_overrides,
    }
    if training_data_wandb_path:
        extra_wandb_config["training_data_wandb_path"] = training_data_wandb_path
    if limit_training_samples is not None:
        extra_wandb_config["limit_training_samples"] = limit_training_samples

    return await train_classifier(
        classifier=classifier,
        wikibase_id=wikibase_id,
        track_and_upload=track_and_upload,
        aws_env=aws_env,
        s3_client=s3_client,
        evaluate=evaluate,
        extra_wandb_config=extra_wandb_config,
        train_validation_data=labelled_passages,
        max_training_samples=limit_training_samples,
        force=force,
    )


if __name__ == "__main__":
    app()
