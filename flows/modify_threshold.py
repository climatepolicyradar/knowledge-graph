import os
from typing import Annotated

import wandb
from prefect import flow, task
from prefect.logging import get_run_logger
from pydantic import Field, SecretStr

from flows.config import Config
from knowledge_graph.classifier import ModelPath, get_local_classifier_path
from knowledge_graph.classifier.classifier import (
    Classifier,
    ProbabilityCapableClassifier,
)
from knowledge_graph.cloud import AwsEnv, Namespace, get_s3_client
from knowledge_graph.config import WANDB_ENTITY
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.wandb_helpers import load_classifier_from_wandb
from scripts.train import (
    StorageUpload,
    get_next_version,
    move_model_to_cpu,
    upload_model_artifact,
)

JOB_TYPE = "modify_threshold"


@task
def login_to_wandb(wandb_api_key: SecretStr) -> None:
    logger = get_run_logger()
    logger.info("Logging in to W&B")
    try:
        wandb.login(key=wandb_api_key.get_secret_value())
    except Exception as e:
        logger.error(f"W&B login failed: {e}")
        raise
    logger.info("W&B login successful")


@task
async def load_classifier_task(
    wandb_path: str,
) -> tuple[Classifier, dict]:
    logger = get_run_logger()
    logger.info(f"Loading original artifact metadata from {wandb_path}...")
    try:
        api = wandb.Api()
        original_artifact = api.artifact(wandb_path)
        original_metadata = dict(original_artifact.metadata)
    except Exception as e:
        logger.error(f"Failed to load artifact metadata from {wandb_path}: {e}")
        raise

    logger.info("Loading classifier from W&B...")
    try:
        classifier = load_classifier_from_wandb(wandb_path)
    except Exception as e:
        logger.error(f"Failed to load classifier from {wandb_path}: {e}")
        raise

    if not isinstance(classifier, ProbabilityCapableClassifier):
        raise ValueError(
            f"Classifier {classifier.name} is not a ProbabilityCapableClassifier, cannot modify threshold"
        )

    logger.info(f"Loaded classifier {classifier.name}")
    return classifier, original_metadata


@task
def upload_modified_classifier_task(
    classifier: Classifier,
    original_metadata: dict,
    threshold: float,
    source_wandb_path: str,
    aws_env: AwsEnv,
) -> str:
    logger = get_run_logger()
    logger.info(f"Setting threshold to {threshold}...")
    classifier.set_prediction_threshold(threshold)

    wikibase_id = classifier.concept.wikibase_id
    assert isinstance(wikibase_id, WikibaseID)

    namespace = Namespace(entity=WANDB_ENTITY, project=wikibase_id)
    model_path = ModelPath(wikibase_id=wikibase_id, classifier_id=classifier.id)

    logger.info("Determining next version...")
    next_version = get_next_version(namespace, model_path, classifier)
    logger.info(f"Next version: {next_version}")

    classifier_path = get_local_classifier_path(
        target_path=model_path, version=next_version
    )
    logger.info(f"Saving classifier to {classifier_path}...")
    classifier_path.parent.mkdir(parents=True, exist_ok=True)

    move_model_to_cpu(classifier)
    classifier.save(classifier_path)

    logger.info("Uploading to S3...")
    try:
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
    except Exception as e:
        logger.error(f"S3 upload failed: {e}")
        raise

    logged_artifact = None
    logger.info("Initialising Weights & Biases run...")
    with wandb.init(entity=WANDB_ENTITY, project=wikibase_id, job_type=JOB_TYPE) as run:
        artifact = wandb.Artifact(
            name=classifier.id,
            type="model",
            metadata={
                **original_metadata,
                "prediction_threshold": threshold,
                "source_artifact": source_wandb_path,
            },
        )
        uri = os.path.join("s3://", bucket, key)
        artifact.add_reference(uri=uri, checksum=True)
        logged_artifact = run.log_artifact(artifact, aliases=[])

    if logged_artifact is not None:
        logged_artifact.wait()

    logger.info(f"Successfully modified classifier threshold to {threshold}")
    logger.info(f"New classifier ID: {classifier.id}")
    return f"{WANDB_ENTITY}/{wikibase_id}/{classifier.id}:{next_version}"


@flow
async def modify_threshold(
    wandb_path: Annotated[
        str,
        Field(
            description="W&B artifact path (e.g., 'climatepolicyradar/Q913/rsgz5ygh:v0')"
        ),
    ],
    threshold: Annotated[
        float,
        Field(description="Prediction threshold to set for the classifier"),
    ],
    aws_env: AwsEnv = AwsEnv.production,
    config: Config | None = None,
) -> str:
    """
    Load a classifier from W&B, set a new prediction threshold, and upload to S3/W&B.

    The classifier with the new threshold will have a different id from the original,
    so this process is nondestructive.

    Returns the W&B artifact path for the new classifier version.
    """
    if not config:
        config = await Config.create()

    if config.wandb_api_key:
        login_to_wandb(wandb_api_key=config.wandb_api_key)

    classifier, original_metadata = await load_classifier_task(wandb_path=wandb_path)
    return upload_modified_classifier_task(
        classifier=classifier,
        original_metadata=original_metadata,
        threshold=threshold,
        source_wandb_path=wandb_path,
        aws_env=aws_env,
    )
