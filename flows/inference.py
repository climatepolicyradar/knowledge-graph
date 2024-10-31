import asyncio
import json
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Optional

import boto3
import wandb
from cpr_sdk.parser_models import BaseParserOutput
from prefect import flow, task
from prefect.blocks.system import JSON
from prefect.task_runners import ConcurrentTaskRunner

from src.classifier import Classifier
from src.labelled_passage import LabelledPassage
from src.span import Span


async def get_prefect_job_variable(param_name: str) -> str:
    """Get a single variable from the Prefect job variables."""
    aws_env = os.environ["AWS_ENV"]
    block_name = f"default-job-variables-prefect-mvp-{aws_env}"
    workpool_default_job_variables = await JSON.load(block_name)
    return workpool_default_job_variables.value[param_name]


def get_aws_ssm_param(param_name: str) -> str:
    """Retrieve a parameter from AWS SSM."""
    ssm = boto3.client("ssm")
    response = ssm.get_parameter(Name=param_name, WithDecryption=True)
    return response["Parameter"]["Value"]


@dataclass()
class Config:
    """Configuration used across flow runs."""

    cache_bucket: Optional[str] = None
    document_source_prefix: str = "embeddings_input"
    document_target_prefix: str = "labelled_passages"
    bucket_region: str = "eu-west-1"
    local_classifier_dir: Path = Path("data") / "processed" / "classifiers"
    wandb_model_registry: str = "climatepolicyradar_UZODYJSN66HCQ/wandb-registry-model/"  # noqa: E501

    @classmethod
    async def create(cls) -> "Config":
        """Create a new Config instance with initialized values."""
        config = cls()
        if not config.cache_bucket:
            config.cache_bucket = await get_prefect_job_variable(
                "pipeline_cache_bucket_name"
            )
        return config


def list_bucket_doc_ids(config: Config) -> list[str]:
    """Scan configured bucket and return all IDs."""
    s3 = boto3.client("s3", region_name=config.bucket_region)
    paginator = s3.get_paginator("list_objects_v2")
    page_iterator = paginator.paginate(
        Bucket=config.cache_bucket,
        Prefix=config.document_source_prefix,
    )
    doc_ids = []

    for p in page_iterator:
        if "Contents" in p:
            for o in p["Contents"]:
                doc_id = Path(o["Key"]).stem
                doc_ids.append(doc_id)
    return doc_ids


def determine_document_ids(
    requested_document_ids: Optional[list[str]], current_bucket_ids: list[str]
) -> list[str]:
    """
    Confirm chosen document ids or default to all if not specified.

    Compares the requested_document_ids to what actually exists in the bucket.
    If a document id has been requested but does not exist this will
    raise a ValueError If no document id were requested, this will
    instead return the current_bucket_ids.
    """
    if requested_document_ids is None:
        return current_bucket_ids

    missing_from_bucket = list(set(requested_document_ids) - set(current_bucket_ids))
    if len(missing_from_bucket) > 0:
        raise ValueError(
            f"Requested document_ids not found in bucket: {missing_from_bucket}"
        )

    return requested_document_ids


def download_classifier_from_wandb_to_local(
    config: Config, classifier_name: str, alias: str
) -> str:
    """
    Download a classifier from W&B to local.

    Models referenced by weights and biases are stored in s3. This
    means that to download the model via the W&B API, we need access
    to both the s3 bucket via iam in your environment and WanDB via
    the api key.
    """
    wandb.login(key=get_aws_ssm_param("WANDB_API_KEY"))
    run = wandb.init()
    artifact = config.wandb_model_registry + f"{classifier_name}:{alias or 'latest'}"
    print(f"Downloading artifact from W&B: {artifact}")
    artifact = run.use_artifact(artifact, type="model")
    return artifact.download()


@task(log_prints=True)
async def load_classifier(
    config: Config, classifier_name: str, alias: str
) -> Classifier:
    """
    Load a classifier into memory.

    If the classifier is available locally, this will be used. Otherwise the
    classifier will be downloaded from W&B (Once implemented)
    """
    local_classifier_path: Path = config.local_classifier_dir / classifier_name

    if not local_classifier_path.exists():
        model_cache_dir = download_classifier_from_wandb_to_local(
            config, classifier_name, alias
        )
        local_classifier_path = Path(model_cache_dir) / "model.pickle"

    classifier = Classifier.load(local_classifier_path)

    return classifier


def load_document(config: Config, document_id: str) -> BaseParserOutput:
    """Download and opens a parser output based on a document ID."""
    s3 = boto3.client("s3", region_name=config.bucket_region)

    file_key = os.path.join(
        config.document_source_prefix,
        f"{document_id}.json",
    )

    response = s3.get_object(Bucket=config.cache_bucket, Key=file_key)
    content = response["Body"].read().decode("utf-8")
    document = BaseParserOutput.model_validate_json(content)
    return document


def _stringify(text: list[str]) -> str:
    return " ".join([line.strip() for line in text])


def document_passages(document: BaseParserOutput):
    """Yield the text block irrespective of content type."""
    match document.document_content_type:
        case "application/pdf":
            text_blocks = document.pdf_data.text_blocks
        case "text/html":
            text_blocks = document.html_data.text_blocks
        case _:
            raise ValueError(
                "Invalid document content type: "
                f"{document.document_content_type}, for "
                f"document: {document.document_id}"
            )
    for text_block in text_blocks:
        yield _stringify(text_block.text), text_block.text_block_id


@task(log_prints=True)
def store_labels(
    config: Config,
    labels: list[LabelledPassage],
    document_id: str,
    classifier_name: str,
    classifier_alias: str,
) -> None:
    """Store the labels in the cache bucket."""
    key = os.path.join(
        config.document_target_prefix,
        classifier_name,
        classifier_alias,
        f"{document_id}.json",
    )
    print(f"Storing labels for document {document_id} at {key}")

    data = [label.model_dump() for label in labels]
    body = BytesIO(json.dumps(data).encode("utf-8"))

    s3 = boto3.client("s3", region_name=config.bucket_region)
    s3.put_object(
        Bucket=config.cache_bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
    )


@task(log_prints=True)
def text_block_inference(
    classifier: Classifier, block_id: str, text: str
) -> LabelledPassage:
    """Run predict on a single text block."""
    spans: list[Span] = classifier.predict(text)
    labelled_passage = LabelledPassage(
        id=block_id,
        text=text,
        spans=spans,
    )
    return labelled_passage


@flow(log_prints=True)
async def run_classifier_inference_on_document(
    config: Config,
    document_id: str,
    classifier_name: str,
    classifier_alias: str,
) -> None:
    """Run the classifier inference flow on a document."""
    print(
        f"Loading classifier with name: {classifier_name}, and alias: {classifier_alias}"  # noqa: E501
    )
    classifier = await load_classifier(
        config,
        classifier_name,
        classifier_alias,
    )
    print(
        f"Loaded classifier with name: {classifier_name}, and alias: {classifier_alias}"  # noqa: E501
    )

    print(f"Loading document with ID {document_id}")
    document = load_document(config, document_id)
    print(f"Loaded document with ID {document_id}")

    futures = []

    for text, block_id in document_passages(document):
        futures.append(
            text_block_inference.submit(
                classifier=classifier, block_id=block_id, text=text
            )
        )

    doc_labels = [future.wait() for future in futures]

    store_labels(
        config=config,
        labels=doc_labels,
        document_id=document_id,
        classifier_name=classifier_name,
        classifier_alias=classifier_alias,
    )


@flow(log_prints=True, task_runner=ConcurrentTaskRunner())
async def classifier_inference(
    classifier_spec: list[tuple[str, str]],
    document_ids: Optional[list[str]] = None,
    config: Optional[Config] = None,
):
    """
    Flow to run inference on documents within a bucket prefix.

    Default behaviour is to run on everything, pass document_ids to
    limit to specific files.

    Iterates: classifiers > documents > passages. Loading output into s3

    params:
    - document_ids: List of document ids to run inference on
    - classifier_spec: List of classifier names and aliases (alias tag
      for the version) to run inference with
    - config: A Config object, uses the default if not given. Usually
      there is no need to change this outside of local dev
    Example classifier_spec: ["Q788", "latest")]
    """
    if not config:
        config = await Config.create()

    print(f"Running with config: {config}")

    current_bucket_ids = list_bucket_doc_ids(config=config)
    validated_document_ids = determine_document_ids(
        requested_document_ids=document_ids,
        current_bucket_ids=current_bucket_ids,
    )

    for classifier_name, classifier_alias in classifier_spec:
        subflows = [
            run_classifier_inference_on_document(
                config=config,
                document_id=document_id,
                classifier_name=classifier_name,
                classifier_alias=classifier_alias,
            )
            for document_id in validated_document_ids
        ]

        await asyncio.gather(*subflows)
