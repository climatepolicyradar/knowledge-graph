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

from src.classifier import Classifier
from src.labelled_passage import LabelledPassage
from src.span import Span

AWS_ENV = os.environ["AWS_ENV"]
WORKPOOL_DEFAULT_JOB_VARIABLES = JSON.load(
    f"default-job-variables-prefect-mvp-{AWS_ENV}"
).value


def get_aws_ssm_param(param_name: str) -> str:
    """Retrieve a parameter from AWS SSM"""
    ssm = boto3.client("ssm")
    response = ssm.get_parameter(Name=param_name, WithDecryption=True)
    return response["Parameter"]["Value"]


@dataclass()
class Config:
    """Settings used across flow runs"""

    cache_bucket: str = WORKPOOL_DEFAULT_JOB_VARIABLES["pipeline_cache_bucket_name"]
    document_source_prefix: str = "embeddings_input"
    document_target_prefix: str = "labelled_passages"
    bucket_region: str = "eu-west-1"
    local_classifier_dir: Path = Path("data") / "processed" / "classifiers"
    wandb_model_registry: str = "climatepolicyradar_UZODYJSN66HCQ/wandb-registry-model/"


config = Config()


def list_bucket_doc_ids() -> list[str]:
    """Scan configured bucket and return all ids"""
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
    If a document id has been requested but does not exist this will raise a ValueError
    If no document id were requested, this will instead return the current_bucket_ids.
    """
    if requested_document_ids is None:
        return current_bucket_ids

    missing_from_bucket = list(set(requested_document_ids) - set(current_bucket_ids))
    if len(missing_from_bucket) > 0:
        raise ValueError(
            f"Requested document_ids not found in bucket: {missing_from_bucket}"
        )

    return requested_document_ids


def download_classifier_from_wandb_to_local(classifier_name: str, alias: str) -> str:
    """
    Function for downloading a classifier from W&B to local.

    Models referenced by weights and biases are stored in s3. This means that to
    download the model via the W&B API, we need access to both the s3 bucket via iam
    in your environment and WanDB via the api key.
    """
    wandb.login(key=get_aws_ssm_param("WANDB_API_KEY"))
    run = wandb.init()
    artifact = config.wandb_model_registry + f"{classifier_name}:{alias or 'latest'}"
    print(f"Downloading artifact from W&B: {artifact}")
    artifact = run.use_artifact(artifact, type="model")
    return artifact.download()


def load_classifier(classifier_name: str, alias: str) -> Classifier:
    """
    Loads a classifier into memory

    If the classifier is available locally, this will be used. Otherwise the
    classifier will be downloaded from W&B (Once implemented)
    """
    local_classifier_path: Path = config.local_classifier_dir / classifier_name

    if not local_classifier_path.exists():
        model_cache_dir = download_classifier_from_wandb_to_local(
            classifier_name, alias
        )
        local_classifier_path = Path(model_cache_dir) / "model.pickle"

    classifier = Classifier.load(local_classifier_path)

    return classifier


def load_document(document_id: str) -> BaseParserOutput:
    """Downloads and opens a parser output based on a document id"""
    s3 = boto3.client("s3", region_name=config.bucket_region)

    file_key = os.path.join(config.document_source_prefix, f"{document_id}.json")

    response = s3.get_object(Bucket=config.cache_bucket, Key=file_key)
    content = response["Body"].read().decode("utf-8")
    document = BaseParserOutput.model_validate_json(content)
    return document


def stringify(text: list[str]) -> str:
    return " ".join([line.strip() for line in text])


def document_passages(document: BaseParserOutput):
    """Yields the text block irrespective of content type"""
    match document.document_content_type:
        case "application/pdf":
            text_blocks = document.pdf_data.text_blocks
        case "text/html":
            text_blocks = document.html_data.text_blocks
        case _:
            raise ValueError(
                f"Invalid document content type: {document.document_content_type}, for "
                f"document: {document.document_id}"
            )
    for text_block in text_blocks:
        yield stringify(text_block.text), text_block.text_block_id


@task(log_prints=True)
def store_labels(
    labels: list[LabelledPassage],
    document_id: str,
    classifier_name: str,
    classifier_alias: str,
) -> None:
    """Stores the labels in the cache bucket"""
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
        Bucket=config.cache_bucket, Key=key, Body=body, ContentType="application/json"
    )


@task()
def text_block_inference(
    classifier: Classifier, block_id: str, text: str
) -> LabelledPassage:
    """Runs predict on a single text block"""
    spans: list[Span] = classifier.predict(text)
    labelled_passage = LabelledPassage(
        id=block_id,
        text=text,
        spans=spans,
    )
    return labelled_passage


@flow(log_prints=True)
def run_classifier_inference_on_document(
    document_id: str,
    classifier: Classifier,
    classifier_name: str,
    classifier_alias: str,
) -> None:
    """Run the classifier inference flow on a document."""
    print(f"Loading document with id {document_id}")
    document = load_document(document_id)

    doc_labels = []
    for text, block_id in document_passages(document):
        labelled_passage = text_block_inference(
            classifier=classifier, block_id=block_id, text=text
        )
        doc_labels.append(labelled_passage)

    store_labels(
        labels=doc_labels,
        document_id=document_id,
        classifier_name=classifier_name,
        classifier_alias=classifier_alias,
    )


@flow(log_prints=True)
def classifier_inference(
    classifier_spec: list[tuple[str, str]], document_ids: Optional[list[str]] = None
):
    """
    Flow to run inference on documents within a bucket prefix

    Default behaviour is to run on everything, pass document_ids to limit to specific
    files.

    Iterates: classifiers > documents > passages. Loading output into s3

    params:
    - document_ids: List of document ids to run inference on
    - classifier_spec: List of classifier names and aliases (alias tag for the version) to run inference with
    Example classifier_spec: ["Q788", "latest")]
    """
    print(f"Running with config: {config}")

    current_bucket_ids = list_bucket_doc_ids()
    validated_document_ids = determine_document_ids(
        requested_document_ids=document_ids, current_bucket_ids=current_bucket_ids
    )

    for classifier_name, classifier_alias in classifier_spec:
        print(
            f"Loading classifier with name: {classifier_name}, and alias: {classifier_alias}"
        )
        classifier = load_classifier(classifier_name, classifier_alias)
        for document_id in validated_document_ids:
            run_classifier_inference_on_document(
                document_id=document_id,
                classifier=classifier,
                classifier_name=classifier_name,
                classifier_alias=classifier_alias,
            )
