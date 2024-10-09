import json
import os
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import boto3
from cpr_sdk.parser_models import BaseParserOutput
from prefect import flow, task

from src.classifier import Classifier
from src.identifiers import WikibaseID
from src.labelled_passage import LabelledPassage
from src.span import Span


@dataclass()
class Config:
    """Settings used across flow runs"""

    cache_bucket: str = os.environ.get("CACHE_BUCKET")
    document_source_prefix: str = "embeddings_input"
    document_target_prefix: str = "labelled_passages"
    bucket_region: str = "eu-west-1"
    local_classifier_dir: Path = Path("data") / "processed" / "classifiers"


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
    requested_document_ids: list[str], current_bucket_ids: list[str]
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


def load_classifier(wikibase_id: WikibaseID) -> Classifier:
    """
    Loads a classifier into memory

    If the classifier is available locally, this will be used. Otherwise the
    classifier will be downloaded from W&B (Once implemented)
    """
    local_classifier_path: Path = config.local_classifier_dir / wikibase_id

    if not local_classifier_path.exists():
        raise NotImplementedError("Still need to add W&B download")

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


def document_passages(document: BaseParserOutput):
    """Yields the text block irrespective of content type"""
    match document.document_content_type:
        case "application/pdf":
            text_blocks = document.pdf_data.text_blocks
        case "text/html":
            text_blocks = document.html_data.text_blocks
    for text_block in text_blocks:
        yield text_block.to_string(), text_block.text_block_id


def store_labels(labels: list[LabelledPassage], document_id: str, classifier_id: str):
    key = os.path.join(
        config.document_target_prefix, f"{document_id}.{classifier_id}.json"
    )

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
def classifier_inference(document_ids: list[str] = None):
    """
    Flow to run inference on documents within a bucket prefix

    Default behaviour is to run on everything, pass document_ids to limit to specific
    files.

    Iterates: classifiers > documents > passages. Loading output into s3
    """
    print(f"Running with config: {config}")

    current_bucket_ids = list_bucket_doc_ids()
    validated_document_ids = determine_document_ids(
        requested_document_ids=document_ids, current_bucket_ids=current_bucket_ids
    )

    # TODO Better way of choosing classifiers
    wikibase_ids = [WikibaseID("Q788")]

    for wikibase_id in wikibase_ids:
        print(f"Loading classifier with id: {wikibase_id}")
        classifier = load_classifier(wikibase_id)
        for document_id in validated_document_ids:
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
                classifier_id=wikibase_id,
            )


if __name__ == "__main__":
    classifier_inference()
