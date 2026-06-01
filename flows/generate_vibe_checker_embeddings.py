"""
Flow to generate passage embeddings for the vibe checker.

Reads the combined dataset from s3://cpr-kg-feather-files (produced by the
build_dataset Prefect flow), generates embeddings using a sentence transformer
model, and uploads the three input files required by the vibe_check_inference
flow to the vibe-checker S3 bucket.

This flow should run on a schedule (e.g. weekly) to keep the vibe-checker
embeddings up to date with the latest combined dataset.
"""

import io
import json
import os

import boto3
import numpy as np
import pandas as pd
from prefect import flow, task

from flows.vibe_check import (
    get_bucket_name_from_ssm,
    get_s3_client,
    push_object_bytes_to_s3,
)
from knowledge_graph.utils import get_logger

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 1000
FEATHER_FILES_BUCKET = "cpr-kg-feather-files"
COMBINED_DATASET_KEY = "combined_dataset.feather"

aws_region = os.getenv("AWS_REGION", "eu-west-1")
aws_profile = (
    os.getenv("AWS_PROFILE")
    if os.environ.get("USE_AWS_PROFILES", "false").lower() == "true"
    else None
)

logger = get_logger()


@task(retries=3, retry_delay_seconds=5)
def load_combined_dataset() -> pd.DataFrame:
    """Load the combined dataset from the feather files S3 bucket."""
    session = boto3.Session(region_name=aws_region, profile_name=aws_profile)
    s3 = session.client("s3")
    logger.info(
        f"Loading combined dataset from s3://{FEATHER_FILES_BUCKET}/{COMBINED_DATASET_KEY}"
    )
    response = s3.get_object(Bucket=FEATHER_FILES_BUCKET, Key=COMBINED_DATASET_KEY)
    df = pd.read_feather(io.BytesIO(response["Body"].read()))
    if df.empty:
        raise ValueError("Combined dataset is empty")
    logger.info(f"Loaded {len(df)} passages")
    return df


@task
def generate_embeddings(
    df: pd.DataFrame, embedding_model_name: str = EMBEDDING_MODEL
) -> np.ndarray:
    """Generate normalised embeddings for all passages in the dataset."""
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading embedding model: {embedding_model_name}")
    model = SentenceTransformer(embedding_model_name)
    texts = df["text_block.text"].tolist()
    logger.info(f"Computing embeddings for {len(texts)} passages...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    assert isinstance(embeddings, np.ndarray)
    logger.info(f"Generated embeddings with shape {embeddings.shape}")
    return embeddings


@task(retries=3, retry_delay_seconds=5)
def upload_vibe_checker_files(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    embedding_model_name: str = EMBEDDING_MODEL,
) -> None:
    """Upload passages dataset, embeddings, and metadata to the vibe-checker S3 bucket."""
    s3_client = get_s3_client()
    bucket_name = get_bucket_name_from_ssm()
    logger.info(f"Uploading vibe-checker files to s3://{bucket_name}/")

    feather_buffer = io.BytesIO()
    df.to_feather(feather_buffer)
    feather_buffer.seek(0)
    push_object_bytes_to_s3(
        s3_client, "passages_dataset.feather", feather_buffer.read()
    )
    logger.info(f"Uploaded passages_dataset.feather ({len(df)} passages)")

    npy_buffer = io.BytesIO()
    np.save(npy_buffer, embeddings)
    npy_buffer.seek(0)
    push_object_bytes_to_s3(s3_client, "passages_embeddings.npy", npy_buffer.read())
    logger.info(f"Uploaded passages_embeddings.npy (shape: {embeddings.shape})")

    metadata = {
        "embedding_model_name": embedding_model_name,
        "batch_size": BATCH_SIZE,
        "passages_count": len(df),
    }
    push_object_bytes_to_s3(
        s3_client,
        "passages_embeddings_metadata.json",
        json.dumps(metadata).encode("utf-8"),
    )
    logger.info("Uploaded passages_embeddings_metadata.json")


@flow(timeout_seconds=None)
def generate_vibe_checker_embeddings(
    embedding_model_name: str = EMBEDDING_MODEL,
) -> None:
    """
    Generate passage embeddings for the vibe checker.

    Reads the combined dataset from s3://cpr-kg-feather-files, computes
    embeddings for all passages, and uploads the resulting files to the
    vibe-checker S3 bucket for use by the vibe_check_inference flow.
    """
    df = load_combined_dataset()
    embeddings = generate_embeddings(df, embedding_model_name=embedding_model_name)
    upload_vibe_checker_files(df, embeddings, embedding_model_name=embedding_model_name)
    logger.info("Vibe checker embeddings generation complete")


if __name__ == "__main__":
    generate_vibe_checker_embeddings()
