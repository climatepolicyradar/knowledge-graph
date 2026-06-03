"""
Flow to generate datasets for the vibe checker.

Reads the combined dataset feather file produced by the build_dataset flow), generates
embeddings using a sentence transformer model, and uploads the three input files
required by the vibe_check_inference flow to the vibe-checker S3 bucket.
"""

import io
import json
import os

import boto3
import numpy as np
import pandas as pd
from mypy_boto3_s3 import S3Client
from prefect import flow, task

from flows.build_dataset_flow import COMBINED_S3_KEY
from flows.config import Config
from flows.vibe_check import (
    get_bucket_name_from_ssm,
    push_object_bytes_to_s3,
)
from knowledge_graph.utils import get_logger, iterate_batch

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 1000


logger = get_logger()


async def _set_up_environment(
    config: Config | None,
) -> tuple[Config, S3Client]:
    """
    Set up the shared config and S3 client for the flow.

    :param config: Optional pre-configured Config object. If not provided, will be created.
    """
    if not config:
        config = await Config.create()

    use_aws_profiles = os.environ.get("USE_AWS_PROFILES", "false").lower() == "true"
    session = boto3.session.Session(
        profile_name=config.aws_env.value if use_aws_profiles else None,
        region_name=config.bucket_region,
    )
    s3_client = session.client("s3")

    return config, s3_client


@task(retries=3, retry_delay_seconds=5)
def load_combined_dataset(s3_client: S3Client, dataset_s3_bucket: str) -> pd.DataFrame:
    """Load the combined dataset from the feather files S3 bucket."""
    logger.info(
        f"Loading combined dataset from s3://{dataset_s3_bucket}/{COMBINED_S3_KEY}"
    )
    response = s3_client.get_object(Bucket=dataset_s3_bucket, Key=COMBINED_S3_KEY)
    df = pd.read_feather(io.BytesIO(response["Body"].read()))
    if df.empty:
        raise ValueError("Combined dataset is empty")
    logger.info(f"Loaded {len(df)} passages")
    return df


@task
def generate_embeddings(
    df: pd.DataFrame, embedding_model_name: str, batch_size: int
) -> np.ndarray:
    """Generate normalised embeddings for all passages in the dataset."""
    from sentence_transformers import SentenceTransformer

    logger.info(f"Loading embedding model: {embedding_model_name}")
    model = SentenceTransformer(embedding_model_name)
    texts = df["text_block.text"].tolist()
    logger.info(f"Computing embeddings for {len(texts)} passages...")

    chunks = list(iterate_batch(texts, batch_size))
    parts: list[np.ndarray] = []
    for i, chunk in enumerate(chunks):
        embedded = model.encode(
            list(chunk),
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        assert isinstance(embedded, np.ndarray)
        parts.append(embedded)
        logger.info(
            f"Encoded {min((i + 1) * batch_size, len(texts))}/{len(texts)} passages "
            f"({i + 1}/{len(chunks)} batches)"
        )

    embeddings = np.concatenate(parts, axis=0)
    logger.info(f"Generated embeddings with shape {embeddings.shape}")
    return embeddings


@task(retries=3, retry_delay_seconds=5)
def upload_vibe_checker_files(
    s3_client: S3Client,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    embedding_model_name: str,
    batch_size: int,
) -> None:
    """Upload passages dataset, embeddings, and metadata to the vibe-checker S3 bucket."""
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
        "batch_size": batch_size,
        "passages_count": len(df),
    }
    push_object_bytes_to_s3(
        s3_client,
        "passages_embeddings_metadata.json",
        json.dumps(metadata).encode("utf-8"),
    )
    logger.info("Uploaded passages_embeddings_metadata.json")


@flow(timeout_seconds=None)
async def generate_vibe_checker_datasets(
    embedding_model_name: str = EMBEDDING_MODEL,
    batch_size: int = BATCH_SIZE,
    config: Config | None = None,
) -> None:
    """
    Generate passage embeddings for the vibe checker.

    Reads the combined dataset from s3, computes embeddings for all passages, and
    uploads the resulting files to the vibe-checker S3 bucket for use by the
    vibe_check_inference flow.

    :param embedding_model_name: Sentence transformer model used to embed passages
    :param batch_size: Batch size used when encoding passages
    :param config: Optional pre-configured Config object. If not provided, will be created.
    """
    config, s3_client = await _set_up_environment(config=config)

    combined_dataset_df = load_combined_dataset(s3_client, config.dataset_s3_bucket)
    embeddings = generate_embeddings(
        combined_dataset_df,
        embedding_model_name=embedding_model_name,
        batch_size=batch_size,
    )
    upload_vibe_checker_files(
        s3_client,
        combined_dataset_df,
        embeddings,
        embedding_model_name=embedding_model_name,
        batch_size=batch_size,
    )
    logger.info("Vibe checker embeddings generation complete")
