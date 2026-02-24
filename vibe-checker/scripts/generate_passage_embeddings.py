"""
Builds the passages dataset and embeddings for the vibe_check_inference flow.

Takes a balanced sample of passages from the document corpus (produced by
scripts/build_dataset.py) and computes a corresponding set of passage embeddings. If
prompted by the user, the script will also upload the three input files to the
vibe-checker S3 bucket.

You should re-run this script whenever you want to refresh the passages which are
available for the vibe_check_inference flow.

Usage:
    uv run vibe-checker/scripts/generate_passage_embeddings.py data/processed/sampled_dataset.feather
"""

import json
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import typer
from rich.console import Console
from sentence_transformers import SentenceTransformer

console = Console()

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
BATCH_SIZE = 1000
AWS_PROFILE = "labs"
AWS_REGION = "eu-west-1"


def generate(dataset_path: Path) -> None:
    """Build and upload the vibe-checker S3 input files from a sampled passages feather."""
    session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
    bucket_name = session.client("ssm").get_parameter(Name="/vibe-checker/bucket-name")[
        "Parameter"
    ]["Value"]
    s3 = session.client("s3")

    # Load the sampled dataset
    console.log(f"Loading dataset from {dataset_path}...")
    df = pd.read_feather(dataset_path)
    console.log(f"Loaded {len(df)} passages")

    # Compute embeddings
    console.log(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    texts = df["text_block.text"].tolist()

    console.log(f"Computing embeddings for {len(texts)} passages...")
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    assert isinstance(embeddings, np.ndarray)

    # Save outputs alongside the input file
    out_dir = dataset_path.parent
    feather_path = out_dir / "passages_dataset.feather"
    npy_path = out_dir / "passages_embeddings.npy"
    json_path = out_dir / "passages_embeddings_metadata.json"

    metadata = {
        "embedding_model_name": EMBEDDING_MODEL,
        "batch_size": BATCH_SIZE,
        "passages_count": len(texts),
    }

    console.log(f"Saving {feather_path}...")
    df.to_feather(feather_path)

    console.log(f"Saving {npy_path}...")
    np.save(npy_path, embeddings)

    console.log(f"Saving {json_path}...")
    json_path.write_text(json.dumps(metadata))

    console.log(f"✅ passages_dataset.feather ({len(df)} passages)")
    console.log(f"✅ passages_embeddings.npy (shape: {embeddings.shape})")
    console.log("✅ passages_embeddings_metadata.json")

    if not typer.confirm(f"\nUpload all three files to s3://{bucket_name}/?"):
        raise typer.Abort()

    s3.upload_file(str(feather_path), bucket_name, "passages_dataset.feather")
    s3.upload_file(str(npy_path), bucket_name, "passages_embeddings.npy")
    s3.upload_file(str(json_path), bucket_name, "passages_embeddings_metadata.json")
    console.log("✅ Uploaded to S3")


if __name__ == "__main__":
    typer.run(generate)
