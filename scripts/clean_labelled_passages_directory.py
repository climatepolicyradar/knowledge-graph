# This is a script for clearing down the labelled passages directory of outputs that
# were from running inference on non-English language documents.
#
# The script will:
# 1. Find all the non-english language documents from the embeddings input prefix.
# 2. Delete the non-english language documents from the labelled passages prefix;
#   using the corresponding file stems.
#
# E.g. embeddings_input/CCLW.exec.1.1["languages"] != ["en"]
#   -> delete labelled_passages/Q123/v1/CCLW.exec.1.1.json

import json
from pathlib import Path
from typing import Any, Dict, Generator

import boto3
import typer


def list_s3_objects(bucket_name: str, prefix: str) -> Generator[str, None, None]:
    """List all objects in an S3 bucket."""
    typer.echo(f"Listing objects in {bucket_name} with prefix {prefix}")
    s3_client = boto3.client("s3")
    paginator = s3_client.get_paginator("list_objects_v2")

    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            yield obj["Key"]


def load_json_from_s3(bucket_name: str, key: str) -> Dict[str, Any]:
    """Load JSON content from an S3 object."""
    s3_client = boto3.client("s3")
    response = s3_client.get_object(Bucket=bucket_name, Key=key)
    content = response["Body"].read().decode("utf-8")
    return json.loads(content)


def json_generator(
    bucket_name: str, prefix: str
) -> Generator[tuple[Dict[str, Any], str], None, None]:
    """Generator that yields JSON objects from S3 bucket."""
    for key in list_s3_objects(bucket_name, prefix):
        try:
            yield load_json_from_s3(bucket_name, key), key
        except json.JSONDecodeError:
            print(f"Skipping {key}: Invalid JSON format")
        except Exception as e:
            print(f"Skipping {key}: {e}")


def extract_non_english_language_documents(
    bucket_name: str, prefix: str
) -> list[tuple[Dict[str, Any], str]]:
    """Extract the 'languages' key from JSON objects in S3 bucket."""
    typer.echo(f"Extracting non-English language documents from {bucket_name}/{prefix}")
    non_english_documents = []
    for json_obj, key in json_generator(bucket_name, prefix):
        languages = json_obj.get("languages", [])
        if languages and languages != ["en"]:
            typer.echo(f"Non-English document found: {key}")
            non_english_documents.append((json_obj, key))
    return non_english_documents


def delete_s3_object(bucket_name: str, key: str):
    """Delete an object from an S3 bucket."""
    s3_client = boto3.client("s3")
    s3_client.delete_object(Bucket=bucket_name, Key=key)
    typer.echo(f"Deleted {key} from {bucket_name}")


def main(
    bucket_name: str = "cpr-staging-data-pipeline-cache",
    embeddings_input_prefix: str = "embeddings_input/",
    labelled_passages_prefix: str = "labelled_passages/",
):
    typer.echo("Cleaning Labelled Passages Directory...")
    typer.echo(f"Bucket Name: {bucket_name}")
    typer.echo(f"Embeddings Input Prefix: {embeddings_input_prefix}")
    typer.echo(f"Labelled Passages Prefix: {labelled_passages_prefix}")

    non_english_documents = extract_non_english_language_documents(
        bucket_name, embeddings_input_prefix
    )

    # The  stem of the file in the embeddings input directory will match the stem of
    # the file in the labelled passages output directory.
    non_english_documents_stems = [Path(key).stem for _, key in non_english_documents]

    # Clear down documents with a matching file stem.
    #
    # E.g. inference run on cclw.exec.1.1 might produce:
    #  labelled_passages/Q123/later/cclw.exec.1.1.json
    #  labelled_passages/Q123/v1/cclw.exec.1.1.json
    #  labelled_passages/Q123/v2/cclw.exec.1.1.json
    #
    # So we remove all the files with matching stems.
    for data, key in json_generator(bucket_name, labelled_passages_prefix):
        if Path(key).stem in non_english_documents_stems:
            delete_s3_object(bucket_name, key)


if __name__ == "__main__":
    typer.run(main)
