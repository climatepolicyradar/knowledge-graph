import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import boto3
import typer
import yaml
from botocore.exceptions import ClientError

from scripts.cloud import AwsEnv

app = typer.Typer()

BASE_PREFIX = os.getenv("LABELLED_PASSAGES_PREFIX", "labelled_passages")
YAML_FILES_MAP = {
    "prod": "flows/classifier_specs/prod.yaml",
    "staging": "flows/classifier_specs/staging.yaml",
    "sandbox": "flows/classifier_specs/sandbox.yaml",
    "labs": "flows/classifier_specs/labs.yaml",
}


@dataclass
class Result:
    """Result of checking a single classifier spec"""

    path_exists: bool = False
    classifier_spec: str = ""


def collect_file_names(bucket_name: str, prefix: str) -> list[str]:
    """Collect the names of all files under a given prefix in an s3 bucket."""

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    file_names = []
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        if "Contents" in page:
            file_names.extend([obj["Key"] for obj in page["Contents"]])
    return file_names


def check_single_spec(bucket_name: str, classifier_spec: str) -> Result:
    """Check inference output for a single classifier_spec."""
    classifier_model, classifier_alias = classifier_spec.split(":")
    prefix = os.path.join(BASE_PREFIX, classifier_model, classifier_alias)
    try:
        file_names = collect_file_names(bucket_name, prefix)
    except ClientError as e:
        print(f"Error checking results for {classifier_spec}: {e}")
        return Result(path_exists=False, classifier_spec=classifier_spec)

    if file_names:
        print(f"✅ Results for {classifier_spec}: {len(file_names)} objects")
        return Result(path_exists=True, classifier_spec=classifier_spec)
    else:
        print(f"❌ No results for {classifier_spec}")
        return Result(path_exists=False, classifier_spec=classifier_spec)


@app.command()
def check_classifier_specs(
    aws_env: AwsEnv = typer.Argument(
        help="Which aws environment to look for results in. Determines which spec file"
        "to use",
        default="sandbox",
    ),
    bucket_name: str = typer.Argument(
        help=(
            "Name of the s3 bucket, should be the root without protocol or prefix"
            "i.e. my-bucket-name"
        )
    ),
    max_workers: int = typer.Option(
        default=10,
        help="Maximum number of parallel workers to use for checking specs",
    ),
) -> None:
    """
    Check if the classifier specs have s3 outputs and there respective counts.

    Once inference has been run we want to validate whether the classifiers that we
    have listed in the classifier spec yaml files have labelled passage outputs
    as well as the respective counts of the labelled passages for the classifiers.

    This can be used to help us validate that inference has run correctly.
    """
    typer.echo(f"Checking {aws_env} classifier specs in {bucket_name}/{BASE_PREFIX}")
    with open(YAML_FILES_MAP[aws_env], "r") as file:
        data = yaml.safe_load(file)

    to_process = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_spec = {
            executor.submit(check_single_spec, bucket_name, spec): spec for spec in data
        }

        for future in as_completed(future_to_spec):
            result = future.result()
            if not result.path_exists:
                to_process.append(result.classifier_spec)

    typer.echo(f"to_process: {to_process}")


if __name__ == "__main__":
    app()
