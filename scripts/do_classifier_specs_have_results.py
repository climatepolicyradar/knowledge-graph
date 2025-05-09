import os

import boto3
import typer
import yaml
from botocore.exceptions import ClientError

from scripts.cloud import AwsEnv

app = typer.Typer()

PREFIX = os.getenv("LABELLED_PASSAGES_PREFIX", "labelled_passages")
YAML_FILES_MAP = {
    "prod": "flows/classifier_specs/prod.yaml",
    "staging": "flows/classifier_specs/staging.yaml",
    "sandbox": "flows/classifier_specs/sandbox.yaml",
    "labs": "flows/classifier_specs/labs.yaml",
}


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
) -> None:
    """
    Check if the classifier specs have s3 outputs and there respective counts.

    Once inference has been run we want to validate whether the classifiers that we
    have listed in the classifier spec yaml files have labelled passage outputs
    as well as the respective counts of the labelled passages for the classifiers.

    This can be used to help us validate that inference has run correctly.
    """
    with open(YAML_FILES_MAP[aws_env], "r") as file:
        data = yaml.safe_load(file)

    s3 = boto3.client("s3")

    to_process = []
    for classifier_spec in data:
        classifier_model, classifier_alias = classifier_spec.split(":")

        # Use os.path.join for path construction
        s3_path = os.path.join(bucket_name, PREFIX, classifier_model, classifier_alias)

        try:
            # List objects with the given prefix to see if path exists
            # Check if path exists and count objects with pagination
            paginator = s3.get_paginator("list_objects_v2")
            total_objects = 0
            for page in paginator.paginate(Bucket=bucket_name, Prefix=PREFIX):
                if "Contents" in page:
                    total_objects += len(page["Contents"])

            response = {"Contents": []} if total_objects > 0 else {}
            if total_objects > 0:
                print(
                    f"✅ S3 path exists: {s3_path} (contains {total_objects} objects)"
                )

            if "Contents" not in response:
                print(f"❌ S3 path does not exist: {s3_path}")
                to_process.append(f"{classifier_model}:{classifier_alias}")
        except ClientError as e:
            print(f"Error checking S3 path {s3_path}: {e}")

    print(f"to_process: {to_process}")


if __name__ == "__main__":
    app()
