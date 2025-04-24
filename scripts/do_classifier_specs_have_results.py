import os

import boto3
import typer
import yaml
from botocore.exceptions import ClientError

app = typer.Typer()


@app.command()
def check_classifier_specs(
    yaml_path: str = typer.Argument(
        help="Path to the YAML file containing classifier specifications"
    ),
    labelled_passages_s3_path: str = typer.Argument(
        help="S3 path where labelled passages should be stored"
    ),
) -> None:
    """
    Check if the classifier specs have s3 outputs and there respective counts.

    Once inference has been run we want to validate whether the classifiers that we 
    have listed in the classifier spec yaml files have labelled passage outputs 
    as well as the respective counts of the labelled passages for the classifiers. 

    This can be used to help us validate that inference has run correctly. 
    """
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)

    s3 = boto3.client("s3")

    to_process = []
    for classifier_spec in data:
        classifier_model, classifier_alias = classifier_spec.split(":")

        # Use os.path.join for path construction
        s3_path = os.path.join(
            labelled_passages_s3_path, classifier_model, classifier_alias
        )

        # Parse the S3 URI to get bucket and prefix
        bucket_name = s3_path.split("/")[2]
        prefix = "/".join(s3_path.split("/")[3:])

        try:
            # List objects with the given prefix to see if path exists
            # Check if path exists and count objects with pagination
            paginator = s3.get_paginator("list_objects_v2")
            total_objects = 0
            for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
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
