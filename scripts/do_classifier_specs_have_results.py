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
    """Check if the classifier specs in the YAML file have corresponding S3 paths."""
    with open(yaml_path, "r") as file:
        data = yaml.safe_load(file)

    s3 = boto3.client("s3")

    to_process = []
    for classifier_spec in data:
        classifier_model = classifier_spec.split(":")[0]
        classifier_alias = classifier_spec.split(":")[1]

        # Use os.path.join for path construction
        s3_path = os.path.join(
            labelled_passages_s3_path, classifier_model, classifier_alias
        )

        # Parse the S3 URI to get bucket and prefix
        bucket_name = s3_path.split("/")[2]
        prefix = "/".join(s3_path.split("/")[3:])

        try:
            # List objects with the given prefix to see if path exists
            response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, MaxKeys=1)

            if "Contents" in response:
                print(f"✅ S3 path exists: {s3_path}")
            else:
                print(f"❌ S3 path does not exist: {s3_path}")
                to_process.append(f"{classifier_model}:{classifier_alias}")
        except ClientError as e:
            print(f"Error checking S3 path {s3_path}: {e}")

    print(f"to_process: {to_process}")


if __name__ == "__main__":
    app()
