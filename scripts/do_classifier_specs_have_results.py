import boto3
import yaml
from botocore.exceptions import ClientError

PROD_YAML = "flows/classifier_specs/prod.yaml"
PROD_LABELLED_PASSAGES_S3_PATH = "s3://cpr-prod-data-pipeline-cache/labelled_passages/"

with open(PROD_YAML, "r") as file:
    data = yaml.safe_load(file)

s3 = boto3.client("s3")

to_process = []
for classifier_spec in data:
    classifier_model = classifier_spec.split(":")[0]
    classifier_alias = classifier_spec.split(":")[1]

    labelled_passages_s3_path = (
        PROD_LABELLED_PASSAGES_S3_PATH + classifier_model + "/" + classifier_alias + "/"
    )

    # Parse the S3 URI to get bucket and prefix
    bucket_name = labelled_passages_s3_path.split("/")[2]
    prefix = "/".join(labelled_passages_s3_path.split("/")[3:])

    try:
        # List objects with the given prefix to see if path exists
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix, MaxKeys=1)

        if "Contents" in response:
            print(f"✅ S3 path exists: {labelled_passages_s3_path}")
        else:
            print(f"❌ S3 path does not exist: {labelled_passages_s3_path}")
            to_process.append(f"{classifier_model}:{classifier_alias}")
    except ClientError as e:
        print(f"Error checking S3 path {labelled_passages_s3_path}: {e}")

print(f"to_process: {to_process}")
