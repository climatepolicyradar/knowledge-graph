import boto3
import typer
from rich.console import Console
from rich.progress import track

from scripts.config import aws_region, data_dir

console = Console()

session = boto3.Session(profile_name="labs")
s3_client = session.client("s3", region_name=aws_region)
bucket_name = "cpr-sectors-classifier-sampling"
existing_buckets = [
    bucket["Name"] for bucket in s3_client.list_buckets().get("Buckets")
]
if bucket_name in existing_buckets:
    # first empty the bucket
    if typer.confirm(
        f"🪣 Bucket {bucket_name} already exists. Do you want to delete it?"
    ):
        objects = s3_client.list_objects_v2(Bucket=bucket_name).get("Contents", [])
        for obj in track(
            objects,
            description="🪣 Deleting existing data from AWS S3...",
            transient=True,
        ):
            s3_client.delete_object(Bucket=bucket_name, Key=obj["Key"])
        # then delete the bucket
        s3_client.delete_bucket(Bucket=bucket_name)
        console.print(
            f"🪣 Deleted existing AWS S3 bucket: {bucket_name}", style="green"
        )
    else:
        console.print(
            f"🪣 Bucket {bucket_name} already exists. Please delete it and try again."
        )
        raise typer.Abort()

bucket = s3_client.create_bucket(
    Bucket=bucket_name, CreateBucketConfiguration={"LocationConstraint": aws_region}
)
console.print(f"🪣 Created new AWS S3 bucket: {bucket_name}", style="green")

file_paths = list(data_dir.rglob("*"))
for file_path in track(
    file_paths,
    description="☁️ Uploading data to AWS S3...",
    total=len(file_paths),
    transient=True,
):
    if file_path.is_file():
        with file_path.open("rb") as file:
            s3_client.upload_fileobj(
                file, bucket_name, str(file_path.relative_to(data_dir))
            )

console.print("☁️ All data uploaded successfully to AWS S3", style="green")
console.print(f"🔗 S3 bucket URL: https://{bucket_name}.s3.amazonaws.com/")
