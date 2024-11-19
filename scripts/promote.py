"""A script for model promotion."""

import logging
import os
from pathlib import Path
from typing import Annotated, List, Optional

import botocore
import botocore.client
import typer
from pydantic import (
    BaseModel,
    TypeAdapter,
    model_validator,
)
from rich.logging import RichHandler
from tqdm import tqdm
from typing_extensions import Self

import wandb
from scripts.cloud import AwsEnv, get_s3_client, is_logged_in
from src.identifiers import WikibaseID
from src.version import Version

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[
        RichHandler(rich_tracebacks=True),
    ],
)

log = logging.getLogger("rich")

ORG_ENTITY = "climatepolicyradar_UZODYJSN66HCQ"
REGISTRY_NAME = "model"
ENTITY = "climatepolicyradar"
JOB_TYPE = "promote_model"

REGION_NAME = "eu-west-1"


VALID_FROM_TO_TRANSITIONS = [
    (AwsEnv.sandbox, AwsEnv.labs),
    (AwsEnv.sandbox, AwsEnv.staging),
    (AwsEnv.labs, AwsEnv.staging),
    (AwsEnv.staging, AwsEnv.production),
]


class Across(BaseModel):
    """Promoting a model across 2 different AWS environments."""

    src: AwsEnv
    dst: AwsEnv
    primary: bool = False

    @model_validator(mode="after")
    def verify(self) -> Self:
        """Verify that valid combinations were used."""
        if self.src == self.dst:
            raise ValueError("src and dst must be different")

        if (self.src, self.dst) not in VALID_FROM_TO_TRANSITIONS:
            raise ValueError(f"cannot promote from {self.src.value} → {self.dst.value}")

        return self


class Within(BaseModel):
    """Promoting a model within the same AWS environment."""

    value: AwsEnv
    primary: bool = False


# A really simple ADT equivalent, with some helpers from Pydantic
# through the TypeAdapter, for initialisation.
Promotion = TypeAdapter(Across | Within)


def get_aliases(promotion: Promotion) -> Optional[List[str]]:
    """
    Get the aliases that will be attached to the entry in the model registry.

    This function determines the appropriate aliases for a model based on the
    promotion environment and whether it's a primary version.
    """
    match promotion:
        case Within(value=env, primary=True):
            return [env.value]
        case Across(dst=dst_env, primary=True):
            return [dst_env.value]
        case _:
            return None


def get_bucket_name_for_aws_env(aws_env: AwsEnv) -> str:
    """Generate the S3 bucket name for a given AWS environment."""
    return f"cpr-{aws_env.value}-models"


def get_object_key(concept: str, classifier: str, version: Version) -> Path:
    """Generate the S3 object key for a model."""
    return Path(concept) / classifier / str(version) / "model.pickle"


def copy_across_aws_envs(
    promotion: Promotion,
    concept: str,
    classifier: str,
    from_version: Version,
    to_version: Version,
    use_aws_profiles: bool,
) -> [str, str]:
    """
    Copy a model artifact from one AWS environment to another.

    This requires the callee to have done SSO login for both AWS environments.
    """
    from_object_key = get_object_key(concept, classifier, from_version)
    to_object_key = get_object_key(concept, classifier, to_version)

    cache_dir = Path.home() / ".cache" / "climatepolicyradar" / "models"

    # Make model cache directories
    from_file = cache_dir / promotion.src.value / from_object_key

    os.makedirs(from_file.parent, exist_ok=True)

    # Set bucket names (adjust these as needed)
    from_bucket = get_bucket_name_for_aws_env(promotion.src)
    to_bucket = get_bucket_name_for_aws_env(promotion.dst)

    log.info(
        "Copying model artifact "
        f"from [bold]{from_bucket}/{from_object_key}[/bold] "
        f"to [bold]{to_bucket}/{from_object_key}[/bold]..."
    )

    # Create S3 clients for both environments
    region_name = "eu-west-1"

    src_aws_env = promotion.src if use_aws_profiles else None
    dst_aws_env = promotion.dst if use_aws_profiles else None

    from_s3_client = get_s3_client(src_aws_env, region_name)
    to_s3_client = get_s3_client(dst_aws_env, region_name)

    log.info("Downloading...")
    # Download from source environment
    download(from_s3_client, from_bucket, from_object_key, from_file)
    log.info("Downloaded")

    log.info("Uploading...")
    # Upload to destination environment
    upload(to_s3_client, from_file, to_bucket, to_object_key)
    log.info("Uploaded")

    log.info(
        "Copied model artifact "
        f"from [bold]{from_bucket}/{from_object_key}[/bold] "
        f"to [bold]{to_bucket}/{to_object_key}[/bold]"
    )

    return to_bucket, to_object_key


def download(
    from_s3_client: botocore.client.BaseClient,
    from_bucket: str,
    object_key: Path,
    from_file: Path,
) -> None:
    """Download a file from S3."""
    response = from_s3_client.head_object(
        Bucket=from_bucket,
        Key=str(object_key),
    )
    total_length = int(response["ContentLength"])

    progress_bar = tqdm(
        total=total_length,
        unit="iB",
        unit_scale=True,
        desc=str(object_key),
    )

    from_s3_client.download_file(
        from_bucket,
        str(object_key),
        str(from_file),
        Callback=lambda bytes_transferred: progress_bar.update(bytes_transferred),
    )

    progress_bar.close()


def upload(
    to_s3_client: botocore.client.BaseClient,
    from_file: Path,
    to_bucket: str,
    object_key: Path,
):
    """Upload a file to S3."""
    file_size = os.path.getsize(from_file)

    progress_bar = tqdm(
        total=file_size,
        unit="iB",
        unit_scale=True,
        desc=str(object_key),
    )

    to_s3_client.upload_file(
        str(from_file),
        to_bucket,
        str(object_key),
        Callback=lambda bytes_transferred: progress_bar.update(bytes_transferred),
    )

    progress_bar.close()


def throw_not_logged_in(aws_env: AwsEnv):
    """Raise a typer.BadParameter exception for a not logged in AWS environment."""
    raise typer.BadParameter(
        f"you're not logged into {aws_env.value}. "
        f"Do `aws sso --login {aws_env.value}`"
    )


def validate_logins(
    promotion: Promotion,
    use_aws_profiles: bool,
) -> None:
    """Validate that the user is logged in to the necessary AWS environments."""
    match promotion:
        case Within(value=env):
            if not is_logged_in(env, use_aws_profiles):
                throw_not_logged_in(env)
        case Across(src=src_env, dst=dst_env):
            if not is_logged_in(src_env, use_aws_profiles):
                throw_not_logged_in(src_env)
            elif not is_logged_in(dst_env, use_aws_profiles):
                throw_not_logged_in(dst_env)


app = typer.Typer()


def parse_aws_env(value: str) -> str:
    """
    Parse a string a string as a possible enum value.

    We rely on a somewhat custom enum, to allow `"dev"`|`"staging"` for
    `staging`.
    """
    try:
        # This would convert `"dev"` to `AwsEnv.staging`.
        #
        # The raw value is returned, since we can't return an `AwsEnv` from
        # this function.
        return AwsEnv(value).value
    except ValueError as e:
        if "is not a valid AwsEnv" in str(e):
            valid = ", ".join([f"'{env.value}'" for env in AwsEnv])
            raise typer.BadParameter(f"'{value}' is not one of {valid}.")
        else:
            raise typer.BadParameter(str(e))


@app.command()
def main(
    wikibase_id: Annotated[
        WikibaseID,
        typer.Option(
            help="Wikibase ID of the concept",
            parser=WikibaseID,
        ),
    ],
    classifier: Annotated[
        str,
        typer.Option(
            help="Classifier name that aligns with the Python class name",
        ),
    ],
    version: Annotated[
        Version,
        typer.Option(
            help="Version of the model (e.g., v3)",
            parser=Version,
        ),
    ],
    from_aws_env: Annotated[
        Optional[AwsEnv],
        typer.Option(
            help="AWS environment to promote the model artifact from",
            parser=parse_aws_env,
        ),
    ] = None,
    to_aws_env: Annotated[
        Optional[AwsEnv],
        typer.Option(
            help="AWS environment to promote the model artifact to",
            parser=parse_aws_env,
        ),
    ] = None,
    within_aws_env: Annotated[
        Optional[AwsEnv],
        typer.Option(
            help="AWS environment to promote the model artifact within",
            parser=parse_aws_env,
        ),
    ] = None,
    primary: Annotated[
        bool,
        typer.Option(
            help="Whether this will be the primary version for this AWS environment",
        ),
    ] = False,
):
    """
    Promote a model from one account to another.

    Optionally as the primary model. If a W&B model registry
    collection doesn't exist for the concept, it'll
    automatically be made as part of this script.
    """
    log.info("Starting model promotion process")

    log.info("Parsing promotion...")

    if from_aws_env is not None or to_aws_env is not None:
        raise NotImplementedError(
            "Promotion across AWS environments is not yet implemented"
        )

    promotion = Promotion.validate_python(
        {
            "value": within_aws_env,
            "src": from_aws_env,
            "dst": to_aws_env,
            "primary": primary,
        }
    )

    use_aws_profiles = os.environ.get("USE_AWS_PROFILES", "true").lower() == "true"

    log.info("Validating AWS logins...")
    validate_logins(promotion, use_aws_profiles)

    log.info("Initializing Weights & Biases run...")
    run = wandb.init(entity=ENTITY, project=wikibase_id, job_type=JOB_TYPE)

    # Regardless of the promotion, we'll always be using some artifact.
    #
    # This also validates that the classifier exists. It relies on an
    # artifiact not existing. That is, when trying to `use_artifact`
    # below, it'll throw an exception.
    artifact_id = f"{wikibase_id}/{classifier}:{version}"
    log.info(f"Using artifact: {artifact_id}")
    artifact: wandb.Artifact = run.use_artifact(artifact_id)

    match promotion:
        case Across():
            log.info("Copying artifact between AWS environments...")

            from_version = version
            log.info(f"Incrementing version from {from_version}...")
            to_version = from_version.increment()
            log.info(f"Incrementing version to {to_version}")

            to_bucket, to_object_key = copy_across_aws_envs(
                promotion,
                wikibase_id,
                classifier,
                from_version,
                to_version,
                use_aws_profiles,
            )

            os.environ["AWS_PROFILE"] = promotion.dst.value

            metadata = {"aws_env": promotion.dst.value}

            # Re-set this variable to the new artifact
            log.info("Creating new W&B artifact...")

            artifact: wandb.Artifact = wandb.Artifact(
                name=classifier,
                type="model",
                metadata=metadata,
            )

            uri = os.path.join(
                "s3://",
                to_bucket,
                to_object_key,
            )
            log.info(f"Adding reference `{str(uri)}` to artifact...")

            # Don't checksum files since that means that W&B will try
            # and be too smart and will think a model artifact file in
            # a different AWS environment is the same, I think.
            artifact.add_reference(uri=uri, checksum=False)

            log.info("Logging new artifact to W&B...")
            artifact = run.log_artifact(artifact)
            log.info("Waiting for artifact in W&B...")
            artifact = artifact.wait()
            log.info(
                "Logged new artifact to W&B with metadata: "
                f"{artifact.metadata}, aliases: {artifact.aliases}, version: "
                f"{artifact._version}, name: {artifact._name}"
            )
        case Within():
            to_bucket = get_bucket_name_for_aws_env(promotion.value)
            to_object_key = get_object_key(wikibase_id, classifier, version)

    aliases = get_aliases(promotion)

    collection_name = wikibase_id

    # This magic value was from the W&B webapp.
    #
    # This is the hierarchy we use: CPR / {concept} / {model architecture}(s)
    #
    # The concept aka Wikibase ID is the collection name.
    #
    # > W&B automatically creates a collection with the name you specify
    # > in the target path if you try to link an artifact to a collection
    # > that does not exist. [1]
    #
    # [1] https://docs.wandb.ai/guides/registry/create_collection#programmatically-create-a-collection
    target_path = f"{ORG_ENTITY}/wandb-registry-{REGISTRY_NAME}/{collection_name}"

    # Link the artifact to a collection
    #
    # It will either be the Artifact that we originally used, if a _within_, or a newly logged
    # Artifact, if _across_.
    log.info(f"Linking artifact to collection: {target_path}...")
    run.link_artifact(
        artifact=artifact,
        target_path=target_path,
        aliases=aliases,
    )

    log.info("Finishing W&B run...")

    run.finish()

    log.info("Model promoted")


if __name__ == "__main__":
    app()
