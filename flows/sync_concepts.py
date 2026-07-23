import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
from cpr_sdk.ssm import get_aws_ssm_param
from prefect import flow, task
from prefect.cache_policies import NONE
from pydantic import AnyHttpUrl, SecretStr

from flows.utils import (
    S3Uri,
    SlackNotify,
    total_milliseconds,
)
from knowledge_graph.cloud import AwsEnv, get_async_session
from knowledge_graph.concept import Concept
from knowledge_graph.utils import get_logger
from knowledge_graph.wikibase import WikibaseAuth, WikibaseSession

VESPA_MAX_TIMEOUT_MS: int = total_milliseconds(timedelta(minutes=5))
VESPA_CONNECTION_POOL_SIZE: int = 5

WIKIBASE_PASSWORD_SSM_NAME = "/Wikibase/Cloud/ServiceAccount/Password"
WIKIBASE_USERNAME_SSM_NAME = "/Wikibase/Cloud/ServiceAccount/Username"
WIKIBASE_URL_SSM_NAME = "/Wikibase/Cloud/URL"


def concepts_to_dataframe(concepts: list[Concept]) -> pl.DataFrame:
    """
    Convert a list of Concepts to a Polars DataFrame for Parquet storage.

    Args:
        concepts: List of Concept objects to convert

    Returns:
        pl.DataFrame: A DataFrame with all concepts, ready for Parquet write
    """
    df = pl.DataFrame(
        [
            concept.model_dump(
                exclude={"labelled_passages", "classifier_ids"}, mode="python"
            )
            for concept in concepts
        ]
    )

    # Force explicit dtypes so empty/None columns don't infer as Null or
    # List(Null), which differ from populated columns and break unioning the
    # archive's Parquet files across runs.
    schema_casts = {
        "description": pl.Utf8,
        "definition": pl.Utf8,
        "wikibase_id": pl.Utf8,
        "wikibase_revision": pl.Int64,
        "alternative_labels": pl.List(pl.Utf8),
        "negative_labels": pl.List(pl.Utf8),
        "subconcept_of": pl.List(pl.Utf8),
        "has_subconcept": pl.List(pl.Utf8),
        "related_concepts": pl.List(pl.Utf8),
        "negative_concepts": pl.List(pl.Utf8),
        "recursive_subconcept_of": pl.List(pl.Utf8),
        "recursive_has_subconcept": pl.List(pl.Utf8),
    }

    if casts := [
        pl.col(col).cast(dtype)
        for col, dtype in schema_casts.items()
        if col in df.columns and df[col].dtype != dtype
    ]:
        df = df.with_columns(casts)

    # Add sync timestamp to track when this version was synced
    df = df.with_columns(pl.lit(datetime.now(timezone.utc)).alias("synced_at"))

    return df


async def s3_prefix_has_objects(s3_uri: S3Uri, region: str, aws_env: AwsEnv) -> bool:
    """
    Check if an S3 prefix has any objects.

    I found the Polars errors to be misleading, about the remote state
    of the data. This explicit check is not misleading.

    Returns `False` if the prefix has no objects.
    """
    session = get_async_session(
        region_name=region,
        aws_env=aws_env,
    )
    async with session.client("s3") as s3:
        response = await s3.list_objects_v2(
            Bucket=s3_uri.bucket,
            Prefix=s3_uri.key,
            MaxKeys=1,
        )
        # Empty prefix returns successfully with no Contents key
        return "Contents" in response and len(response["Contents"]) > 0


@task(cache_policy=NONE)
async def get_new_versions(
    current_df: pl.LazyFrame,
    existing_ids: pl.LazyFrame | None,
) -> pl.DataFrame:
    """
    Get new versions not yet synced.

    Concepts whose content-based ID doesn't exist in previous state.
    The ID is a hash of the concept's content, so any change creates a new ID.
    """
    if existing_ids is None:
        return current_df.collect()

    return current_df.join(
        existing_ids,
        on="id",
        how="anti",  # Anti-join: rows in current DF NOT in existing IDs
    ).collect()


@task(persist_result=False)
async def load_concepts(
    wikibase_auth: WikibaseAuth,
    wikibase_cache_path: Path | None,  # Path to a JSONL file
    wikibase_cache_save_if_missing: bool,
) -> list[Concept]:
    """
    Load concepts.

    Either from a Wikibase cache or from S3.
    """
    logger = get_logger()

    wikibase = WikibaseSession(
        username=wikibase_auth.username,
        password=str(wikibase_auth.password),
        url=str(wikibase_auth.url),
    )

    if wikibase_cache_path and wikibase_cache_path.exists():
        logger.info(f"loading concepts from cache: {wikibase_cache_path}")
        concepts = []
        with open(wikibase_cache_path, "r") as f:
            for line in f:
                concepts.append(Concept.model_validate_json(line))
        logger.info(f"loaded {len(concepts)} concepts from cache")
    else:
        logger.info("getting concepts from Wikibase")
        concepts = await wikibase.get_concepts_async(limit=None)

        logger.info(f"got {len(concepts)} concepts")

        # Save to cache
        if wikibase_cache_path and wikibase_cache_save_if_missing:
            logger.info(f"saving concepts to cache: {wikibase_cache_path}")
            wikibase_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(wikibase_cache_path, "w") as f:
                for concept in concepts:
                    f.write(concept.model_dump_json() + "\n")
            logger.info(f"saved {len(concepts)} concepts to cache")

    return concepts


@flow(
    persist_result=False,
    on_failure=[SlackNotify.message],
    on_crashed=[SlackNotify.message],
)
async def sync_concepts(
    aws_env: AwsEnv | None = None,
    wikibase_auth: WikibaseAuth | None = None,
    wikibase_cache_path: Path | None = None,
    wikibase_cache_save_if_missing: bool = False,
    concepts_archive_path: Path | S3Uri | None = None,
):
    """
    Sync new, or all, concepts' versions from Wikibase to S3.

    The sync state is stored in data frames in S3. If there's no state
    so far, a new data frame is written, and all concepts are synced.
    If there is existing state, then only the new concepts or new
    versions of existing concepts are synced.

    The side-effects are data frames in S3 and documents inserted ino
    Vespa.

    If no WikibaseAuth is passed, credentials are fetched from AWS SSM.

    If no AWS env. is passed, it's read from the environment.

    You can conditionally use a Wikibase cache, though this is
    currently just for local runs.
    """
    logger = get_logger()

    if aws_env is None:
        aws_env = AwsEnv(os.environ["AWS_ENV"])

    if wikibase_auth is None:
        wikibase_password = SecretStr(get_aws_ssm_param(WIKIBASE_PASSWORD_SSM_NAME))
        wikibase_username = get_aws_ssm_param(WIKIBASE_USERNAME_SSM_NAME)
        wikibase_url = get_aws_ssm_param(WIKIBASE_URL_SSM_NAME)
        # Set as env var so Concept.wikibase_url property can access it
        os.environ["WIKIBASE_URL"] = wikibase_url
        wikibase_auth = WikibaseAuth(
            username=wikibase_username,
            password=wikibase_password,
            url=AnyHttpUrl(wikibase_url),
        )

    if concepts_archive_path is None:
        concepts_archive_path = S3Uri(
            bucket=f"cpr-{aws_env.value}-data-pipeline-cache",
            key="wikibase_concepts",
        )

    concepts = await load_concepts(
        wikibase_auth,
        wikibase_cache_path,
        wikibase_cache_save_if_missing,
    )

    logger.info("converting to dataframe")
    current_df = concepts_to_dataframe(concepts).lazy()
    logger.info("converted to dataframe")

    region = "eu-west-2" if aws_env == AwsEnv.sandbox else "eu-west-1"

    credential_provider: pl.CredentialProvider | None = None
    storage_options: dict[str, str] | None = None
    if isinstance(concepts_archive_path, S3Uri):
        credential_provider = pl.CredentialProviderAWS(region_name=region)
        storage_options = {
            "region": region,
            "default-region": region,
        }

    # Check if archive(s) exists before trying to scan it
    logger.info("checking for existing archive(s)")
    match concepts_archive_path:
        case S3Uri():
            archives_exist = await s3_prefix_has_objects(
                concepts_archive_path,
                region,
                aws_env,
            )
        case Path():
            archives_exist = concepts_archive_path.exists()
    logger.info(f"found existing archive(s): {archives_exist}")

    # Load previous state from all Parquet files if archive exists
    existing_ids: pl.LazyFrame | None = None
    if archives_exist:
        logger.info("loading existing ID(s) from archive(s)")
        parquet_pattern = f"{concepts_archive_path}/*.parquet"
        existing_ids = (
            pl.scan_parquet(
                parquet_pattern,
                credential_provider=credential_provider,
                storage_options=storage_options,
            )
            .select("id")
            .unique()
        )

    logger.info("getting new versions")
    new_versions = await get_new_versions(
        current_df,
        existing_ids,
    )

    logger.info(f"new versions found: {new_versions}")

    if not len(new_versions):
        return

    append_path: str | Path | None = None

    # Append new versions of concepts with timestamp-based filename
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    obj_name = f"concepts_{timestamp}.parquet"

    match concepts_archive_path:
        case S3Uri():
            append_path = f"{concepts_archive_path}/{obj_name}"
        case Path():
            append_path = concepts_archive_path / obj_name

    logger.info(f"appending {len(new_versions)} new versions to {append_path}")
    new_versions.write_parquet(
        append_path,
        credential_provider=credential_provider,
        storage_options=storage_options,
    )
    logger.info(f"successfully appended {len(new_versions)} rows to dataframe")
