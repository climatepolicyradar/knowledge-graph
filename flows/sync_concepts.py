import os
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
from cpr_sdk.ssm import get_aws_ssm_param
from prefect import flow, task
from pydantic import AnyHttpUrl, SecretStr

from flows.utils import (
    S3Uri,
    SlackNotify,
)
from knowledge_graph.cloud import AwsEnv
from knowledge_graph.concept import Concept
from knowledge_graph.utils import get_logger
from knowledge_graph.wikibase import WikibaseAuth, WikibaseSession

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
    Sync all concepts from Wikibase to S3, as a full snapshot.

    Each run writes every concept currently in Wikibase to a timestamped
    Parquet file in a snapshots location, then overwrites `latest.parquet`
    in the archive location with the same data. Downstream (Snowflake)
    only reads the archive location, so it only ever sees concepts that
    currently exist. Snapshots are never deleted by this flow, so
    they can be used to recover from failures and to debug; expire them
    with a lifecycle policy instead.

    The snapshots location is the archive location with a `_snapshots`
    suffix, e.g. `wikibase_concepts` and `wikibase_concepts_snapshots`.

    The side-effects are data frames in S3.

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
    current_df = concepts_to_dataframe(concepts)
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

    if not concepts:
        # Don't replace latest with nothing if Wikibase returned no concepts
        raise ValueError("no concepts returned, refusing to replace latest")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    match concepts_archive_path:
        case S3Uri():
            snapshots_dir = S3Uri(
                bucket=concepts_archive_path.bucket,
                key=f"{concepts_archive_path.key}_snapshots",
            )
        case Path():
            snapshots_dir = concepts_archive_path.with_name(
                f"{concepts_archive_path.name}_snapshots"
            )
            snapshots_dir.mkdir(parents=True, exist_ok=True)
            concepts_archive_path.mkdir(parents=True, exist_ok=True)

    snapshot_path = f"{snapshots_dir}/concepts_{timestamp}.parquet"
    latest_path = f"{concepts_archive_path}/latest.parquet"

    # Write the snapshot first, so if overwriting latest fails the
    # snapshot is kept and latest still holds the previous export.
    logger.info(f"writing snapshot of {len(current_df)} rows to {snapshot_path}")
    current_df.write_parquet(
        snapshot_path,
        credential_provider=credential_provider,
        storage_options=storage_options,
    )

    logger.info(f"overwriting {latest_path} with {len(current_df)} rows")
    current_df.write_parquet(
        latest_path,
        credential_provider=credential_provider,
        storage_options=storage_options,
    )
    logger.info(f"successfully wrote {len(current_df)} rows to {latest_path}")
