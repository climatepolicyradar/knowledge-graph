import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import polars as pl
from cpr_sdk.models.search import Concept as VespaConcept
from cpr_sdk.models.search import WikibaseId as VespaWikibaseId
from cpr_sdk.search_adaptors import VespaSearchAdapter
from cpr_sdk.ssm import get_aws_ssm_param
from prefect import flow, task
from prefect.artifacts import acreate_table_artifact
from prefect.cache_policies import NONE
from pydantic import AnyHttpUrl, SecretStr
from tenacity import RetryError
from vespa.application import VespaAsync
from vespa.io import VespaResponse

from flows.boundary import get_vespa_search_adapter_from_aws_secrets
from flows.result import Err, Error, Ok, Result
from flows.utils import JsonDict, S3Uri, SlackNotify, get_logger, total_milliseconds
from knowledge_graph.cloud import AwsEnv
from knowledge_graph.concept import Concept
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
            concept.model_dump(exclude={"labelled_passages"}, mode="python")
            for concept in concepts
        ]
    )

    # Cast Null columns to proper types for consistent schema Polars
    # infers Null type when all values are None, but we want explicit
    # types.
    schema_casts = {
        "description": pl.Utf8,
        "definition": pl.Utf8,
        "recursive_subconcept_of": pl.List(pl.Utf8),
        "recursive_has_subconcept": pl.List(pl.Utf8),
    }

    for col, dtype in schema_casts.items():
        if col in df.columns and df[col].dtype == pl.Null:
            df = df.with_columns(pl.col(col).cast(dtype))

    # Add sync timestamp to track when this version was synced
    df = df.with_columns(pl.lit(datetime.now(timezone.utc)).alias("synced_at"))

    return df


def dataframe_to_concepts(df: pl.DataFrame) -> list[Concept]:
    """
    Convert a Polars DataFrame from data lake storage back to Concept objects.

    Inverse of concepts_to_dataframe(). Excludes the synced_at column
    and any other columns not in the Concept model.

    Args:
        df: DataFrame with concept data

    Returns:
        list[Concept]: List of reconstructed Concept objects
    """
    if df.is_empty():
        return []

    # Get Concept model fields to filter DataFrame columns
    concept_fields = set(Concept.model_fields.keys())

    # Remove synced_at and any other non-Concept columns
    df_filtered = df.select([col for col in df.columns if col in concept_fields])

    concepts = [Concept.model_validate(row) for row in df_filtered.to_dicts()]

    return concepts


@task(
    persist_result=False,
    task_run_name="update-concept-{kg_concept.wikibase_id}-{kg_concept.id}",
)
async def update_concept_in_vespa(
    kg_concept: Concept,
    vespa_connection_pool: VespaAsync,
) -> Result[Concept, Error]:
    logger = get_logger()

    logger.info(
        f"updating data in Vespa for concept id={kg_concept.id}, wikibase_id={kg_concept.wikibase_id}, wikibase_revision={kg_concept.wikibase_revision}, preferred_label={kg_concept.preferred_label}"
    )

    try:
        # Convert KG Concept to SDK Concept
        sdk_concept = VespaConcept(
            id=str(kg_concept.id),
            wikibase_id=VespaWikibaseId(str(kg_concept.wikibase_id)),
            wikibase_revision=kg_concept.wikibase_revision,  # pyright: ignore[reportArgumentType]
            wikibase_url=AnyHttpUrl(kg_concept.wikibase_url),
            preferred_label=kg_concept.preferred_label,
            description=kg_concept.description or None,
            definition=kg_concept.definition or None,
            alternative_labels=kg_concept.alternative_labels,
            negative_labels=kg_concept.negative_labels,
            subconcept_of=[str(id) for id in kg_concept.subconcept_of],
            has_subconcept=[str(id) for id in kg_concept.has_subconcept],
            related_concepts=[str(id) for id in kg_concept.related_concepts],
            recursive_subconcept_of=(
                [str(id) for id in kg_concept.recursive_subconcept_of]
                if kg_concept.recursive_subconcept_of
                else None
            ),
            recursive_has_subconcept=(
                [str(id) for id in kg_concept.recursive_has_subconcept]
                if kg_concept.recursive_has_subconcept
                else None
            ),
            response_raw={},
        )
    except Exception as e:
        logger.error(f"Failed to create VespaConcept for {kg_concept.wikibase_id}: {e}")
        return Err(
            Error(
                msg=f"Failed to create VespaConcept: {e}",
                metadata={"concept": kg_concept},
            )
        )

    try:
        fields = JsonDict(
            sdk_concept.model_dump(
                mode="json",
                exclude={"response_raw"},
            )
        )

        id = f"{kg_concept.wikibase_id}.{kg_concept.id}"

        path = vespa_connection_pool.app.get_document_v1_path(
            id=id,
            schema="concept",
            namespace="doc_search",
            group=None,
        )

        logger.info(f"using path {path} with fields {fields}")

        response: VespaResponse = await vespa_connection_pool.update_data(
            schema="concept",
            namespace="doc_search",
            data_id=id,
            create=True,
            fields=fields,
        )

        if not response.is_successful():
            # `get_json` returns a Dict[1].
            #
            # [1]: https://github.com/vespa-engine/pyvespa/blob/1b42923b77d73666e0bcd1e53431906fc3be5d83/vespa/io.py#L44-L46
            logger.error(
                f"Vespa update failed for {kg_concept.wikibase_id}: {json.dumps(response.get_json())}"
            )
            return Err(
                Error(
                    msg="Vespa update failed",
                    metadata={
                        "response": response,
                        "concept": kg_concept,
                    },
                )
            )

        logger.info("updated Vespa")

        return Ok(kg_concept)
    except Exception as e:
        logger.error(
            f"Unexpected error updating Vespa for {kg_concept.wikibase_id}: {e}"
        )
        return Err(
            Error(
                msg=f"Unexpected error during Vespa update: {e}",
                metadata={"concept": kg_concept},
            )
        )


@task(cache_policy=NONE)
async def get_new_versions(
    current_df: pl.LazyFrame,
    existing_ids: pl.LazyFrame,
    concepts_archive_path: Path | S3Uri,
) -> pl.DataFrame:
    """
    Get new versions not yet synced.

    Concepts whose content-based ID doesn't exist in previous state
    The ID is a hash of the concept's content, so any change creates a
    new ID.
    """
    logger = get_logger()

    try:
        return current_df.join(
            existing_ids,
            on="id",
            how="anti",  # Anti-join: rows in current DF NOT in existing IDs
        ).collect()
    except pl.exceptions.ComputeError as e:
        if "expanded paths were empty" in str(e):
            logger.info(
                f"no existing data at {concepts_archive_path}, all concepts are new"
            )
            return current_df.collect()
        else:
            raise


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

        try:
            concepts = await wikibase.get_concepts_async(limit=None)
        except RetryError as e:
            # RetryError contains unpicklable objects (Future, Lock).
            # Prefect's JSON serialiser does work with it, but I
            # decided to handle it here, this way.
            #
            # Extract the underlying exception and re-raise with a
            # clean message.
            underlying_error = e.last_attempt.exception() if e.last_attempt else None
            error_msg = f"Failed to fetch concepts from Wikibase after retries: {underlying_error}"
            raise RuntimeError(error_msg) from underlying_error

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


async def update_concepts_in_vespa(
    kg_concepts: list[Concept],
    vespa_connection_pool: VespaAsync,
) -> list[Result[Concept, Error]]:
    """
    Update a list of concepts in Vespa.

    Validates and updates each concept, collecting results for both
    successful and failed updates.
    """
    logger = get_logger()
    results: list[Result[Concept, Error]] = []

    logger.info(f"updating {len(kg_concepts)} concepts in Vespa")

    for kg_concept in kg_concepts:
        # Validate concept has required fields
        if not kg_concept.wikibase_revision:
            results.append(
                Err(
                    Error(
                        msg="concept missing Wikibase revision",
                        metadata={"concept": kg_concept},
                    )
                )
            )
            continue

        # Update concept in Vespa
        result = await update_concept_in_vespa(
            kg_concept=kg_concept,
            vespa_connection_pool=vespa_connection_pool,
        )

        results.append(result)

    successes = [r for r in results if isinstance(r, Ok)]
    failures = [r for r in results if isinstance(r, Err)]
    logger.info(
        f"completed Vespa updates: {len(successes)} successful, {len(failures)} failed"
    )

    return results


async def create_vespa_sync_summary_artifact(
    results: list[Result[Concept, Error]],
    parquet_path: str | None,
):
    """Create an artifact with a summary about the Vespa sync run."""

    successes = [r._value for r in results if isinstance(r, Ok)]
    failures = [r._error for r in results if isinstance(r, Err)]

    total_concepts = len(results)
    successful_syncs = len(successes)
    failed_syncs = len(failures)

    if parquet_path:
        parquet_info = f"- **New versions appended at**: `{parquet_path}`"
    else:
        parquet_info = "- **New versions appended**: None (no successful syncs)"

    overview_description = f"""# Vespa Sync Summary

## Overview
- **Total concepts processed**: {total_concepts}
- **Successful syncs**: {successful_syncs}
- **Failed syncs**: {failed_syncs}
{parquet_info}
"""

    concept_details = [
        {
            "Concept ID": str(concept.id),
            "Wikibase ID": str(concept.wikibase_id),
            "Preferred Label": concept.preferred_label,
            "Wikibase Revision": concept.wikibase_revision or "N/A",
            "Status": "✓",
            "Error": "N/A",
        }
        for concept in successes
    ] + [
        {
            "Concept ID": str(
                (error.metadata or {}).get("concept", {}).get("id", "Unknown")
            ),
            "Wikibase ID": str(
                (error.metadata or {}).get("concept", {}).get("wikibase_id", "Unknown")
            ),
            "Preferred Label": str(
                (error.metadata or {})
                .get("concept", {})
                .get("preferred_label", "Unknown")
            ),
            "Wikibase Revision": str(
                (error.metadata or {})
                .get("concept", {})
                .get("wikibase_revision", "N/A")
            ),
            "Status": "✗",
            "Error": (
                f"{error.msg}: {json.dumps((error.metadata or {}).get('response').get_json())}"  # pyright: ignore[reportOptionalMemberAccess]
                if error.metadata and error.metadata.get("response")
                else error.msg
            ),
        }
        for error in failures
    ]

    await acreate_table_artifact(
        key="vespa-sync",
        table=concept_details,
        description=overview_description,
    )


@flow(
    persist_result=False,
    on_failure=[SlackNotify.message],
    on_crashed=[SlackNotify.message],
)
async def wikibase_to_vespa(
    aws_env: AwsEnv | None = None,
    wikibase_auth: WikibaseAuth | None = None,
    vespa_search_adapter: VespaSearchAdapter | None = None,
    wikibase_cache_path: Path | None = None,
    wikibase_cache_save_if_missing: bool = False,
    concepts_archive_path: Path | S3Uri | None = None,
):
    """
    Sync new, or all, concepts' versions from Wikibase to Vespa.

    The sync state is stored in data frames in S3. If there's no state
    so far, a new data frame is written, and all concepts are synced.
    If there is existing state, then only the new concepts or new
    versions of existing concepts are synced.

    New versions are only written in the new data frame if they were
    synced in Vespa. This way you can re-run the flow, without having
    to modify the state, to try again.

    The side-effects are data frames in S3 and documents inserted ino
    Vespa.

    If no Vespa search adapter is passed, then one is gotten from AWS
    parameters.

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

    # Load previous state from all Parquet files
    parquet_pattern = f"{concepts_archive_path}/*.parquet"

    credential_provider = pl.CredentialProviderAWS(
        region_name="eu-west-1",
    )

    logger.info("getting existing versions")
    existing_ids = (
        pl.scan_parquet(
            parquet_pattern,
            credential_provider=credential_provider,
        )
        .select("id")
        .unique()
    )

    logger.info("getting new versions")

    new_versions = await get_new_versions(
        current_df,
        existing_ids,
        concepts_archive_path,
    )

    logger.info(f"new versions found: {new_versions}")

    if not len(new_versions):
        logger.info("no new versions to sync")
        return

    kg_concepts = dataframe_to_concepts(new_versions)

    if not vespa_search_adapter:
        temp_dir = tempfile.TemporaryDirectory()
        vespa_search_adapter = get_vespa_search_adapter_from_aws_secrets(
            cert_dir=temp_dir.name,
            vespa_private_key_param_name="VESPA_PRIVATE_KEY_FULL_ACCESS",
            vespa_public_cert_param_name="VESPA_PUBLIC_CERT_FULL_ACCESS",
            aws_env=aws_env,
        )

    # Update concepts in Vespa using a subflow
    async with vespa_search_adapter.client.asyncio(
        connections=5,
        timeout=httpx.Timeout(VESPA_MAX_TIMEOUT_MS / 1_000),
    ) as vespa_connection_pool:
        results = await update_concepts_in_vespa(
            kg_concepts=kg_concepts,
            vespa_connection_pool=vespa_connection_pool,
        )

    # Filter to only successful Vespa updates
    successful_concepts = [r._value for r in results if isinstance(r, Ok)]

    # Only write Parquet if we have successful syncs
    append_path: str | Path | None = None
    if successful_concepts:
        logger.info(f"successfully synced {len(successful_concepts)} concepts to Vespa")

        # Convert successful concepts back to DataFrame
        successful_df = concepts_to_dataframe(successful_concepts)

        # Append successful concepts with timestamp-based filename
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        obj_name = f"concepts_{timestamp}.parquet"

        match concepts_archive_path:
            case S3Uri():
                append_path = f"{concepts_archive_path}/{obj_name}"
            case Path():
                append_path = concepts_archive_path / obj_name

        logger.info(
            f"appending {len(successful_concepts)} successful syncs to {append_path}"
        )
        successful_df.write_parquet(
            append_path, credential_provider=credential_provider
        )
        logger.info(
            f"successfully appended {len(successful_concepts)} rows to dataframe"
        )
    else:
        logger.warning(
            "no concepts successfully synced to Vespa, skipping Parquet write"
        )

    # Create artifact with results
    await create_vespa_sync_summary_artifact(
        results=results,
        parquet_path=str(append_path) if append_path else None,
    )
