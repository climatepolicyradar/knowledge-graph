import asyncio
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
from prefect import flow

from flows.utils import get_logger
from knowledge_graph.cloud import AwsEnv, get_aws_ssm_param
from knowledge_graph.concept import Concept
from knowledge_graph.wikibase import WikibaseSession

# SSM
WIKIBASE_PASSWORD_SSM_NAME = "/Wikibase/Cloud/ServiceAccount/Password"
WIKIBASE_USERNAME_SSM_NAME = "/Wikibase/Cloud/ServiceAccount/Username"
WIKIBASE_URL_SSM_NAME = "/Wikibase/Cloud/URL"


def concepts_to_dataframe(concepts: list[Concept]) -> pl.DataFrame:
    """
    Convert a list of Concepts to a Polars DataFrame for data lake storage.

    Args:
        concepts: List of Concept objects to convert

    Returns:
        pl.DataFrame: A DataFrame with all concepts, ready for Delta Lake write
    """
    df = pl.DataFrame(
        [
            concept.model_dump(exclude={"labelled_passages"}, mode="python")
            for concept in concepts
        ]
    )

    # Cast Null columns to proper types for Delta Lake compatibility
    # Delta Lake doesn't support Null type, which Polars infers when all values are None
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


@flow
async def wikibase_to_vespa():
    logger = get_logger()

    aws_env = AwsEnv.staging

    username = get_aws_ssm_param(
        WIKIBASE_USERNAME_SSM_NAME,
        aws_env=aws_env,
    )
    password = get_aws_ssm_param(
        WIKIBASE_PASSWORD_SSM_NAME,
        aws_env=aws_env,
    )
    url = get_aws_ssm_param(
        WIKIBASE_URL_SSM_NAME,
        aws_env=aws_env,
    )

    wikibase = WikibaseSession(
        username=username,
        password=password,
        url=url,
    )

    # TODO: Remove, as dev only
    # Check for cached concepts to avoid network download
    cache_path = Path("./tmp/concepts_cache.jsonl")
    if cache_path.exists():
        logger.info(f"loading concepts from cache: {cache_path}")
        concepts = []
        with open(cache_path, "r") as f:
            for line in f:
                concepts.append(Concept.model_validate_json(line))
        logger.info(f"loaded {len(concepts)} concepts from cache")
    else:
        logger.info("getting concepts from wikibase")
        concepts = await wikibase.get_concepts_async(limit=None)
        logger.info(f"got {len(concepts)} concepts")

        # Save to cache
        logger.info(f"saving concepts to cache: {cache_path}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            for concept in concepts:
                f.write(concept.model_dump_json() + "\n")
        logger.info(f"saved {len(concepts)} concepts to cache")

    logger.info("converting to dataframe")
    current_df = concepts_to_dataframe(concepts)
    logger.info(f"dataframe shape: {current_df.shape}")

    output_path = Path("./tmp/concepts_delta")

    # Check if Delta Lake table exists (by checking for _delta_log directory)
    delta_log_path = output_path / "_delta_log"
    if not delta_log_path.exists():
        # First run - write initial state
        logger.info(f"no existing data, writing initial state to {output_path}")
        current_df.write_delta(output_path, mode="overwrite")
        logger.info(
            f"successfully wrote {len(current_df)} initial rows to {output_path}"
        )

        # TODO: Insert all into Vespa
        return

    # Load current data state from Delta Lake
    logger.info(f"loading previous state from {output_path}")
    previous_df = pl.read_delta(output_path)
    logger.info(f"found {len(previous_df)} existing rows")

    # Get all existing concept IDs from previous state
    existing_ids = previous_df.select("id").unique()

    # Find new versions: concepts whose content-based ID doesn't exist in previous state
    # The ID is a hash of the concept's content, so any change creates a new ID
    new_versions = current_df.join(
        existing_ids,
        on="id",
        how="anti",  # Anti-join: rows in current_df NOT in existing_ids
    )

    if len(new_versions) > 0:
        # Append only new versions
        logger.info(f"appending {len(new_versions)} new versions to {output_path}")
        logger.info(f"new versions: {new_versions}")
        if False:
            new_versions.write_delta(output_path, mode="append")
        logger.info(f"successfully appended {len(new_versions)} rows")

        # TODO: Insert new versions into Vespa
    else:
        logger.info("no new versions to sync")


if __name__ == "__main__":
    asyncio.run(wikibase_to_vespa())
