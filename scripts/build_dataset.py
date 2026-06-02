import json
import re
from typing import Annotated

import pandas as pd
import snowflake.connector
import typer
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    load_pem_private_key,
)
from rich.console import Console

from knowledge_graph.config import processed_data_dir
from knowledge_graph.geography import iso_to_world_bank_region
from knowledge_graph.sampling import create_balanced_sample
from knowledge_graph.utils import get_logger

app = typer.Typer()
console = Console(highlight=False)


def get_world_bank_region(geo_array):
    """Extract world bank region from geography array."""
    try:
        if geo_array is None:
            return None
        if hasattr(geo_array, "tolist"):
            geo_list = geo_array.tolist()
        elif isinstance(geo_array, (list, tuple)):
            geo_list = geo_array
        else:
            return None

        # Check if we have any elements
        if not geo_list:
            return None

        # Get the first ISO code
        iso_code = geo_list[0]
        region = iso_to_world_bank_region.get(iso_code)

        return region

    except (AttributeError, IndexError, KeyError, TypeError):
        return None


def _connect_to_snowflake(
    snowflake_user: str | None,
    snowflake_private_key: str | None,
    snowflake_account: str | None,
):
    """
    Connect to Snowflake.

    When explicit credentials are supplied (cloud/ECS path), uses key-pair
    authentication with the DbtBot service account. Falls back to
    connection_name="local_dev" for local development.
    """
    if snowflake_user and snowflake_private_key and snowflake_account:
        private_key = load_pem_private_key(
            snowflake_private_key.encode(), password=None
        )
        private_key_bytes = private_key.private_bytes(
            encoding=Encoding.DER,
            format=PrivateFormat.PKCS8,
            encryption_algorithm=NoEncryption(),
        )
        return snowflake.connector.connect(
            account=snowflake_account,
            user=snowflake_user,
            private_key=private_key_bytes,
        )

    # Local development fallback reads from ~/.snowflake/config.toml
    try:
        return snowflake.connector.connect(connection_name="local_dev")
    except snowflake.connector.errors.Error as e:
        console.log(
            "[red]Failed to connect to Snowflake.[/red] "
            f"Error: {e!r} "
            "Ensure you have a config.toml generated. You can find one in your Snowflake account settings. "
            "See https://docs.snowflake.com/en/developer-guide/snowflake-cli/connecting/configure-connections#define-connections"
        )
        raise


def run_build_dataset(
    n: int = 10000,
    corpus_type: str | None = None,
    snowflake_user: str | None = None,
    snowflake_private_key: str | None = None,
    snowflake_account: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build datasets from Snowflake. Returns (combined_df, sampled_df).

    combined_df: all English passages from the corpus (no row limit).
    sampled_df:  a balanced subset of n rows, stratified by geography,
                 corpus type, and translation status.

    The caller is responsible for writing the output — this function does
    not touch the filesystem or S3.
    """
    presample_size = n * 20
    minimum_text_chars = 20

    corpus_filter = ""
    if corpus_type:
        corpus_filter = f"AND d.METADATA_CORPUS_TYPE_NAME = '{corpus_type}'"

    con = _connect_to_snowflake(
        snowflake_user, snowflake_private_key, snowflake_account
    )
    cur = con.cursor()

    full_query = f"""
    SELECT
        p.CONTENT AS text_block_text,
        p.content_type AS text_block_type,
        d.DOCUMENT_ID,
        d.content_type AS document_content_type,
        d.document_name AS document_name,
        d.document_slug AS document_slug,
        d.METADATA_FAMILY_SLUG AS family_slug,
        d.TRANSLATED AS document_metadata_translated,
        d.METADATA_CORPUS_TYPE_NAME AS document_metadata_corpus_type_name,
        d.METADATA_GEOGRAPHIES AS document_metadata_geographies,
        d.PUBLISHED_DATE AS document_metadata_publication_ts
    FROM PRODUCTION.PUBLISHED.PIPELINE_DOCUMENTS_V1 d
    JOIN PRODUCTION.PUBLISHED.PIPELINE_PASSAGES_V2 p
        ON d.DOCUMENT_ID = p.DOCUMENT_ID
    WHERE p.LANGUAGE = 'en'
      AND p.CONTENT IS NOT NULL
      AND LENGTH(p.CONTENT) > {minimum_text_chars}
      {corpus_filter}
    """

    cur.execute(full_query)
    df_full = cur.fetch_pandas_all()

    presample_query = f"""
    WITH filtered_data AS (
        SELECT
            p.CONTENT AS text_block_text,
            p.content_type AS text_block_type,
            d.DOCUMENT_ID,
            d.content_type AS document_content_type,
            d.document_name AS document_name,
            d.document_slug AS document_slug,
            d.METADATA_FAMILY_SLUG AS family_slug,
            d.TRANSLATED AS document_metadata_translated,
            d.METADATA_CORPUS_TYPE_NAME AS document_metadata_corpus_type_name,
            d.METADATA_GEOGRAPHIES AS document_metadata_geographies,
            d.PUBLISHED_DATE AS document_metadata_publication_ts,
            ROW_NUMBER() OVER (
                PARTITION BY
                    d.METADATA_CORPUS_TYPE_NAME,
                    d.TRANSLATED
                ORDER BY RANDOM()
            ) AS rn
        FROM PRODUCTION.PUBLISHED.PIPELINE_DOCUMENTS_V1 d
        JOIN PRODUCTION.PUBLISHED.PIPELINE_PASSAGES_V2 p
            ON d.DOCUMENT_ID = p.DOCUMENT_ID
        WHERE p.LANGUAGE = 'en'
          AND p.CONTENT IS NOT NULL
          AND LENGTH(p.CONTENT) > {minimum_text_chars}
          {corpus_filter}
    ),
    stratified_sample AS (
        SELECT *
        FROM filtered_data
        WHERE rn <= (
            SELECT {presample_size} / NULLIF(COUNT(DISTINCT CONCAT(document_metadata_corpus_type_name, document_metadata_translated)), 0)
            FROM filtered_data
        )
    )
    SELECT
        text_block_text,
        text_block_type,
        DOCUMENT_ID,
        document_content_type,
        document_name,
        document_slug,
        family_slug,
        document_metadata_translated,
        document_metadata_corpus_type_name,
        document_metadata_geographies,
        document_metadata_publication_ts
    FROM stratified_sample
    ORDER BY RANDOM()
    LIMIT {presample_size}
    """

    cur.execute(presample_query)
    df = cur.fetch_pandas_all()
    con.close()

    logger = get_logger()
    logger.info(f"✅ Full dataset: {len(df_full):,} rows, presample: {len(df):,} rows")

    rename_cols = {
        "TEXT_BLOCK_TEXT": "text_block.text",
        "TEXT_BLOCK_TYPE": "text_block.type",
        "DOCUMENT_ID": "document_id",
        "DOCUMENT_CONTENT_TYPE": "document_content_type",
        "DOCUMENT_NAME": "document_name",
        "DOCUMENT_SLUG": "document_slug",
        "FAMILY_SLUG": "family_slug",
        "DOCUMENT_METADATA_TRANSLATED": "document_metadata.translated",
        "DOCUMENT_METADATA_CORPUS_TYPE_NAME": "document_metadata.corpus_type_name",
        "DOCUMENT_METADATA_GEOGRAPHIES": "document_metadata.geographies",
        "DOCUMENT_METADATA_PUBLICATION_TS": "document_metadata.publication_ts",
    }
    df_full = df_full.rename(columns=rename_cols)
    df = df.rename(columns=rename_cols)

    def parse_geographies(x):
        return json.loads(x) if isinstance(x, str) else x

    df_full["document_metadata.geographies"] = df_full[
        "document_metadata.geographies"
    ].apply(parse_geographies)
    df["document_metadata.geographies"] = df["document_metadata.geographies"].apply(
        parse_geographies
    )

    logger.info("🌍 Adding world bank region metadata")
    df_full["world_bank_region"] = df_full["document_metadata.geographies"].map(
        get_world_bank_region
    )
    df["world_bank_region"] = df["document_metadata.geographies"].map(
        get_world_bank_region
    )
    logger.info("✅ Added world bank region metadata")

    df_combined = df_full.rename(
        columns={
            col: col.replace("document_metadata.", "")
            for col in df.columns
            if col != "document_metadata.corpus_type_name"
            and col.startswith("document_metadata.")
        }
    )

    balance_columns = [
        "world_bank_region",
        "document_metadata.translated",
    ]
    if not corpus_type:
        balance_columns.insert(1, "document_metadata.corpus_type_name")

    df_balanced = create_balanced_sample(
        df=df,
        sample_size=n,
        on_columns=balance_columns,
    )

    df_balanced.columns = [
        col
        if col == "document_metadata.corpus_type_name"
        else col.replace("document_metadata.", "")
        for col in df_balanced.columns
    ]

    return df_combined, df_balanced


@app.command()
def build_dataset(
    n: Annotated[
        int, typer.Option(help="Target number of samples in the final dataset")
    ] = 10000,
    corpus_type: Annotated[
        str | None,
        typer.Option(
            help="Filter to a specific corpus type (i.e., 'Litigation', 'Laws and Policies', 'Intl. agreements', 'Reports', 'AF', 'GEF', 'CIF', 'GCF')"
        ),
    ] = None,
):
    """
    Build a balanced, sampled dataset from Snowflake and save locally.

    Outputs combined_dataset.feather (full corpus) and sampled_dataset.feather
    (balanced n-row subset) to data/processed/. Uses local Snowflake credentials
    from ~/.snowflake/config.toml.

    In production, use the Prefect flow (flows/build_dataset_flow.py) which
    writes to S3 instead.
    """
    if corpus_type:
        console.log(f"Filtering for corpus type: {corpus_type}")

    corpus_suffix = ""
    if corpus_type:
        normalized = re.sub(r"[^\w\s]", "", corpus_type.lower()).replace(" ", "_")
        corpus_suffix = f"_{normalized}"

    console.log("❄️  Connecting to Snowflake")
    combined_df, sampled_df = run_build_dataset(n=n, corpus_type=corpus_type)

    combined_path = processed_data_dir / f"combined_dataset{corpus_suffix}.feather"
    sampled_path = processed_data_dir / f"sampled_dataset{corpus_suffix}.feather"

    combined_df.to_feather(combined_path)
    console.log(
        f"✅ Saved full filtered dataset ({len(combined_df):,} rows) to {combined_path}"
    )

    console.log("📊 Value counts for the balanced dataset:\n")
    for column in [
        "world_bank_region",
        "document_metadata.corpus_type_name",
        "translated",
    ]:
        if column in sampled_df.columns:
            vc = sampled_df[column].value_counts()
            console.log(f"  {column}:")
            for val, cnt in vc.items():
                console.log(f"    {val}: {cnt:,}")
            console.log("")

    console.log(f"💾 Saving the dataset to {sampled_path}")
    sampled_df.to_feather(sampled_path)
    console.log(f"✅ Saved the dataset to {sampled_path}")


if __name__ == "__main__":
    app()
