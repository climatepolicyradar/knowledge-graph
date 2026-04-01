import json
import re
from typing import Annotated

import snowflake.connector
import typer
from rich.console import Console

from knowledge_graph.config import processed_data_dir
from knowledge_graph.geography import iso_to_world_bank_region
from knowledge_graph.sampling import create_balanced_sample

app = typer.Typer()
console = Console(highlight=False)


def get_world_bank_region(geo_array):
    """Extract world bank region from geography array."""
    try:
        # Handle None/empty cases
        if geo_array is None:
            return None

        # Convert numpy array to list if needed
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


@app.command()
def build_dataset(
    n: Annotated[
        int, typer.Option(help="Target number of samples in the final dataset")
    ] = 10000,
    corpus_type: Annotated[
        str | None,
        typer.Option(
            help="Filter to a specific corpus type (i.e., 'Litigation', 'Laws and Policies', 'Intl. agreements', 'Reports', 'AF', 'GEF' , 'CIF', 'GCF'')"
        ),
    ] = None,
):
    """
    Build a balanced, sampled dataset from Snowflake climate document corpus.

    This script queries Snowflake for English text passages, applies stratified
    pre-sampling for efficiency, adds geographic metadata (World Bank regions),
    and creates a balanced sample across multiple dimensions (geography, corpus
    type, translation status).
    """
    if corpus_type:
        console.log(f"🔍 Filtering for corpus type: [white]{corpus_type}[/white]")

    # Pre-sample to manageable size using stratified sampling (20x target size)
    presample_size = n * 20
    minimum_text_chars = 20

    # Build the corpus type filter clause
    corpus_filter = ""
    if corpus_type:
        corpus_filter = f"AND d.METADATA_CORPUS_TYPE_NAME = '{corpus_type}'"

    console.log("❄️  Connecting to Snowflake")
    try:
        # See https://docs.snowflake.com/en/developer-guide/snowflake-cli/connecting/configure-connections#define-connections
        # There's a config.toml generator in Snowflake's account settings.
        con = snowflake.connector.connect(connection_name="local_dev")
    except Exception as e:
        console.log(
            "[red]Failed to connect to Snowflake.[/red] "
            f"Error: {e!r} "
            "Ensure you have a config.toml generated. You can find one in your Snowflake account settings."
            "See https://docs.snowflake.com/en/developer-guide/snowflake-cli/connecting/configure-connections#define-connections "
        )
        raise
    cur = con.cursor()

    # First, fetch the full filtered dataset for combined_dataset
    console.log("📥 Fetching full filtered dataset from Snowflake")
    full_query = f"""
    SELECT
        p.CONTENT AS text_block_text,
        d.DOCUMENT_ID,
        d.TRANSLATED AS document_metadata_translated,
        d.METADATA_CORPUS_TYPE_NAME AS document_metadata_corpus_type_name,
        d.METADATA_GEOGRAPHIES AS document_metadata_geographies
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

    # Then, fetch the stratified presample for balanced sampling
    console.log(f"🎲 Querying with stratified pre-sampling to ~{presample_size:,} rows")
    presample_query = f"""
    WITH filtered_data AS (
        SELECT
            p.CONTENT AS text_block_text,
            d.DOCUMENT_ID,
            d.TRANSLATED AS document_metadata_translated,
            d.METADATA_CORPUS_TYPE_NAME AS document_metadata_corpus_type_name,
            d.METADATA_GEOGRAPHIES AS document_metadata_geographies,
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
        DOCUMENT_ID,
        document_metadata_translated,
        document_metadata_corpus_type_name,
        document_metadata_geographies
    FROM stratified_sample
    ORDER BY RANDOM()
    LIMIT {presample_size}
    """

    cur.execute(presample_query)
    df = cur.fetch_pandas_all()
    con.close()

    rename_cols = {
        "TEXT_BLOCK_TEXT": "text_block.text",
        "DOCUMENT_ID": "document_id",
        "DOCUMENT_METADATA_TRANSLATED": "document_metadata.translated",
        "DOCUMENT_METADATA_CORPUS_TYPE_NAME": "document_metadata.corpus_type_name",
        "DOCUMENT_METADATA_GEOGRAPHIES": "document_metadata.geographies",
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

    console.log(f"✅ Full dataset: {len(df_full):,} rows, presample: {len(df):,} rows")

    console.log("🌍 Adding world bank region metadata")
    df_full["world_bank_region"] = df_full["document_metadata.geographies"].map(
        get_world_bank_region
    )
    df["world_bank_region"] = df["document_metadata.geographies"].map(
        get_world_bank_region
    )
    console.log("✅ Added world bank region metadata")

    # Build output filename suffix with optional corpus type
    if corpus_type:
        # Normalize: lowercase, spaces to underscores, remove punctuation
        normalized = re.sub(r"[^\w\s]", "", corpus_type.lower()).replace(" ", "_")
        corpus_suffix = f"_{normalized}"
    else:
        corpus_suffix = ""

    # Save the full filtered dataset as combined_dataset.feather
    df_combined = df_full.rename(
        columns={
            col: col.replace("document_metadata.", "")
            for col in df.columns
            if col != "document_metadata.corpus_type_name"
            and col.startswith("document_metadata.")
        }
    )
    combined_path = processed_data_dir / f"combined_dataset{corpus_suffix}.feather"
    df_combined.to_feather(combined_path)
    console.log(
        f"✅ Saved full filtered dataset ({len(df_combined):,} rows) to {combined_path}"
    )

    # Adjust balancing columns based on whether we're filtering by corpus type
    balance_columns = [
        "world_bank_region",
        "document_metadata.translated",
    ]
    if not corpus_type:
        balance_columns.insert(1, "document_metadata.corpus_type_name")

    console.log(
        f"🧪 Sampling a balanced subset of {n:,} rows from the filtered dataset"
    )
    df_balanced = create_balanced_sample(
        df=df,
        sample_size=n,
        on_columns=balance_columns,
    )

    # Remove the document_metadata prefix from column names, except corpus_type_name
    df_balanced.columns = [
        col
        if col == "document_metadata.corpus_type_name"
        else col.replace("document_metadata.", "")
        for col in df_balanced.columns
    ]

    console.log("📊 Value counts for the balanced dataset:", end="\n\n")
    for column in [
        "world_bank_region",
        "document_metadata.corpus_type_name",
        "translated",
    ]:
        vc = df_balanced[column].value_counts()
        console.log(f"  {column}:")
        for val, cnt in vc.items():
            console.log(f"    {val}: {cnt:,}")
        console.log("")

    dataset_path = processed_data_dir / f"sampled_dataset{corpus_suffix}.feather"
    console.log("💾 Saving the dataset to feather")
    df_balanced.to_feather(dataset_path)
    console.log(f"✅ Saved the dataset to {dataset_path}")


if __name__ == "__main__":
    app()
