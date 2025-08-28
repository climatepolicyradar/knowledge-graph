from typing import Annotated

import duckdb
import typer
from datasets import Dataset, disable_progress_bars, load_dataset
from rich.console import Console

from src.config import processed_data_dir
from src.geography import iso_to_world_bank_region
from src.sampling import create_balanced_sample

disable_progress_bars()

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
):
    """
    Build a balanced, sampled dataset from the HuggingFace climate document corpus.

    This script downloads the ClimatePolicyRadar/all-document-text-data-weekly dataset,
    filters for English text, applies stratified pre-sampling for efficiency, adds
    geographic metadata (World Bank regions), and creates a balanced sample across
    multiple dimensions (geography, corpus type, translation status).
    """
    dataset_name = "ClimatePolicyRadar/all-document-text-data-weekly"

    console.log(f"🚚 Loading [white]{dataset_name}[/white]")

    dataset = load_dataset(dataset_name, split="train", streaming=False)
    if not isinstance(dataset, Dataset):
        raise TypeError(
            f"Expected Dataset, got {type(dataset)}. This script requires a non-streaming dataset."
        )

    console.log(f"✅ The raw dataset has {len(dataset)} rows")

    # Convert the dataset to an Arrow table for DuckDB querying
    arrow_table = dataset.data.table
    con = duckdb.connect()
    con.register("dataset", arrow_table)

    # Use DuckDB for super fast filtering + intelligent pre-sampling
    console.log("🦆 Filtering for English text with length > 20 chars, using DuckDB")

    # Pre-sample to manageable size using stratified sampling (20x target size)
    presample_size = n * 20
    console.log(
        f"🎲 Pre-sampling to ~{presample_size} rows for efficient balanced sampling"
    )

    query = f"""
    WITH filtered_data AS (
        SELECT 
            "text_block.text",
            "document_id", 
            "document_metadata.translated",
            "document_metadata.corpus_type_name",
            "document_metadata.geographies",
            -- Add row numbers for stratified sampling
            ROW_NUMBER() OVER (
                PARTITION BY 
                    "document_metadata.corpus_type_name",
                    "document_metadata.translated"
                ORDER BY RANDOM()
            ) as rn
        FROM dataset
        WHERE "text_block.language" = 'en' 
          AND "text_block.text" IS NOT NULL 
          AND length("text_block.text") > 20
    ),
    stratified_sample AS (
        -- Take roughly equal samples from each corpus_type + translated combination
        SELECT *
        FROM filtered_data 
        WHERE rn <= (SELECT {presample_size} / COUNT(DISTINCT "document_metadata.corpus_type_name" || "document_metadata.translated") FROM filtered_data)
    )
    SELECT 
        "text_block.text",
        "document_id", 
        "document_metadata.translated",
        "document_metadata.corpus_type_name",
        "document_metadata.geographies"
    FROM stratified_sample
    ORDER BY RANDOM()
    LIMIT {presample_size}
    """

    df = con.execute(query).df()
    console.log(f"✅ Filtered down to {len(df)} rows")

    console.log("🌍 Adding world bank region metadata")
    df["world_bank_region"] = df["document_metadata.geographies"].map(
        get_world_bank_region
    )
    console.log("✅ Added world bank region metadata")

    console.log(f"🧪 Sampling a balanced subset of {n} rows from the filtered dataset")
    df_balanced = create_balanced_sample(
        df=df,
        sample_size=n,
        on_columns=[
            "world_bank_region",
            "document_metadata.corpus_type_name",
            "document_metadata.translated",
        ],
    )

    # remove the document_metadata prefix from the column names
    df_balanced.columns = [
        col.replace("document_metadata.", "") for col in df_balanced.columns
    ]

    console.log("📊 Value counts for the balanced dataset:", end="\n\n")
    for column in ["world_bank_region", "corpus_type_name", "translated"]:
        console.log(df_balanced[column].value_counts(), end="\n\n")

    dataset_path = processed_data_dir / "sampled_dataset.feather"
    console.log("💾 Saving the dataset to feather")
    df_balanced.to_feather(dataset_path)
    console.log(f"✅ Saved the dataset to {dataset_path}")


if __name__ == "__main__":
    app()
