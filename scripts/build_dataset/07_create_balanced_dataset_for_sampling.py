"""
Create a balanced subset of the combined dataset for use in downstream tasks.

This script takes the output of the previous scripts
(data/processed/combined_dataset.feather) and creates a balanced sample of passages
based on three attributes:
- Whether the text was translated
- The World Bank region of the document
- The corpus type of the document (eg litigation, corporate disclosures, etc)

The dataset will be saved to `data/processed/balanced_dataset_for_sampling.feather` for
use in downstream tasks.

Note that the shape of the original dataset might mean that a perfectly balanced sample
won't be possible, but in theory, the sampled dataset should be _better_ across
those dimensions than the original dataset.
"""

import pandas as pd
from rich.console import Console

from scripts.config import processed_data_dir
from src.sampling import create_balanced_sample

console = Console()

with console.status("🚚 Loading the combined dataset") as status:
    combined_df = pd.read_feather(processed_data_dir / "combined_dataset.feather")
console.log(f"✅ Combined dataset loaded with {len(combined_df)} rows")
columns = ["translated", "world_bank_region", "document_metadata.corpus_type_name"]

with console.status("🧪 Sampling a balanced dataset from the combined dataset"):
    balanced_sample_dataframe: pd.DataFrame = create_balanced_sample(
        df=combined_df,
        sample_size=25_000,
        on_columns=columns,
    )

console.log(f"✅ Sampled a new dataset with {len(balanced_sample_dataframe)} rows")

console.log("📊 Value counts for the balanced dataset:", end="\n\n")
for column in columns:
    console.log(balanced_sample_dataframe[column].value_counts(), end="\n\n")

console.log(
    "❗ Note that the counts above are probably not perfectly balanced, but should be "
    "much closer than the original dataset.",
)

# save the sample_df
balanced_dataset_path = processed_data_dir / "balanced_dataset_for_sampling.feather"
balanced_sample_dataframe.to_feather(balanced_dataset_path)
console.log(f"💾 Saved the balanced dataset to {balanced_dataset_path}")
