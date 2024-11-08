from itertools import cycle, product

import pandas as pd
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TimeRemainingColumn,
)

from scripts.config import processed_data_dir

console = Console()
SAMPLE_SIZE = 1_00

with console.status("🚚 Loading the combined dataset") as status:
    combined_df = pd.read_feather(processed_data_dir / "combined_dataset.feather")
console.log(f"✅ Combined dataset loaded with {len(combined_df)} rows")

translated = combined_df["translated"].unique()
world_bank_regions = [
    value for value in combined_df["world_bank_region"].unique() if value != "None"
]
corpus_type_names = [
    value
    for value in combined_df["document_metadata.corpus_type_name"].unique()
    if value != "None"
]

# get all the combinations of values
combination_values = list(product(world_bank_regions, translated, corpus_type_names))
combination_keys = [
    [
        "world_bank_region",
        "translated",
        "document_metadata.corpus_type_name",
    ]
] * len(combination_values)
combinations = list(
    dict(zip(keys, values))
    for keys, values in zip(combination_keys, combination_values)
)
console.log(f"⚙️ Found {len(combinations)} combinations of values in equity strata")


# we're going to sample (without replacement) from combined_df over each combination
# until we have a sample of sample_size
balanced_sample_dataframe = pd.DataFrame()

# first set up a cache for the rows which match each of our constraints so that we don't
# need to run a fresh query each time, and can instead sample from the cached rows
matching_rows_cache = {}

# set up a progress bar to visualise the sampling process
progress_bar = Progress(
    "[progress.description]{task.description}",
    TaskProgressColumn(),
    BarColumn(),
    MofNCompleteColumn(),
    TimeRemainingColumn(),
    console=console,
    transient=True,
)
progress_task = progress_bar.add_task(
    description="Sampling a more balanced dataset", total=SAMPLE_SIZE
)
progress_bar.start()

# cycle through the combinations until we have a dataset of the desired size
combination_cycle = cycle(combinations)

while len(balanced_sample_dataframe) < SAMPLE_SIZE:
    combination = next(combination_cycle)
    matching_rows = matching_rows_cache.get(str(combination), None)
    if matching_rows is None:
        matching_rows = combined_df[
            (combined_df["world_bank_region"] == combination["world_bank_region"])
            & (combined_df["translated"] == combination["translated"])
            & (
                combined_df["document_metadata.corpus_type_name"]
                == combination["document_metadata.corpus_type_name"]
            )
        ]
        matching_rows_cache[str(combination)] = matching_rows

    if len(matching_rows) == 0:
        continue

    else:
        sampled_row = matching_rows.sample(1)
        matching_rows_cache[str(combination)] = matching_rows.drop(sampled_row.index)
        balanced_sample_dataframe = pd.concat([balanced_sample_dataframe, sampled_row])
        progress_bar.update(progress_task, advance=1)

balanced_sample_dataframe = balanced_sample_dataframe.reset_index(drop=True)

# save the sample_df
balanced_dataset_path = processed_data_dir / "balanced_dataset_for_sampling.feather"
balanced_sample_dataframe.to_feather(balanced_dataset_path)
console.log(f"💾 Saved the balanced dataset to {balanced_dataset_path}")

progress_bar.stop()

console.log(f"✅ Sampled a new dataset with {len(balanced_sample_dataframe)} rows")

console.log("📊 Value counts for the balanced dataset:", end="\n\n")

console.log(balanced_sample_dataframe["translated"].value_counts(), end="\n\n")
console.log(balanced_sample_dataframe["world_bank_region"].value_counts(), end="\n\n")
console.log(
    balanced_sample_dataframe["document_metadata.corpus_type_name"].value_counts()
)
