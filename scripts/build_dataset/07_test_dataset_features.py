import pandas as pd
from rich.console import Console

from scripts.config import processed_data_dir

console = Console()

with console.status("Loading combined dataset..."):
    combined_dataset_path = processed_data_dir / "combined_dataset.feather"
    df = pd.read_feather(combined_dataset_path)

# drop rows with empty world_bank_region
df = df[df["world_bank_region"].apply(lambda x: x != "[]")]

# print dataset stats
console.print(df.shape)

# check whether there's a 'translated' value for all rows
translated_counts = df["translated"].value_counts()
console.print(translated_counts)

# check whether there's a world bank region for all rows
world_bank_region_counts = df["world_bank_region"].value_counts()
console.print(world_bank_region_counts)

# check whether the document_metadata.corpus_type_name is present for all rows
corpus_type_name_counts = df["document_metadata.corpus_type_name"].value_counts()
console.print(corpus_type_name_counts)
