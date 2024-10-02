"""
Create a canonical dataset for sampling

This script combines the parsed documents with the CPR and GST datasets into a single
dataset from which we can sample for the classifier evaluation. The script also adds
the world bank region metadata to the dataset for stratification.
"""

import json

import pandas as pd
from cpr_sdk.models import (
    BaseDocument,
    BaseParserOutput,
    CPRDocument,
    GSTDocument,
)
from cpr_sdk.models import Dataset as CPRDataset
from rich.console import Console
from rich.progress import track

from scripts.config import interim_data_dir, processed_data_dir
from src.geography import geography_string_to_iso, iso_to_world_bank_region

console = Console()

combined_dataset_path = processed_data_dir / "combined_dataset.feather"

datasets = []

for subdir in (interim_data_dir / "output").iterdir():
    if subdir.is_dir():
        text_blocks = []
        for file_path in track(
            list(subdir.rglob("*.json")),
            description=f"📄 Loading documents from {subdir}...",
            transient=True,
        ):
            with open(file_path, encoding="utf-8") as f:
                parser_output_data = json.load(f)

            parser_output = BaseParserOutput(**parser_output_data)
            document = BaseDocument.from_parser_output(parser_output)
            if document.text_blocks is None:
                document_text_blocks = []
            else:
                doc_metadata_dict = (
                    document.model_dump(
                        exclude={"text_blocks", "page_metadata", "document_metadata"}
                    )
                    | document.document_metadata.model_dump()
                )

                document_text_blocks = [
                    doc_metadata_dict
                    | block.model_dump(exclude={"text"})
                    | {"text": block.to_string(), "block_index": idx}
                    for idx, block in enumerate(document.text_blocks)
                ]

            text_blocks.extend(document_text_blocks)

        dataset = pd.DataFrame.from_records(text_blocks)
        dataset["dataset_name"] = subdir.name
        datasets.append(dataset)
        console.log(f"✅ Loaded {len(dataset)} passages from {subdir.name} dataset")

cclw_dataset = (
    CPRDataset(CPRDocument)
    .from_huggingface()
    .filter_by_language("en")
    .to_huggingface()
    .to_pandas()
)
cclw_dataset["dataset_name"] = "cclw"
datasets.append(cclw_dataset)
console.log(f"✅ Loaded {len(cclw_dataset)} passages from CCLW dataset")


gst_dataset = (
    CPRDataset(GSTDocument)
    .from_huggingface()
    .filter_by_language("en")
    .to_huggingface()
    .to_pandas()
)
gst_dataset["dataset_name"] = "gst"
datasets.append(gst_dataset)
console.log(f"✅ Loaded {len(gst_dataset)} passages from GST dataset")

# Combine the datasets
combined_dataset = pd.concat(datasets, ignore_index=True, sort=False, axis=0)
console.log(f"📄 Combined dataset contains {len(combined_dataset)} passages")

# Add the world_bank_region column to the dataset
world_bank_regions = []
for _, document in track(
    combined_dataset.iterrows(),
    description="Adding world bank region metadata",
    transient=True,
    total=len(combined_dataset),
):
    if document.get("geography_iso"):
        world_bank_region = iso_to_world_bank_region.get(document["geography_iso"], "")
    elif document.get("geography"):
        geography_iso = geography_string_to_iso(document["geography"])
        world_bank_region = iso_to_world_bank_region.get(geography_iso, "")
    else:
        world_bank_region = ""
    world_bank_regions.append(world_bank_region)

combined_dataset["world_bank_region"] = world_bank_regions
console.log("🌍 Added region metadata to the dataset")

# Save the combined dataset
combined_dataset.to_feather(combined_dataset_path)
console.log(f"📄 Saved the combined dataset to {combined_dataset_path}")
