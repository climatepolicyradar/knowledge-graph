import json

import pandas as pd
from cpr_sdk.models import BaseDocument, BaseParserOutput
from datasets import load_dataset
from rich.console import Console
from rich.progress import track

from scripts.config import processed_data_dir
from src.geography import iso_to_world_bank_region

console = Console()

combined_dataset_path = processed_data_dir / "combined_dataset.feather"

local_documents_dataframe = pd.DataFrame()

# restructure the parsed documents into a dataframe which can be joined with the others
# from huggingface.local_documents_dataframe = pd.DataFrame()
processed_documents_dir = processed_data_dir / "documents"
for subdir in processed_documents_dir.iterdir():
    if subdir.is_dir():
        all_text_blocks = []
        file_paths = list(subdir.rglob("*.json"))
        for file_path in track(
            file_paths,
            description=f"📄 Loading documents from {subdir}...",
            transient=True,
        ):
            # load each translated parser output and format it as a document
            with open(file_path, encoding="utf-8") as f:
                parser_output_data = json.load(f)
            parser_output = BaseParserOutput(**parser_output_data)
            document = BaseDocument.from_parser_output(parser_output)

            document_metadata_dict = {
                k: v
                for k, v in document.model_dump(
                    exclude=["text_blocks", "document_metadata", "page_metadata"]
                ).items()
            }
            document_metadata_dict.update(
                {
                    f"document_metadata.{k}": v
                    for k, v in document.document_metadata.model_dump().items()
                }
            )
            document_text_blocks = []
            for text_block in document.text_blocks:
                text_block_dict = {
                    f"text_block.{k}": v for k, v in text_block.model_dump().items()
                }
                text_block_dict["text_block.text"] = text_block.to_string()
                text_block_dict.update(document_metadata_dict.copy())
                document_text_blocks.append(text_block_dict)

            all_text_blocks.extend(document_text_blocks)

        dataset = pd.DataFrame.from_records(all_text_blocks)
        dataset["document_metadata.corpus_type_name"] = subdir.name

        local_documents_dataframe = pd.concat([local_documents_dataframe, dataset])
        console.log(f"✅ Loaded {len(dataset)} passages from {subdir.name} dataset")

# Load the huggingface dataset
dataset_name = "ClimatePolicyRadar/all-document-text-data-weekly"
huggingface_dataset = load_dataset(dataset_name, split="train")
console.log(f'✅ Loaded "{dataset_name}" from huggingface')
with console.status("🐼 Converting the huggingface dataset into a pandas dataframe"):
    huggingface_dataframe = huggingface_dataset.to_pandas()
console.log(f'✅ Converted "{dataset_name}" to a dataframe')
console.log(f'🔢 "{dataset_name}" contains {len(huggingface_dataframe)} passages')

# Combine the datasets
console.log("🤝 Combining the datasets")
combined_dataset = pd.concat(
    [local_documents_dataframe, huggingface_dataframe],
    ignore_index=True,
    sort=False,
    axis=0,
)
console.log(f"🔢 The combined dataset contains {len(combined_dataset)} passages")

console.log("🧹 Cleaning the combined dataset")

with console.status("🤏 Dropping rows with very short text"):
    combined_dataset = combined_dataset[
        combined_dataset["text_block.text"].str.len() > 20
    ]
console.log("🤏 Dropped rows with very short text")

with console.status("🏴󠁧󠁢󠁥󠁮󠁧󠁿 Dropping non-english language rows"):
    combined_dataset = combined_dataset[combined_dataset["text_block.language"] == "en"]
    combined_dataset["translated"] = (
        combined_dataset["translated"].astype(bool).fillna(False)
    )
console.log("🏴󠁧󠁢󠁥󠁮󠁧󠁿 Dropped non-english language rows")

with console.status("🌍 Adding world bank region metadata"):
    combined_dataset["world_bank_region"] = combined_dataset[
        "document_metadata.geographies"
    ].map(lambda x: iso_to_world_bank_region.get(x[0], None) if x else None)
console.log("🌍 Added world bank region metadata")

# map all problematic column datatypes to strings
with console.status("📊 Mapping all problematic column datatypes to strings"):
    for column in combined_dataset.columns:
        if combined_dataset[column].dtype == "object":
            combined_dataset[column] = combined_dataset[column].astype(str)
console.log("📊 Mapped all problematic column datatypes to strings")

# Save the combined dataset
combined_dataset.to_feather(combined_dataset_path)
console.log(f"📄 Saved the combined dataset to {combined_dataset_path}")
