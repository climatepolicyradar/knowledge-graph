"""
Create sets of golden labelled passages from Argilla datasets.

This script reads the datasets from Argilla, and combines them into a single dataset
for each concept. It then converts the Argilla records into labelled passages, and writes
them to a JSON file.

It also produces a golden set by filtering the labelled passages to only include the spans
which annotators agree on.

The script uses a supplied configuration file to determine which datasets to process.
"""

import os
from pathlib import Path
from typing import Annotated

import argilla as rg
from rich.console import Console
from typer import Argument, Typer

from scripts.config import processed_data_dir
from src.argilla import combine_datasets
from src.identifiers import WikibaseID
from src.labelled_passage import LabelledPassage
from src.sampling import SamplingConfig

console = Console()

app = Typer()


@app.command()
def main(
    config_path: Annotated[Path, Argument(..., help="Path to the sampling config")],
):
    console.log(f"⚙️ Loading config from {config_path}")
    config = SamplingConfig.load(config_path)
    console.log("✅ Config loaded")
    console.log(
        f"🪪 There are {len(config.wikibase_ids)} wikibase IDs about {config_path.stem}: {config.wikibase_ids}"
    )

    console.log("🔗 Connecting to Argilla...")
    rg.init(api_key=os.getenv("ARGILLA_API_KEY"), api_url=os.getenv("ARGILLA_API_URL"))
    console.log("✅ Connected to Argilla")

    all_datasets = rg.list_datasets()
    console.log(f"📊 Found {len(all_datasets)} total datasets in argilla")
    datasets_for_kg = []
    for dataset in all_datasets:
        try:
            # if the dataset.name ends with a valid Wikibase ID, then it's KG-related
            WikibaseID(dataset.name.split("-")[-1])
            datasets_for_kg.append(dataset)
        except ValueError:
            continue

    console.log(
        f"📊 Found {len(datasets_for_kg)} datasets which are about knowledge graph concepts"
    )

    console.log(f"🔍 Checking for {config_path.stem} datasets")
    taxonomy_datasets = [
        dataset
        for dataset in datasets_for_kg
        if any(
            [dataset.name.endswith(sector_qid) for sector_qid in config.wikibase_ids]
        )
    ]
    console.log(f"📊 Found {len(taxonomy_datasets)} datasets about {config_path.stem}")

    console.log("📂 Collating datasets...")

    # group the datasets by wikibase_id
    datasets_by_wikibase_id = {}
    for dataset in taxonomy_datasets:
        wikibase_id = WikibaseID(dataset.name.split("-")[-1])
        datasets_by_wikibase_id.setdefault(wikibase_id, []).append(dataset)

    # combine the datasets
    combined_datasets = {}
    for wikibase_id, datasets in datasets_by_wikibase_id.items():
        combined_datasets[wikibase_id] = combine_datasets(*datasets)
    console.log(f"📊 Combined {len(combined_datasets)} datasets")

    for wikibase_id, dataset in combined_datasets.items():
        # convert the argilla records to labelled passages, and write them to a jsonl file

        output_dir = processed_data_dir / "labelled_passages" / wikibase_id
        output_dir.mkdir(parents=True, exist_ok=True)
        labelled_passages_output_path = output_dir / "labelled_passages.jsonl"
        with open(labelled_passages_output_path, "w", encoding="utf-8") as f:
            jsonl_data = [
                LabelledPassage.from_argilla_record(record).model_dump_json() + "\n"
                for record in dataset.records
            ]
            f.writelines(jsonl_data)

        console.log(
            f"📝 Wrote {len(jsonl_data)} lines to {labelled_passages_output_path}"
        )


if __name__ == "__main__":
    app()
