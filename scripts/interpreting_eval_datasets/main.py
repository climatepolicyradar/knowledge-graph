"""
Create sets of golden labelled passages from Argilla datasets.

This script reads the datasets from Argilla, and combines them into a single dataset
for each concept. It then converts the Argilla records into labelled passages, and writes
them to a JSON file.

It also produces a golden set by filtering the labelled passages to only include the spans
which annotators agree on.

The script uses a supplied configuration file to determine which datasets to process.
"""

import json
import os
from pathlib import Path

import argilla as rg
import typer
from pydantic_core import to_jsonable_python
from rich.console import Console

from scripts.config import processed_data_dir
from src.argilla import combine_datasets
from src.identifiers import WikibaseID
from src.labelled_passage import LabelledPassage
from src.sampling import SamplingConfig

console = Console()

app = typer.Typer()


@app.command()
def main(config_path: Path):
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

    # group the datasets by name
    datasets_by_name = {}
    for dataset in taxonomy_datasets:
        datasets_by_name.setdefault(dataset.name.split("-")[-1], []).append(dataset)

    # combine the datasets
    combined_datasets = {}
    for name, datasets in datasets_by_name.items():
        combined_datasets[name] = combine_datasets(*datasets)
    console.log(f"📊 Combined {len(combined_datasets)} datasets")

    for name, dataset in combined_datasets.items():
        # convert the argilla records to labelled passages, and write them to a JSON file
        labelled_passages = [
            LabelledPassage.from_argilla_record(record) for record in dataset.records
        ]

        output_dir = processed_data_dir / "labelled_passages" / name
        output_dir.mkdir(parents=True, exist_ok=True)
        labelled_passages_output_path = output_dir / "labelled_passages.json"
        with open(labelled_passages_output_path, "w") as f:
            json.dump(to_jsonable_python(labelled_passages), f, indent=2)

        console.log(
            f"📊 Wrote {len(labelled_passages)} labelled passages to "
            f"{labelled_passages_output_path}"
        )

        # produce a golden set by filtering the labelled passages to only include the spans
        # on which annotators agree.
        golden_labelled_passages = []
        for labelled_passage in labelled_passages:
            agreed_spans = set()

            for span in labelled_passage.spans:
                for other_span in labelled_passage.spans:
                    if (
                        (
                            span.start_index == other_span.start_index
                            and span.end_index == other_span.end_index
                        )
                        and span.identifier == other_span.identifier
                        and span.labeller != other_span.labeller
                    ):
                        copy_span = span.model_copy()
                        copy_span.labeller = "all"
                        agreed_spans.add(copy_span)

            copy_labelled_passage = labelled_passage.model_copy()
            copy_labelled_passage.spans = list(agreed_spans)
            golden_labelled_passages.append(copy_labelled_passage)

        # dump the golden labelled passages to a JSON file
        golden_labelled_passages_output_path = (
            output_dir / "golden_labelled_passages.json"
        )
        with open(golden_labelled_passages_output_path, "w") as f:
            json.dump(to_jsonable_python(golden_labelled_passages), f, indent=2)

        console.log(
            f"📊 Wrote {len(golden_labelled_passages)} golden labelled passages to "
            f"{golden_labelled_passages_output_path}"
        )


if __name__ == "__main__":
    app()
