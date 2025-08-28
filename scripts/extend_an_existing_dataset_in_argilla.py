from typing import Annotated

import argilla as rg
import typer
from rich.console import Console

from scripts.config import processed_data_dir
from src.identifiers import WikibaseID
from src.labelled_passage import LabelledPassage
from src.labelling import ArgillaSession
from src.wikibase import WikibaseSession

app = typer.Typer()
console = Console()


@app.command()
def main(
    wikibase_id: Annotated[
        WikibaseID,
        typer.Option(
            ...,
            help="The Wikibase ID of the concept to add passages for",
            parser=WikibaseID,
        ),
    ],
    workspace_name: Annotated[
        str,
        typer.Option(
            ...,
            help="The name of the workspace containing the existing dataset",
        ),
    ],
):
    with console.status("Connecting to Argilla..."):
        argilla = ArgillaSession()
    console.log("✅ Connected to Argilla")

    sampled_passages_dir = processed_data_dir / "sampled_passages"
    sampled_passages_path = sampled_passages_dir / f"{wikibase_id}.jsonl"

    console.log(f"Loading sampled passages for {wikibase_id}")
    try:
        with open(sampled_passages_path, "r", encoding="utf-8") as f:
            labelled_passages = [
                LabelledPassage.model_validate_json(line) for line in f
            ]
        n_annotations = sum([len(entry.spans) for entry in labelled_passages])
        console.log(
            f"Loaded {len(labelled_passages)} labelled passages "
            f"with {n_annotations} individual annotations"
        )
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"No sampled passages found for {wikibase_id}. Please run"
            f"  just sample {wikibase_id}"
        ) from e

    # Get concept metadata from wikibase
    wikibase = WikibaseSession()
    concept = wikibase.get_concept(wikibase_id)
    console.log(f"✅ Loaded metadata for {concept}")

    # Find existing dataset in Argilla
    with console.status(f"Looking for existing dataset for {concept}..."):
        dataset = argilla.client.datasets(
            str(concept.wikibase_id), workspace=workspace_name
        )

    if not dataset:
        raise ValueError(
            f"No existing dataset found for {concept} in workspace '{workspace_name}'. "
            f"Please create the dataset first using push_a_fresh_dataset_to_argilla.py"
        )

    console.log(
        f"✅ Found existing dataset '{dataset.name}' with {len(list(dataset.records))} records"
    )

    # Create Record objects from the labelled passages
    with console.status("Preparing new records..."):

        def reformat_metadata(metadata: dict) -> dict:
            """Reformat metadata for Argilla compatibility"""
            # Create a copy to avoid modifying the original
            clean_metadata = metadata.copy()
            # Remove fields that can't be serialized by Argilla
            clean_metadata.pop("KeywordClassifier", None)
            clean_metadata.pop("EmbeddingClassifier", None)
            # Convert dots to hyphens and lowercase keys
            return {
                key.replace(".", "-").lower(): value
                for key, value in clean_metadata.items()
            }

        records = [
            rg.Record(
                fields={"text": passage.text},
                metadata=reformat_metadata(passage.metadata),
            )
            for passage in labelled_passages
        ]

    console.log(f"✅ Prepared {len(records)} new records")

    # Add the new records to the existing dataset
    with console.status(f"Adding {len(records)} records to dataset..."):
        dataset.records.log(records)

    console.log(
        f"✅ Successfully added {len(records)} records to dataset '{dataset.name}'"
    )
    console.log(f"Dataset now contains {len(list(dataset.records))} total records")


if __name__ == "__main__":
    app()
