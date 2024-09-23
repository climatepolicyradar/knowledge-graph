import os
from typing import Annotated

import argilla as rg
import typer
from rich.console import Console

from scripts.config import concept_dir
from src.argilla import combine_datasets
from src.identifiers import WikibaseID
from src.labelled_passage import LabelledPassage
from src.wikibase import WikibaseSession

console = Console()
app = typer.Typer()


@app.command()
def main(
    wikibase_id: Annotated[
        WikibaseID,
        typer.Option(
            ..., help="The Wikibase ID of the concept to fetch", parser=WikibaseID
        ),
    ],
):
    with console.status("Connecting to Wikibase..."):
        wikibase = WikibaseSession()
    console.log("✅ Connected to Wikibase")

    concept = wikibase.get_concept(wikibase_id)
    console.log(f'🔍 Fetched metadata for "{concept.preferred_label}" from wikibase')

    with console.status("Connecting to Argilla..."):
        rg.init(
            api_key=os.getenv("ARGILLA_API_KEY"), api_url=os.getenv("ARGILLA_API_URL")
        )
    console.log("✅ Connected to Argilla")

    datasets_about_our_concept = []
    with console.status("Fetching and filtering datasets from Argilla..."):
        for dataset in rg.list_datasets():
            try:
                # if the dataset.name ends with our wikibase_id, then it's one we want to process
                if WikibaseID(dataset.name.split("-")[-1]) == wikibase_id:
                    datasets_about_our_concept.append(dataset)
                    console.log(
                        f'🕵️  Found "{dataset.name}" in the "{dataset.workspace.name}" workspace in Argilla'
                    )
            except ValueError:
                continue

    if not datasets_about_our_concept:
        console.log("No labelled passages found for this concept")
    else:
        dataset = combine_datasets(*datasets_about_our_concept)
        concept.labelled_passages = [
            LabelledPassage.from_argilla_record(record) for record in dataset.records
        ]
        console.log(
            f"📚 Found {len(concept.labelled_passages)} labelled passages for {wikibase_id}"
        )

    # Save the concept to disk
    output_path = concept_dir / f"{wikibase_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(concept.model_dump_json(indent=2))
    console.log(f"💾 Concept saved to {output_path}")


if __name__ == "__main__":
    app()
