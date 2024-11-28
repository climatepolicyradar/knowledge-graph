import os
from typing import Annotated

import typer
from rich.console import Console

import argilla as rg
from scripts.config import concept_dir
from src.argilla import dataset_name_to_wikibase_id, dataset_to_labelled_passages
from src.identifiers import WikibaseID
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

    with console.status("Fetching and filtering datasets from Argilla..."):
        datasets = rg.list_datasets()
        if len(datasets) == 0:
            console.log(
                "❌ No datasets were returned from Argilla, you may need to be "
                "granted access to the workspace(s)"
            )

        for dataset in datasets:
            try:
                # if the dataset.name ends with our wikibase_id, then it's one we want to process
                if dataset_name_to_wikibase_id(dataset.name) == wikibase_id:
                    console.log(
                        f'🕵️  Found "{dataset.name}" in the "{dataset.workspace.name}" workspace'
                    )
                    concept.labelled_passages = dataset_to_labelled_passages(dataset)
                    console.log(
                        f"📚 Found {len(concept.labelled_passages)} labelled passages"
                    )
                    break
            except ValueError:
                continue

    if not concept.labelled_passages:
        console.log(f"❌ No labelled passages found for {wikibase_id}")

    # Save the concept to disk
    output_path = concept_dir / f"{wikibase_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    concept.save(output_path)
    console.log(f"💾 Concept saved to {output_path}")


if __name__ == "__main__":
    app()
