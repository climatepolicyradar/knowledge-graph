from typing import Annotated

import typer
from rich.console import Console

from scripts.config import concept_dir
from src.identifiers import WikibaseID
from src.labelling import ArgillaSession
from src.wikibase import WikibaseSession

console = Console()
app = typer.Typer()


@app.command()
def main(
    wikibase_id: Annotated[
        WikibaseID,
        typer.Option(..., help="The Wikibase ID of the concept to fetch"),
    ],
):
    with console.status("Connecting to Wikibase..."):
        wikibase = WikibaseSession()
    console.log("✅ Connected to Wikibase")

    with console.status("Connecting to Argilla..."):
        argilla = ArgillaSession()
    console.log("✅ Connected to Argilla")

    concept = wikibase.get_concept(wikibase_id)
    console.log(f'🔍 Fetched metadata for "{concept}" from wikibase')

    try:
        with console.status("Fetching labelled passages from Argilla..."):
            labelled_passages = argilla.pull_labelled_passages(concept)
        console.log(
            f"🏷️ Found {len(labelled_passages)} labelled passages for {wikibase_id} in Argilla"
        )
        concept.labelled_passages = labelled_passages
    except ValueError:
        console.log(
            f"⚠️ No labelled passages found for {wikibase_id} in Argilla",
            style="yellow",
        )

    # Save the concept to disk
    output_path = concept_dir / f"{wikibase_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    concept.save(output_path)
    console.log(f"💾 Concept saved to {output_path}")


if __name__ == "__main__":
    app()
