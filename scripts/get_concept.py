from typing import Annotated

import typer
from rich.console import Console

from scripts.config import concept_dir
from src.argilla import get_labelled_passages_from_argilla
from src.identifiers import WikibaseID
from src.keywords_from_csv import load_concept_keywords_from_csv
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

    try:
        labelled_passages = get_labelled_passages_from_argilla(concept)
        console.log(
            f"🏷️ Fetched {len(labelled_passages)} labelled passages for {wikibase_id} from Argilla"
        )
        concept.labelled_passages = labelled_passages
    except ValueError:
        console.log(
            f"⚠️ No labelled passages found for {wikibase_id} in Argilla",
            style="yellow",
        )

    try:
        csv_keywords = load_concept_keywords_from_csv(concept)
        console.log(
            f"🔗 Fetched {len(csv_keywords)} keywords for {wikibase_id} from CSV. Adding as aliases."
        )
        concept.alternative_labels.extend(csv_keywords)

    except FileNotFoundError:
        console.log(
            f"⚠️ No external keywords found for {wikibase_id}",
            style="yellow",
        )

    # Save the concept to disk
    output_path = concept_dir / f"{wikibase_id}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    concept.save(output_path)
    console.log(f"💾 Concept saved to {output_path}")


if __name__ == "__main__":
    app()
