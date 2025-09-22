from typing import Annotated, NamedTuple, Optional

import typer
from pydantic import SecretStr
from rich.console import Console

from knowledge_graph.config import concept_dir
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelling import ArgillaSession
from knowledge_graph.wikibase import WikibaseSession

console = Console()
app = typer.Typer()

WikibaseConfig = NamedTuple(
    "WikibaseConfig",
    [
        ("username", str),
        ("password", SecretStr),
        ("url", str),
    ],
)


@app.command()
def main(
    wikibase_id: Annotated[
        WikibaseID,
        typer.Option(
            ..., help="The Wikibase ID of the concept to fetch", parser=WikibaseID
        ),
    ],
    include_recursive_has_subconcept: bool = True,
    include_labels_from_subconcepts=True,
    wikibase_config: Optional[WikibaseConfig] = None,
):
    with console.status("Connecting to Wikibase..."):
        # Fetch all of its subconcepts recursively
        if wikibase_config:
            wikibase = WikibaseSession(
                username=wikibase_config.username,
                password=wikibase_config.password.get_secret_value(),
                url=wikibase_config.url,
            )
        else:
            wikibase = WikibaseSession()

    console.log("✅ Connected to Wikibase")

    with console.status("Connecting to Argilla..."):
        argilla = ArgillaSession()
    console.log("✅ Connected to Argilla")

    concept = wikibase.get_concept(
        wikibase_id,
        include_recursive_has_subconcept=include_recursive_has_subconcept,
        include_labels_from_subconcepts=include_labels_from_subconcepts,
    )
    # To handle redirects where the wikibase_id is overwritten
    concept.wikibase_id = wikibase_id
    # Ensure concept data can be serialised and rebuilt without failing validations
    concept.model_validate_json(concept.model_dump_json())

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

    return concept


if __name__ == "__main__":
    app()
