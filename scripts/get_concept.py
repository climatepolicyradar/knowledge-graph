import asyncio
from typing import Annotated, Optional

import typer
from pydantic import SecretStr
from rich.console import Console

from knowledge_graph.concept import Concept
from knowledge_graph.config import concept_dir
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelling import ArgillaSession
from knowledge_graph.wikibase import WikibaseConfig, WikibaseSession

console = Console()
app = typer.Typer()


async def get_concept_async(
    wikibase_id: WikibaseID,
    include_recursive_has_subconcept: bool = True,
    include_labels_from_subconcepts: bool = True,
    wikibase_config: Optional[WikibaseConfig] = None,
) -> Concept:
    """Async function to get concept and labelled passages."""
    console.log("Connecting to Wikibase...")
    if wikibase_config:
        wikibase = WikibaseSession(
            username=wikibase_config.username,
            password=wikibase_config.password.get_secret_value(),
            url=wikibase_config.url,
        )
    else:
        wikibase = WikibaseSession()
    console.log("✅ Connected to Wikibase")

    console.log("Connecting to Argilla...")
    argilla = ArgillaSession()
    console.log("✅ Connected to Argilla")

    concept = await wikibase.get_concept_async(
        wikibase_id,
        include_recursive_has_subconcept=include_recursive_has_subconcept,
        include_labels_from_subconcepts=include_labels_from_subconcepts,
    )
    # To handle redirects where the wikibase_id is overwritten
    if concept.wikibase_id != wikibase_id:
        raise typer.BadParameter(
            f"{wikibase_id} is a redirect to {concept.wikibase_id}, run with "
            f"{concept.wikibase_id} instead."
        )
    # Ensure concept data can be serialised and rebuilt without failing validations
    concept.model_validate_json(concept.model_dump_json())

    console.log(f'🔍 Fetched metadata for "{concept}" from wikibase')

    try:
        console.log("Fetching labelled passages from Argilla...")
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


def parse_wikibase_config(value) -> Optional[WikibaseConfig]:
    url, username, password = value.split()
    return WikibaseConfig(
        url=url,
        username=username,
        password=SecretStr(password),
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
    wikibase_config: Annotated[
        Optional[WikibaseConfig],
        typer.Option(
            ...,
            parser=parse_wikibase_config,
            help=(
                "Optional override of env variables for wikibase (username, "
                "password, url)"
            ),
        ),
    ] = None,
):
    concept = asyncio.run(
        get_concept_async(
            wikibase_id=wikibase_id,
            include_recursive_has_subconcept=include_recursive_has_subconcept,
            include_labels_from_subconcepts=include_labels_from_subconcepts,
            wikibase_config=wikibase_config,
        )
    )

    return concept


if __name__ == "__main__":
    app()
