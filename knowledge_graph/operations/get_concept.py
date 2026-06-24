"""
Get-concept operation: reusable, Prefect-free domain logic.

Fetches a concept's metadata from Wikibase and its labelled passages from Argilla, saves
the concept to disk, and returns it; also loads a previously-saved concept from disk.

See `knowledge_graph/operations/README.md` for the conventions shared across operations.
"""

import typer
from pydantic import SecretStr
from rich.console import Console

from knowledge_graph.concept import Concept
from knowledge_graph.config import concept_dir
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelling import (
    ArgillaConfig,
    ArgillaSession,
    ResourceDoesNotExistError,
)
from knowledge_graph.wikibase import WikibaseConfig, WikibaseSession

console = Console()


def load_concept_local(wikibase_id: WikibaseID) -> Concept:
    """Load a concept from local storage by its Wikibase ID."""
    try:
        return Concept.load(concept_dir / f"{wikibase_id}.json")
    except FileNotFoundError as e:
        raise typer.BadParameter(
            f"Data for {wikibase_id} not found. \n"
            "If you haven't already, you should run:\n"
            f"  just get-concept {wikibase_id}\n"
        ) from e


async def get_concept_async(
    wikibase_id: WikibaseID,
    include_recursive_has_subconcept: bool = True,
    include_labels_from_subconcepts: bool = True,
    wikibase_config: WikibaseConfig | None = None,
    argilla_config: ArgillaConfig | None = None,
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
    if argilla_config:
        argilla = ArgillaSession(
            api_url=argilla_config.url,
            api_key=argilla_config.api_key.get_secret_value(),
        )
    else:
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
        labelled_passages = argilla.get_labelled_passages(wikibase_id=wikibase_id)
        console.log(
            f"🏷️ Found {len(labelled_passages)} labelled passages for {wikibase_id} in Argilla"
        )
        concept.labelled_passages = labelled_passages
    except (ValueError, ResourceDoesNotExistError):
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


def parse_wikibase_config(value) -> WikibaseConfig | None:
    url, username, password = value.split()
    return WikibaseConfig(
        url=url,
        username=username,
        password=SecretStr(password),
    )
