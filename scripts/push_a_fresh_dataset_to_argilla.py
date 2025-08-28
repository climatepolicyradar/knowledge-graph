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
            help="The Wikibase ID of the concept to create a dataset for",
            parser=WikibaseID,
        ),
    ],
    workspace_name: Annotated[
        str,
        typer.Option(
            ...,
            help="The name of the existing workspace in Argilla",
        ),
    ],
):
    with console.status("Connecting to Argilla..."):
        argilla = ArgillaSession()
    console.log("✅ Connected to Argilla")
    sampled_passages_dir = processed_data_dir / "sampled_passages"
    sampled_passages_path = sampled_passages_dir / f"{wikibase_id}.jsonl"

    wikibase = WikibaseSession()
    concept = wikibase.get_concept(wikibase_id)
    console.log(f"✅ Loaded metadata for {concept}")

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

    # Get existing workspace
    with console.status(f"Looking for workspace '{workspace_name}'..."):
        workspace = argilla.client.workspaces(name=workspace_name)

    if not workspace:
        raise ValueError(
            f"Workspace '{workspace_name}' not found. Please create the workspace first."
        )

    assert isinstance(workspace, rg.Workspace)
    console.log(f'✅ Found workspace "{workspace.name}", with id: {workspace.id}')

    dataset = argilla.labelled_passages_to_dataset(
        labelled_passages, concept, workspace
    )

    console.log(f'✅ Created dataset for "{concept}" at {dataset.name}')


if __name__ == "__main__":
    app()
