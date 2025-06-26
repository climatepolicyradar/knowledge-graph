import os
import uuid
from typing import Annotated

import argilla as rg
import typer
from argilla._exceptions._api import ConflictError, UnprocessableEntityError
from rich.console import Console
from tqdm.auto import tqdm  # type: ignore

from scripts.config import concept_dir, processed_data_dir
from src.argilla_v2 import ArgillaSession
from src.concept import Concept
from src.identifiers import Identifier, WikibaseID
from src.labelled_passage import LabelledPassage

tqdm.pandas()

app = typer.Typer()
console = Console()


@app.command()
def main(
    wikibase_id: Annotated[
        WikibaseID,
        typer.Option(
            ...,
            help="The Wikibase ID of the concept to sample passages for",
            parser=WikibaseID,
        ),
    ],
    usernames: Annotated[
        str,
        typer.Option(
            ...,
            help="Comma separated list of usernames who will be assigned to the labelling tasks",
            callback=lambda x: x.split(","),
        ),
    ],
    workspace_name: Annotated[
        str,
        typer.Option(
            ...,
            help="The name of the workspace to create in Argilla",
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

    with console.status("Connecting to Argilla..."):
        client = rg.Argilla(
            api_key=os.getenv("ARGILLA_API_KEY"), api_url=os.getenv("ARGILLA_API_URL")
        )
    console.log("✅ Connected to Argilla")

    # Create a workspace for the datasets, and add each of the labellers to it.
    # N.B. for now, we're not doing any fancy assignment of labellers to datasets. We're
    # coordinating manually, and we'll come back round to efficient assignment in code in
    # a future iteration if needed.
    try:
        workspace = rg.Workspace(name=workspace_name, id=uuid.uuid4())
        workspace.create()
        console.log(f'✅ Created workspace "{workspace.name}", with id: {workspace.id}')
    except (ValueError, ConflictError):
        workspace = client.workspaces(name=workspace_name)
        assert isinstance(workspace, rg.Workspace)
        console.log(f'✅ Loaded workspace "{workspace.name}", with id: {workspace.id}')

    for username in usernames:
        try:
            password = Identifier.generate(username)
            user = rg.User(
                id=uuid.uuid4(),
                username=username,
                password=password,
                role="annotator",  # type: ignore
                client=client,
            )
            user.create()
            console.log(f'✅ Created user "{username}" with password "{password}"')
        except (KeyError, ConflictError):
            console.log(f'✅ User "{username}" already exists')
            user = client.users(username)
            assert isinstance(user, rg.User)

        try:
            workspace.add_user(user)
            console.log(
                f'✅ Added user "{user.username}" to workspace "{workspace.name}"'
            )
        except (ValueError, UnprocessableEntityError):
            console.log(
                f'✅ User "{user.username}" already in workspace "{workspace.name}"'
            )

    try:
        concept = Concept.load(concept_dir / f"{wikibase_id}.json")
    except FileNotFoundError as e:
        raise typer.BadParameter(
            f"Data for {wikibase_id} not found. \n"
            "If you haven't already, you should run:\n"
            f"  just get-concept {wikibase_id}\n"
        ) from e

    console.log(f"✅ Loaded metadata for {concept}")

    dataset = argilla.labelled_passages_to_dataset(
        labelled_passages, concept, workspace
    )

    console.log(f'✅ Created dataset for "{concept}" at {dataset.name}')


if __name__ == "__main__":
    app()
