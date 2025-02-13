import os
from typing import Annotated

import typer
from rich.console import Console
from tqdm.auto import tqdm  # type: ignore

import argilla as rg
from scripts.config import concept_dir, processed_data_dir
from src.argilla import concept_to_dataset_name, labelled_passages_to_feedback_dataset
from src.concept import Concept
from src.identifiers import WikibaseID, generate_identifier
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
            "--workspace",
            help="The name of the workspace to create in Argilla",
            default="knowledge-graph",
        ),
    ],
):
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
        rg.init(
            api_key=os.getenv("ARGILLA_API_KEY"), api_url=os.getenv("ARGILLA_API_URL")
        )
    console.log("✅ Connected to Argilla")

    # Create a workspace for the datasets, and add each of the labellers to it.
    # N.B. for now, we're not doing any fancy assignment of labellers to datasets. We're
    # coordinating manually, and we'll come back round to efficient assignment in code in
    # a future iteration if needed.
    try:
        workspace = rg.Workspace.create(name=workspace_name)
        console.log(f'✅ Created workspace "{workspace.name}"')
    except ValueError:
        workspace = rg.Workspace.from_name(name=workspace_name)
        console.log(f'✅ Loaded workspace "{workspace.name}"')

    for username in usernames:
        try:
            password = generate_identifier(username)
            user = rg.User.create(
                username=username,
                password=password,
                role="annotator",  # type: ignore
            )
            console.log(f'✅ Created user "{username}" with password "{password}"')
        except KeyError:
            console.log(f'✅ User "{username}" already exists')
            user = rg.User.from_name(username)

        try:
            workspace.add_user(user.id)
            console.log(
                f'✅ Added user "{user.username}" to workspace "{workspace.name}"'
            )
        except ValueError:
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
    dataset_name = concept_to_dataset_name(concept)
    console.log(f"✅ Loaded metadata for {concept}")

    dataset = labelled_passages_to_feedback_dataset(labelled_passages, concept)
    dataset_in_argilla = dataset.push_to_argilla(
        name=dataset_name, workspace=workspace_name, show_progress=False
    )
    console.log(f'✅ Created dataset for "{concept}" at {dataset_in_argilla.url}')


if __name__ == "__main__":
    app()
