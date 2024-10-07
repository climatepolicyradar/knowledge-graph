"""
Creates Argilla datasets for a set of concepts with candidate passages for labelling.

Takes the artefacts from the sample_passages_for_concept.py script and pushes each set
to a dataset in Argilla.

The dataset should be named f"{preferred_label} ({concept_id})" and the labelling task
should be a span labelling task with only one option (the preferred label of the concept
in question).

The passages should have been sampled equitably from the dataset(s) based on the source
document metadata.
"""

import os
from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.progress import track
from rich.table import Table

import argilla as rg
from scripts.config import processed_data_dir
from src.identifiers import generate_identifier
from src.sampling import SamplingConfig
from src.wikibase import WikibaseSession

app = typer.Typer()

console = Console()


@app.command()
def main(config_path: Path):
    console.log(f"Loading config from {config_path}")
    sampling_config = SamplingConfig.load(config_path)
    console.log(f"Config loaded: {sampling_config}")

    console.log("🔗 Connecting to Wikibase...")
    wikibase = WikibaseSession()
    console.log("✅ Connected to Wikibase")

    console.log("🔗 Connecting to Argilla...")
    rg.init(api_key=os.getenv("ARGILLA_API_KEY"), api_url=os.getenv("ARGILLA_API_URL"))
    console.log("✅ Connected to Argilla")

    # Create a workspace for the datasets, and add each of the labellers to it.
    # N.B. for now, we're not doing any fancy assignment of labellers to datasets. We're
    # coordinating manually, and we'll come back round to efficient assignment in code in
    # a future iteration if needed.
    workspace_name = config_path.stem
    try:
        workspace = rg.Workspace.create(name=workspace_name)
        console.log(f'✅ Created workspace "{workspace.name}"')
    except ValueError:
        workspace = rg.Workspace.from_name(name=workspace_name)
        console.log(f'✅ Loaded workspace "{workspace.name}"')

    for username in sampling_config.labellers:
        try:
            password = generate_identifier(username)
            user = rg.User.create(
                username=username, password=password, role="annotator"
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

    # create a rich table for the results to be displayed in
    table = Table()
    table.add_column("Concept ID", justify="right")
    table.add_column("Wikibase URL", justify="right")
    table.add_column("Argilla URL", justify="right")

    sampled_passage_file_paths = [
        processed_data_dir / "sampled_passages" / f"{concept_id}.json"
        for concept_id in sampling_config.wikibase_ids
    ]
    missing_files = [file for file in sampled_passage_file_paths if not file.exists()]
    if missing_files:
        raise FileNotFoundError(f"Missing files for concepts: {missing_files}")

    datasets = {
        file.stem: pd.read_json(file, orient="records", lines=True)
        for file in track(
            sampled_passage_file_paths,
            description="Loading datasets",
            transient=True,
        )
    }
    console.log(f"✅ Loaded {len(datasets)} datasets")

    for concept_id, df in track(
        datasets.items(),
        description="Creating datasets",
        transient=True,
        console=console,
    ):
        concept = wikibase.get_concept(concept_id)
        dataset_name = f"{concept.preferred_label}-{concept_id}".replace(" ", "-")
        console.log(f"✅ Retrieved metadata for {dataset_name}")

        dataset = rg.FeedbackDataset(
            guidelines="Highlight the entity if it is present in the text",
            fields=[
                rg.TextField(name="text", title="Text", use_markdown=True),
            ],
            questions=[
                rg.SpanQuestion(
                    name="entities",
                    labels={concept.wikibase_id: concept.preferred_label},
                    field="text",
                    required=True,
                    allow_overlapping=False,
                )
            ],
        )

        records = [
            rg.FeedbackRecord(
                fields={"text": row["text"]}, metadata=row.apply(str).to_dict()
            )
            for _, row in df.iterrows()
        ]
        dataset.add_records(records)
        dataset_in_argilla = dataset.push_to_argilla(
            name=dataset_name, workspace=workspace_name, show_progress=False
        )
        console.log(f"✅ Created dataset {dataset_in_argilla.url}")
        table.add_row(concept_id, concept.wikibase_url, dataset_in_argilla.url)

    # display the results in the rich table
    console.print(table)


if __name__ == "__main__":
    app()
