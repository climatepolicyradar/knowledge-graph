"""
Creates Argilla datasets for a set of concepts with candidate passages for labeling

Takes the artefacts from the sample_passages_for_concept.py script and pushes each set
to a dataset in Argilla. If the dataset already exists, it will add the new passages to
the existing dataset.

The dataset should be named f"{preferred_label} ({concept_id})" and the labelling task
should be a span labelling task with only one option (the preferred label of the concept
in question).

The passages should have been sampled equitably from the dataset(s) based on the source
document metadata.
"""

import os

import argilla as rg
import pandas as pd
import yaml
from rich.console import Console
from rich.progress import track
from rich.table import Table

from scripts.config import config_dir, processed_data_dir
from src.argilla import distribute_labelling_projects
from src.wikibase import WikibaseSession

console = Console()
wikibase = WikibaseSession()

with console.status("🔗 Connecting to Argilla..."):
    rg.init(
        api_key=os.getenv("ARGILLA_API_KEY"),
        api_url=os.getenv("ARGILLA_API_URL"),
    )

console.log("✅ Connected to Argilla")

# Load the sampling config
console.log("🔬 Loading sampling config and initializing the sampler")
config_path = config_dir / "sectors.yaml"
sampling_config = yaml.safe_load(config_path.read_text())
labellers = sampling_config.get("labellers", [])

# Create workspaces for the labellers
workspaces = [workspace.name for workspace in rg.list_workspaces()]
for labeller in labellers:
    if labeller not in workspaces:
        workspace = rg.Workspace.create(labeller)
        console.log(f"✅ Created workspace for {labeller}")

# create a rich table for the results to be displayed in
table = Table()
table.add_column("Concept ID", justify="right")
table.add_column("Wikibase URL", justify="right")
table.add_column("Argilla URL", justify="right")

datasets = {
    file.stem: pd.read_json(file, orient="records", lines=True).fillna("")
    for file in track(
        list((processed_data_dir / "sampled_passages").rglob("*.json")),
        description="Loading datasets",
        transient=True,
    )
}
console.log(f"✅ Loaded {len(datasets)} datasets")

labelling_assignments = distribute_labelling_projects(
    datasets=datasets.items(), labellers=labellers, min_labellers=2
)

for (concept_id, df), labeller_name in track(
    labelling_assignments,
    description="Creating datasets",
    transient=True,
    console=console,
):
    concept = wikibase.get_concept(concept_id)
    dataset_name = f"{concept.preferred_label}-{concept_id}".replace(" ", "-")
    console.log(f"✅ Retrieved metadata for {dataset_name}")

    try:
        rg.FeedbackDataset.from_argilla(
            name=dataset_name, workspace=labeller_name
        ).delete()
    except ValueError:
        pass

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
        name=dataset_name, workspace=labeller_name, show_progress=False
    )
    console.log(f"✅ Created dataset {dataset_in_argilla.url}")
    table.add_row(concept_id, concept.wikibase_url, dataset_in_argilla.url)

# display the results in the rich table
console.print(table)
