"""
Run classifiers on the documents in the combined dataset, and save the results

This script runs inference of all the classifiers specified in the config on all of the
documents in the combined dataset, and saves the resulting positive passages for each
concept to a file.
"""

from pathlib import Path

import pandas as pd
import yaml
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from src.classifier import Classifier

console = Console()

# Load the config
with console.status("🔬 Loading config"):
    config_path = Path("scripts/sampling_for_sectors_classifier/config/sectors.yaml")
    sampling_config = yaml.safe_load(config_path.read_text())
console.log("✅ Config loaded")

# Set up the output directory
candidate_passages_path = Path("data/processed/candidate_passages")
candidate_passages_path.mkdir(parents=True, exist_ok=True)

# Load the combined dataset
with console.status("🚚 Loading combined dataset"):
    combined_dataset_path = Path("data/processed/combined_dataset")
    df = pd.read_feather(combined_dataset_path).sample(frac=0.1)
console.log(f"✅ Loaded {len(df)} passages from local file")

classifier_directory = Path("data/processed/classifiers")

progress = Progress(
    TextColumn("[progress.description]{task.description}", justify="right"),
    SpinnerColumn(),
    BarColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
)
wikibase_ids = sampling_config.get("wikibase_ids", [])
classifiers: dict[str, Classifier] = {}
for wikibase_id in wikibase_ids:
    classifier = Classifier.load(classifier_directory / wikibase_id)
    classifiers[wikibase_id] = classifier

    progress.add_task(
        description=f"{classifier.concept.preferred_label} ({wikibase_id})",
        total=len(df),
    )

with progress:
    for task in progress.tasks:
        wikibase_id = task.description
        classifier = classifiers[wikibase_id]

        predictions = []
        for _, row in df.iterrows():
            predictions.append(bool(classifier.predict(row["text"])))
            task.completed += 1

        console.log(
            f"Found {len(predictions)} passages which are about {classifier.concept.preferred_label}"
        )

        # save the candidate passages to a file with the concept ID in the name
        candidate_passages_file = candidate_passages_path / f"{wikibase_id}.json"
        df[predictions].to_json(candidate_passages_file, orient="records", lines=True)
