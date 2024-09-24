"""
Run classifiers on the documents in the combined dataset, and save the results

This script runs inference of all the classifiers specified in the config on all of the
documents in the combined dataset, and saves the resulting positive passages for each
concept to a file.
"""

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.progress import track

from scripts.config import classifier_dir, processed_data_dir
from src.classifier import Classifier
from src.sampling import SamplingConfig

console = Console()

app = typer.Typer()


@app.command()
def main(config_path: Path):
    console.log(f"Loading config from {config_path}")
    sampling_config = SamplingConfig.load(config_path)
    console.log(f"Config loaded: {sampling_config}")

    # Set up the output directory
    candidate_passages_path = processed_data_dir / "candidate_passages"
    candidate_passages_path.mkdir(parents=True, exist_ok=True)

    # Load the combined dataset
    console.log("🚚 Loading combined dataset")
    combined_dataset_path = processed_data_dir / "combined_dataset.feather"
    df = pd.read_feather(combined_dataset_path)
    console.log(f"✅ Loaded {len(df)} passages from local file")

    for wikibase_id in sampling_config.wikibase_ids:
        classifier = Classifier.load(classifier_dir / wikibase_id)
        console.log(f"Loaded classifier for concept {wikibase_id}")
        console.log(f"Running classifier {classifier} on passages...")
        predictions = [
            bool(classifier.predict(text))
            for text in track(
                df["text"].values,
                console=console,
                transient=True,
            )
        ]

        console.log(
            f"Found {sum(predictions)} positive passages for concept {wikibase_id}"
        )

        # save the candidate passages to a file with the concept ID in the name
        candidate_passages_file = candidate_passages_path / f"{wikibase_id}.json"
        df[predictions].to_json(candidate_passages_file, orient="records", lines=True)

        console.log(f"Saved candidate passages to {candidate_passages_file}")

    console.log("🎉 All classifiers have been run")


if __name__ == "__main__":
    app()
