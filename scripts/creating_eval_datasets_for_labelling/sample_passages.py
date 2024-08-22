"""
Equitably sample passages for concepts, and populate an argilla project for labelling.

This script is used to equitably passages from our dataset(s) for instances of a given
set of concepts. It fetches concept metadata for the supplied concept and all
subconcept IDs, and uses their metadata to create a classifier. It then samples
passages from the dataset(s) that are predicted to be instances of the concept by the
classifier.

The passages are sampled equitably from the dataset(s) based on the source document
metadata. We want to evenly sample from source documents across a few strata:
- world bank region
- translated or untranslated
- type of document, eg CCLW, MCF, corporate disclosure

The script also optionally saves a set of passages which are _not_ predicted to be
instances of the concept, for use as negative examples in the labelling project.

The sampled passages are saved to a local file.
"""

from pathlib import Path

import pandas as pd
import typer
from rich.console import Console
from rich.progress import track
from tqdm.auto import tqdm

from scripts.config import processed_data_dir
from src.sampling import Sampler, SamplingConfig

tqdm.pandas()

app = typer.Typer()
console = Console()


@app.command()
def main(config_path: Path):
    console.log(f"Loading config from {config_path}")
    sampling_config = SamplingConfig.load(config_path)
    console.log(f"Config loaded: {sampling_config}")

    # Calculate the number of positive and negative samples to take
    negative_sample_size = int(
        sampling_config.sample_size * sampling_config.negative_proportion
    )
    negative_sampling_config = sampling_config.model_copy(
        update={"sample_size": negative_sample_size}, deep=True
    )
    negative_sampler = Sampler(config=negative_sampling_config)

    positive_sample_size = int(sampling_config.sample_size - negative_sample_size)
    positive_sampling_config = sampling_config.model_copy(
        update={"sample_size": positive_sample_size}, deep=True
    )
    positive_sampler = Sampler(config=positive_sampling_config)

    console.log(
        "Loading the combined dataset as a reference for sampling distributions"
    )
    combined_dataset = pd.read_feather(processed_data_dir / "combined_dataset.feather")
    candidate_passages_dir = processed_data_dir / "candidate_passages"

    # Sample passages for each concept
    for wikibase_id in track(
        sampling_config.wikibase_ids,
        description="Sampling passages",
        console=console,
    ):
        console.log(f"🔍 Sampling passages for {wikibase_id}")
        candidate_passages: pd.DataFrame = pd.read_json(
            candidate_passages_dir / f"{wikibase_id}.json", lines=True
        )

        # sample the positive candidates
        positive_sampled_passages = positive_sampler.sample(
            candidate_passages, reference_dataset=combined_dataset
        )
        console.log(
            f"Sampled {len(positive_sampled_passages)} positive passages for {wikibase_id}"
        )

        # i reckon we can assume that random sampling is sufficiently unlikely to result
        # in any duplicate passages being sampled, but it's worth being thorough
        negative_candidate_passages = candidate_passages[
            ~candidate_passages.index.isin(positive_sampled_passages.index)
        ]
        negative_sampled_passages = negative_sampler.sample(negative_candidate_passages)
        console.log(
            f"Sampled {len(negative_sampled_passages)} negative passages for {wikibase_id}"
        )

        # combine and shuffle the sampled passages
        sampled_passages = pd.concat(
            [positive_sampled_passages, negative_sampled_passages]
        )
        sampled_passages = sampled_passages.sample(frac=1)

        # save the sampled passages to a file with the concept ID in the name
        sampled_passages_path = processed_data_dir / "sampled_passages"
        sampled_passages_path.mkdir(parents=True, exist_ok=True)
        sampled_passages_file = sampled_passages_path / f"{wikibase_id}.json"
        sampled_passages.to_json(sampled_passages_file, orient="records", lines=True)
        console.log(f"Saved sampled passages to {sampled_passages_file}")

    console.log("🎉 All concepts have been sampled")


if __name__ == "__main__":
    app()
