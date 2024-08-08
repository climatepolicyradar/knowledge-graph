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

import pandas as pd
import yaml
from rich.console import Console
from rich.progress import track
from tqdm.auto import tqdm

from scripts.config import config_dir, processed_data_dir
from src.sampling import Sampler
from src.wikibase import WikibaseSession

tqdm.pandas()

console = Console()

# Load the sampling config
console.log("🔬 Loading sampling config and initializing the sampler")
config_path = config_dir / "sectors.yaml"
sampling_config = yaml.safe_load(config_path.read_text())
negative_proportion = sampling_config.get("negative_proportion", 0)
negative_sample_size = sampling_config["sample_size"] * negative_proportion
positive_sample_size = sampling_config["sample_size"] - negative_sample_size

positive_sampler = Sampler(
    stratified_columns=sampling_config["stratified_columns"],
    equal_columns=sampling_config["equal_columns"],
    sample_size=positive_sample_size,
)

combined_dataset = pd.read_feather(processed_data_dir / "combined_dataset.feather")

negative_sampler = Sampler(
    # passing no stratified columns and no equal columns will result in a random sample
    sample_size=negative_sample_size,
)
candidate_passages_dir = processed_data_dir / "candidate_passages"

# Sample passages for each concept
wikibase = WikibaseSession()
for wikibase_id in track(
    sampling_config.get("wikibase_ids", []),
    description="Sampling passages for concepts",
    console=console,
):
    console.log(f"🔍 Sampling passages for {wikibase_id}")
    candidate_passages: pd.DataFrame = pd.read_json(
        candidate_passages_dir / f"{wikibase_id}.json", lines=True
    )

    # sample the candidate passages
    positive_sampled_passages = positive_sampler.sample(
        candidate_passages, ref_dataset=combined_dataset
    )
    # i reckon we could probably assume that random sampling is sufficiently unlikely to
    # result in any duplicate passages being sampled, but it's worth being thorough
    negative_candidate_passages = candidate_passages[
        ~candidate_passages.index.isin(positive_sampled_passages.index)
    ]
    negative_sampled_passages = negative_sampler.sample(negative_candidate_passages)
    sampled_passages = pd.concat([positive_sampled_passages, negative_sampled_passages])
    # shuffle the sampled passages
    sampled_passages = sampled_passages.sample(frac=1)

    # save the sampled passages to a file with the concept ID in the name
    sampled_passages_path = processed_data_dir / "sampled_passages"
    sampled_passages_path.mkdir(parents=True, exist_ok=True)
    sampled_passages_file = sampled_passages_path / f"{wikibase_id}.json"
    sampled_passages.to_json(sampled_passages_file, orient="records", lines=True)
