import random
from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from tqdm.auto import tqdm

from scripts.config import NEGATIVE_PROPORTION, SAMPLE_SIZE, processed_data_dir
from src.identifiers import WikibaseID
from src.labelled_passage import LabelledPassage
from src.sampling import Sampler

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
):
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
    # Calculate the number of positive and negative samples to take

    negative_sample_size = int(SAMPLE_SIZE * NEGATIVE_PROPORTION)
    positive_sample_size = int(SAMPLE_SIZE - negative_sample_size)
    sampler = Sampler()
    console.log(
        f"Created a sampler which will stratify samples on {sampler.stratified_columns} and "
        f"return equal numbers of samples from {sampler.equal_columns}"
    )

    console.log(
        "Loading the combined dataset as a reference for sampling distributions"
    )
    try:
        combined_dataset_path = processed_data_dir / "combined_dataset.feather"
        combined_dataset = pd.read_feather(combined_dataset_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Combined dataset not found. If you haven't already, you should run:\n"
            "  just build-dataset"
        ) from e

    console.log(f"🔍 Sampling passages for {wikibase_id}")
    predictions_path = processed_data_dir / "predictions" / f"{wikibase_id}.jsonl"
    predictions: list[LabelledPassage] = [
        LabelledPassage.model_validate_json(line)
        for line in predictions_path.read_text(encoding="utf-8").splitlines()
    ]

    # sample the positive candidates
    positive_sampled_passages = sampler.sample(
        sample_size=positive_sample_size,
        dataset=predictions,
        reference_dataset=combined_dataset,
    )
    console.log(
        f"Sampled {len(positive_sampled_passages)} positive passages for {wikibase_id}"
    )

    negative_predictions = sampler.dataframe_to_labelled_passages(
        combined_dataset.sample(100_000)
        # this is an ugly hack and i promise !!! to come back and fix it soon, alongside
        # the more robust sampling implementation. for now, we're just taking a random
        # sample of 100,000 passages from the combined dataset and assuming that they were
        # negatively labelled by the model (the probability of a passage being positively
        # labelled is very low, so this is not _too_ bad an assumption).
    )

    negative_sampled_passages = sampler.sample(
        sample_size=negative_sample_size,
        dataset=negative_predictions,
        reference_dataset=combined_dataset,
    )
    console.log(
        f"Sampled {len(negative_sampled_passages)} negative passages for {wikibase_id}"
    )

    # combine the sampled passages
    sampled_passages = positive_sampled_passages + negative_sampled_passages
    # shuffle them so that the positive and negative examples are interleaved
    random.shuffle(sampled_passages)

    # save the sampled passages to a file with the concept ID in the name
    sampled_passages_dir = processed_data_dir / "sampled_passages"
    sampled_passages_dir.mkdir(parents=True, exist_ok=True)
    sampled_passages_path = sampled_passages_dir / f"{wikibase_id}.json"

    with open(sampled_passages_path, "w", encoding="utf-8") as f:
        f.writelines([entry.model_dump_json() + "\n" for entry in sampled_passages])

    console.log(f"Saved sampled passages to {sampled_passages_path}")


if __name__ == "__main__":
    app()
