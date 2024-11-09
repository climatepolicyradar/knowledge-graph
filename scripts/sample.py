from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.progress import track

from scripts.config import processed_data_dir
from src.classifier import EmbeddingClassifier, KeywordClassifier
from src.concept import Concept
from src.identifiers import WikibaseID
from src.labelled_passage import LabelledPassage
from src.wikibase import WikibaseSession

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
    sample_size: int = typer.Option(130, help="The number of passages to sample"),
    min_negative_proportion: float = typer.Option(
        0.1, help="The minimum proportion of negative samples to take"
    ),
):
    """
    Equitably sample passages for concepts from the balanced dataset.

    This script is used to equitably passages from our dataset(s) for instances of a
    given concept. It loads concept metadata for the supplied concept and all
    subconcept IDs, and uses their metadata to create a classifier. It then samples
    passages from the passages which are likely to be instances of the concept.

    The passages are sampled as evenly as possible from the dataset(s) based on the
    source document metadata. We want to evenly sample from source documents across a
    few strata:
    - world bank region
    - translated or untranslated
    - type of document, eg CCLW, MCF, corporate disclosure

    The sampled passages are saved to a local file.
    """
    # Calculate the number of positive and negative samples to take
    negative_sample_size = int(sample_size * min_negative_proportion)
    positive_sample_size = int(sample_size - negative_sample_size)

    console.log("Loading the balanced passage dataset for inference and sampling")
    try:
        balanced_dataset_path = (
            processed_data_dir / "balanced_dataset_for_sampling.feather"
        )
        balanced_dataset = pd.read_feather(balanced_dataset_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Balanced dataset not found. If you haven't already, you should run:\n"
            "  just build-dataset"
        ) from e

    # Get the concept metadata from wikibase
    wikibase = WikibaseSession()
    top_level_concept = wikibase.get_concept(wikibase_id)
    subconcepts = wikibase.get_subconcepts(wikibase_id, recursive=True)
    concept = Concept(
        preferred_label=top_level_concept.preferred_label,
        description=top_level_concept.description,
        alternative_labels=top_level_concept.all_labels
        + [label for subconcept in subconcepts for label in subconcept.all_labels],
        negative_labels=top_level_concept.negative_labels
        + [label for subconcept in subconcepts for label in subconcept.negative_labels],
    )

    models = []
    for model_class in [KeywordClassifier, EmbeddingClassifier]:
        model = model_class(concept)
        models.append(model)
        console.log(f"🤖 Created a {model}")

        raw_text_passages = balanced_dataset["text_block.text"].tolist()
        predictions = [
            model.predict(text)
            for text in track(
                raw_text_passages,
                description=f"Predicting spans for {model}",
                transient=True,
            )
        ]
        balanced_dataset[str(model)] = predictions
        console.log(
            f"📊 Found {sum(bool(pred) for pred in predictions)} positive passages "
            f"using the {model}"
        )

    # filter the dataset to only include the positive passages
    positive_indices = balanced_dataset[[str(model) for model in models]].any(axis=1)
    positive_samples = balanced_dataset[positive_indices]
    # and the negative passages
    negative_samples = balanced_dataset[~positive_indices]

    positive_sample_size = min(positive_sample_size, len(positive_samples))
    negative_sample_size = sample_size - positive_sample_size

    # combine the sampled passages
    sampled_passages = pd.concat(
        [
            positive_samples.sample(positive_sample_size),
            negative_samples.sample(negative_sample_size),
        ]
    )
    # shuffle them so that the positive and negative examples are interleaved
    sampled_passages = sampled_passages.sample(frac=1)

    console.log("📊 Value counts for the sampled dataset:", end="\n\n")
    console.log(sampled_passages["translated"].value_counts(), end="\n\n")
    console.log(sampled_passages["world_bank_region"].value_counts(), end="\n\n")
    console.log(
        sampled_passages["document_metadata.corpus_type_name"].value_counts(),
        end="\n\n",
    )

    # make the sampled passages dataframe into a list of labelledpassage objects
    labelled_passages = []
    for _, row in sampled_passages.iterrows():
        metadata = row.to_dict()
        metadata.pop("text_block.text")
        labelled_passages.append(
            LabelledPassage(text=row["text_block.text"], metadata=metadata, spans=[])
        )

    # save the sampled passages to a file with the concept ID in the name
    sampled_passages_dir = processed_data_dir / "sampled_passages"
    sampled_passages_dir.mkdir(parents=True, exist_ok=True)
    sampled_passages_path = sampled_passages_dir / f"{wikibase_id}.json"

    with open(sampled_passages_path, "w", encoding="utf-8") as f:
        f.writelines([entry.model_dump_json() + "\n" for entry in labelled_passages])

    console.log(f"Saved sampled passages to {sampled_passages_path}")


if __name__ == "__main__":
    app()
