import math
from typing import Annotated

import pandas as pd
import typer
from more_itertools import chunked
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

from src.classifier import EmbeddingClassifier, KeywordClassifier
from src.classifier.classifier import Classifier
from src.config import equity_columns, processed_data_dir
from src.identifiers import WikibaseID
from src.labelled_passage import LabelledPassage
from src.sampling import create_balanced_sample, split_evenly
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
    Evenly sample passages for concepts from the balanced dataset.

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
    # Calculate the optimal number of positive and negative samples to take
    negative_sample_size = math.floor(sample_size * min_negative_proportion)
    positive_sample_size = sample_size - negative_sample_size

    with console.status(
        "Loading the balanced passage dataset for inference and sampling"
    ):
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
    console.log(
        f"✅ Loaded {len(balanced_dataset)} passages from {balanced_dataset_path}"
    )

    # Get the concept metadata from wikibase
    wikibase = WikibaseSession()
    concept = wikibase.get_concept(wikibase_id, include_labels_from_subconcepts=True)

    # Run inference with all classifiers
    raw_text_passages = balanced_dataset["text_block.text"].tolist()

    model_classes = [KeywordClassifier, EmbeddingClassifier]
    models: list[Classifier] = [model_class(concept) for model_class in model_classes]

    for model in models:
        model.fit()
        console.log(f"🤖 Created a {model}")

        progress_bar = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
        )
        with progress_bar:
            task = progress_bar.add_task(
                description=f"Predicting spans for {model}",
                total=len(raw_text_passages),
            )
            predictions = []
            for text_chunk in chunked(raw_text_passages, 100):
                predictions.extend(model.predict_batch(text_chunk))
                progress_bar.update(task, advance=len(text_chunk))

        # Add a column to the dataset for each classifier's predictions
        balanced_dataset[model.name] = predictions
        console.log(
            f"📊 Found {sum(bool(pred) for pred in predictions)} positive passages "
            f"using the {model}"
        )

    # Calculate the optimal number of positive samples to take per classifier
    samples_per_classifier = {
        model.name: sample_size
        for model, sample_size in zip(
            models, split_evenly(positive_sample_size, len(models))
        )
    }

    # Sample from each classifier's predictions
    positive_samples_list = []
    for model in models:
        df = balanced_dataset[balanced_dataset[model.name].astype(bool)]
        optimal_sample_size = samples_per_classifier[model.name]

        if len(df) > 0:  # Only sample if we have positives
            sampled_df = create_balanced_sample(
                df=df,  # type: ignore
                sample_size=min(optimal_sample_size, len(df)),
                on_columns=equity_columns,
            )
            positive_samples_list.append(sampled_df)

    # Combine positive samples
    positive_samples = pd.concat(positive_samples_list, ignore_index=True)
    positive_samples = positive_samples.drop_duplicates(subset=["text_block.text"])

    # Calculate the number of negative samples we need to take
    negative_sample_size = sample_size - len(positive_samples)

    # Get negative samples (passages not identified by any classifier)
    negative_indices = ~balanced_dataset[[model.name for model in models]].any(axis=1)
    negative_candidates = balanced_dataset[negative_indices]

    # Sample negative examples
    negative_samples = create_balanced_sample(
        df=negative_candidates,  # type: ignore
        sample_size=negative_sample_size,
        on_columns=equity_columns,
    )

    console.log(
        f"📊 Sampled {len(positive_samples)} positive passages, "
        f"{negative_sample_size} negative passages"
    )

    # Combine positive and negative samples
    sampled_passages = pd.concat(
        [positive_samples, negative_samples], ignore_index=True
    )

    # Shuffle the final dataset so that the positive and negative examples are interleaved
    sampled_passages = sampled_passages.sample(frac=1)

    # Log the distribution of samples for the user
    console.log("📊 Distribution of samples by classifier:")
    for model in models:
        positive_samples = sampled_passages[sampled_passages[model.name].astype(bool)]
        console.log(f"{model.name}: {len(positive_samples)}")

    console.log("\n📊 Value counts for the sampled dataset:")
    for column in equity_columns:
        console.log(sampled_passages[column].value_counts(), end="\n\n")

    # Convert sampled passage rows to LabelledPassage objects and save them
    labelled_passages = []
    for _, row in sampled_passages.iterrows():
        metadata = row.to_dict()
        metadata.pop("text_block.text")
        labelled_passages.append(
            LabelledPassage(text=row["text_block.text"], metadata=metadata, spans=[])  # type: ignore
        )

    sampled_passages_dir = processed_data_dir / "sampled_passages"
    sampled_passages_dir.mkdir(parents=True, exist_ok=True)
    sampled_passages_path = sampled_passages_dir / f"{wikibase_id}.jsonl"

    with open(sampled_passages_path, "w", encoding="utf-8") as f:
        f.writelines([entry.model_dump_json() + "\n" for entry in labelled_passages])

    console.log(f"Saved sampled passages to {sampled_passages_path}")


if __name__ == "__main__":
    app()
