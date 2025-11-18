import math
from contextlib import nullcontext
from typing import Annotated, Optional

import click
import pandas as pd
import typer
import wandb
from rich.console import Console

from flows.utils import serialise_pydantic_list_as_jsonl
from knowledge_graph.classifier import EmbeddingClassifier, KeywordClassifier
from knowledge_graph.classifier.classifier import Classifier
from knowledge_graph.config import WANDB_ENTITY, equity_columns, processed_data_dir
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.sampling import create_balanced_sample, split_evenly
from knowledge_graph.wandb_helpers import log_labelled_passages_artifact_to_wandb_run
from knowledge_graph.wikibase import WikibaseSession
from scripts.train import parse_kwargs_from_strings

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
    dataset_name: str = typer.Option(
        "balanced",
        help="Dataset to use",
        click_type=click.Choice(["balanced", "combined"]),
    ),
    max_size_to_sample_from: int = typer.Option(
        500_000,
        help="Maximum number of passages to load from the dataset before sampling",
    ),
    track_and_upload: bool = typer.Option(
        False,
        help="Whether to track the run and upload the labelled passages to W&B",
    ),
    concept_override: Annotated[
        Optional[list[str]],
        typer.Option(
            help="Concept property overrides in key=value format. Can be specified multiple times.",
        ),
    ] = None,
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

    :param concept_override: List of concept property overrides in key=value format (e.g., description, labels)
    :type concept_override: Optional[list[str]]
    """
    # Calculate the optimal number of positive and negative samples to take
    negative_sample_size = math.floor(sample_size * min_negative_proportion)
    positive_sample_size = sample_size - negative_sample_size

    if dataset_name == "balanced":
        dataset_filename = "balanced_dataset_for_sampling.feather"
    elif dataset_name == "combined":
        dataset_filename = "combined_dataset.feather"
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    with console.status(
        f"Loading the {dataset_name} passage dataset for inference and sampling"
    ):
        dataset_path = processed_data_dir / dataset_filename

        try:
            dataset = pd.read_feather(dataset_path)
            console.log(f"✅ Loaded {len(dataset)} passages from {dataset_path}")
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"{dataset_path} not found. If you haven't already, you should run:\n"
                "  just build-dataset"
            ) from e

    # Limit dataset size if needed
    if len(dataset) > max_size_to_sample_from:
        console.log(
            f"Limiting input data from {len(dataset)} rows to {max_size_to_sample_from}"
        )
        dataset = dataset.iloc[:max_size_to_sample_from]

    # Get the concept metadata from wikibase
    wikibase = WikibaseSession()
    concept = wikibase.get_concept(wikibase_id, include_labels_from_subconcepts=True)

    if concept_overrides := parse_kwargs_from_strings(concept_override):
        console.log(f"🔧 Applying custom concept properties: {concept_overrides}")
        for key, value in concept_overrides.items():
            if hasattr(concept, key):
                setattr(concept, key, value)
                console.log(f"  ✓ Set concept.{key} = {value}")
            else:
                console.log(
                    f"  ⚠️  Warning: concept has no attribute '{key}'", style="yellow"
                )

    job_type = "sample"
    wandb_config = {
        "concept_id": concept.id,
        "sample_size": sample_size,
        "dataset_name": dataset_name,
        "experimental_concept": concept_overrides is not None
        and len(concept_overrides) > 0,
        "concept_overrides": concept_overrides,
    }

    with (
        wandb.init(
            entity=WANDB_ENTITY,
            project=concept.wikibase_id,
            job_type=job_type,
            config=wandb_config,
        )
        if track_and_upload
        else nullcontext()
    ) as run:
        # Run inference with all classifiers
        raw_text_passages = dataset["text_block.text"].tolist()

        models: list[Classifier] = [
            KeywordClassifier(concept),
            EmbeddingClassifier(concept).set_prediction_threshold(0.7),
        ]
        classifier_ids: list[str] = []

        for model in models:
            model.fit()
            console.log(f"🤖 Created a {model}")
            classifier_ids.append(model.id)

            predictions = model.predict(
                raw_text_passages, batch_size=100, show_progress=True
            )

            # Add a column to the dataset for each classifier's predictions
            dataset[model.name] = predictions
            console.log(
                f"📊 Found {sum(bool(pred) for pred in predictions)} positive passages "
                f"using the {model}"
            )

        if track_and_upload and run:
            run.summary["classifier_ids"] = classifier_ids

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
            df = dataset[dataset[model.name].astype(bool)]
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
        negative_indices = ~dataset[[model.name for model in models]].any(axis=1)
        negative_candidates = dataset[negative_indices]

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
            positive_samples = sampled_passages[
                sampled_passages[model.name].astype(bool)
            ]
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
                LabelledPassage(
                    text=row["text_block.text"],  # type: ignore
                    metadata=metadata,
                    spans=[],
                )
            )

        sampled_passages_dir = processed_data_dir / "sampled_passages"
        sampled_passages_dir.mkdir(parents=True, exist_ok=True)
        sampled_passages_path = sampled_passages_dir / f"{wikibase_id}.jsonl"

        with open(sampled_passages_path, "w", encoding="utf-8") as f:
            f.write(serialise_pydantic_list_as_jsonl(labelled_passages))

        console.log(f"Saved sampled passages to {sampled_passages_path}")

        if track_and_upload and run:
            console.log("📄 Creating labelled passages artifact")
            log_labelled_passages_artifact_to_wandb_run(
                labelled_passages=labelled_passages,
                run=run,
                concept=concept,
            )
            console.log("✅ Labelled passages uploaded successfully")


if __name__ == "__main__":
    app()
