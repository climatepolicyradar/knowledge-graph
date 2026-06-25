"""
Sample operation: reusable, Prefect-free domain logic.

Evenly samples passages for a concept from a dataset, optionally tracking the run in
Weights & Biases and uploading the sampled passages as an artifact. This module knows
nothing about Prefect; it is imported directly by the sampling flow and CLI wrapper.
"""

import math
from contextlib import nullcontext
from typing import Any, cast

import pandas as pd
import wandb

from knowledge_graph.classifier import EmbeddingClassifier, KeywordClassifier
from knowledge_graph.classifier.classifier import Classifier
from knowledge_graph.config import WANDB_ENTITY, equity_columns, processed_data_dir
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.sampling import create_balanced_sample, split_evenly
from knowledge_graph.utils import get_logger, serialise_pydantic_list_as_jsonl
from knowledge_graph.wandb_helpers import log_labelled_passages_artifact_to_wandb_run
from knowledge_graph.wikibase import WikibaseSession

CORPUS_TYPES = [
    "Litigation",
    "Laws and Policies",
    "Intl. agreements",
    "Reports",
    "AF",
    "GEF",
    "CIF",
    "GCF",
]


def run_sampling(
    wikibase_id: WikibaseID,
    dataset: pd.DataFrame,
    dataset_name: str = "balanced",
    sample_size: int = 130,
    min_negative_proportion: float = 0.1,
    corpus_types_include: list[str] | None = None,
    corpus_types_exclude: list[str] | None = None,
    max_size_to_sample_from: int = 500_000,
    max_negative_proportion: float | None = None,
    track_and_upload: bool = True,
    concept_overrides: dict[str, Any] | None = None,
    wikibase_username: str | None = None,
    wikibase_password: str | None = None,
    wikibase_url: str | None = None,
) -> str | None:
    """
    Evenly sample passages for concepts from the balanced dataset.

    This function is used to equitably sample passages from our dataset(s) for instances of a
    given concept. It loads concept metadata for the supplied concept and all
    subconcept IDs, and uses their metadata to create a classifier. It then samples
    passages from the passages which are likely to be instances of the concept.

    The passages are sampled as evenly as possible from the dataset(s) based on the
    source document metadata. We want to evenly sample from source documents across a
    few strata:
    - world bank region
    - translated or untranslated
    - type of document, eg CCLW, MCF, corporate disclosure

    The sampled passages are saved to a local file and uploaded to W&B.

    :param concept_overrides: Dict of concept property overrides from YAML
    :type concept_override: dict[str, Any] | None = None
    """
    logger = get_logger()

    # Calculate the optimal number of positive and negative samples to take
    negative_sample_size = math.floor(sample_size * min_negative_proportion)
    positive_sample_size = sample_size - negative_sample_size

    corpus_type_col = "document_metadata.corpus_type_name"

    if corpus_types_include:
        dataset = cast(
            pd.DataFrame, dataset[dataset[corpus_type_col].isin(corpus_types_include)]
        )
        logger.info(
            f"Filtered to corpus types {corpus_types_include}: "
            f"{len(dataset)} passages remain"
        )

    if corpus_types_exclude:
        dataset = cast(
            pd.DataFrame, dataset[~dataset[corpus_type_col].isin(corpus_types_exclude)]
        )
        logger.info(
            f"Excluded corpus types {corpus_types_exclude}: "
            f"{len(dataset)} passages remain"
        )

    # Limit dataset size if needed
    if len(dataset) > max_size_to_sample_from:
        logger.info(
            f"Limiting input data from {len(dataset)} rows to {max_size_to_sample_from}"
        )
        dataset = cast(pd.DataFrame, dataset.iloc[:max_size_to_sample_from])

    # Get the concept metadata from wikibase
    wikibase = WikibaseSession(
        username=wikibase_username,
        password=wikibase_password,
        url=wikibase_url,
    )
    concept = wikibase.get_concept(wikibase_id, include_labels_from_subconcepts=True)

    if concept_overrides:
        logger.info(f"🔧 Applying custom concept properties: {concept_overrides}")
        for key, value in concept_overrides.items():
            if hasattr(concept, key):
                setattr(concept, key, value)
                logger.info(f"  ✓ Set concept.{key} = {value}")
            else:
                logger.warning(f"  ⚠️  Warning: concept has no attribute '{key}'")

    job_type = "sample"
    wandb_config = {
        "concept_id": concept.id,
        "sample_size": sample_size,
        "dataset_name": dataset_name,
        "experimental_concept": concept_overrides is not None
        and len(concept_overrides) > 0,
        "concept_overrides": concept_overrides,
    }

    logged_artifact = None
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
            logger.info(f"🤖 Created a {model}")
            classifier_ids.append(model.id)
            logger.info(f"Running {model} on {len(raw_text_passages)} passages.")

            predictions = model.predict(
                raw_text_passages, batch_size=100, show_progress=True
            )

            # Add a column to the dataset for each classifier's predictions
            dataset[model.name] = [bool(pred) for pred in predictions]
            logger.info(
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
        if max_negative_proportion is not None:
            negative_sample_size = min(
                negative_sample_size,
                math.floor(sample_size * max_negative_proportion),
            )

        # Get negative samples (passages not identified by any classifier)
        negative_indices = ~dataset[[model.name for model in models]].any(axis=1)
        negative_candidates = dataset[negative_indices]

        # Sample negative examples
        negative_samples = create_balanced_sample(
            df=negative_candidates,  # type: ignore
            sample_size=negative_sample_size,
            on_columns=equity_columns,
        )
        # create_balanced_sample can return more than the requested number of negatives
        if len(negative_samples) > negative_sample_size:
            negative_samples = negative_samples.sample(negative_sample_size)

        logger.info(
            f"📊 Sampled {len(positive_samples)} positive passages, "
            f"{len(negative_samples)} negative passages"
        )

        # Combine positive and negative samples
        sampled_passages = pd.concat(
            [positive_samples, negative_samples], ignore_index=True
        )

        # Shuffle the final dataset so that the positive and negative examples are interleaved
        sampled_passages = sampled_passages.sample(frac=1)

        # Log the distribution of samples for the user
        logger.info("📊 Distribution of samples by classifier:")
        for model in models:
            positive_samples = sampled_passages[
                sampled_passages[model.name].astype(bool)
            ]
            logger.info(f"{model.name}: {len(positive_samples)}")

        logger.info("\n📊 Value counts for the sampled dataset:")
        for column in equity_columns:
            logger.info(str(sampled_passages[column].value_counts()))

        # Convert sampled passage rows to LabelledPassage objects and save them
        labelled_passages = []
        for _, row in sampled_passages.iterrows():
            metadata = row.to_dict()
            metadata.pop("text_block.text")
            # Convert numpy arrays to lists for JSON serialization
            for key, value in metadata.items():
                if hasattr(value, "tolist"):
                    metadata[key] = value.tolist()
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

        logger.info(f"Saved sampled passages to {sampled_passages_path}")

        if track_and_upload and run:
            logger.info("📄 Creating labelled passages artifact")
            logged_artifact = log_labelled_passages_artifact_to_wandb_run(
                labelled_passages=labelled_passages,
                run=run,
                concept=concept,
            )
            logger.info("✅ Labelled passages uploaded successfully")

    if track_and_upload and logged_artifact is not None:
        logged_artifact.wait()
        if logged_artifact.version is None:
            raise RuntimeError(
                f"W&B did not assign a version to the artifact for {wikibase_id}"
            )
        return (
            f"{WANDB_ENTITY}/{wikibase_id}/labelled-passages:{logged_artifact.version}"
        )
    return None
