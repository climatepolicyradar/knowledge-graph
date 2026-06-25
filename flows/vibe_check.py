"""
Vibe check inference flow.

The flow either runs on a custom supplied list of wikibase IDs, or on the default set
listed in the vibe-checker/config.yml file.

The flow then loads the passages dataset from S3, samples a subset of them which seem
like plausible matches for the concept, and uses a default classifier to run inference
on them.

After processing the dataset, the flow pushes the resulting set of labelled passages
(both positive and negative) to S3, as well as the concept data and classifier metadata.

See vibe-checker/README.md for more details!
"""

import io
import json
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np
import pandas as pd
from mypy_boto3_s3 import S3Client
from prefect import flow, task
from prefect.cache_policies import NO_CACHE
from prefect.futures import wait
from prefect.task_runners import ThreadPoolTaskRunner
from pydantic import Field
from sentence_transformers import SentenceTransformer

from flows.config import Config
from flows.train import _set_up_training_environment, load_wikibase_ids_from_config
from knowledge_graph.cloud import (
    AwsEnv,
    get_aws_ssm_param,
)
from knowledge_graph.cloud import (
    get_s3_client as cloud_get_s3_client,
)
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.labelling import ArgillaConfig
from knowledge_graph.utils import get_logger, serialise_pydantic_list_as_jsonl
from knowledge_graph.wikibase import WikibaseConfig, WikibaseSession
from scripts.train import run_training

aws_region = os.getenv("AWS_REGION", "eu-west-1")


class LabelledPassageWithMarkup(LabelledPassage):
    """LabelledPassage wrapper including an extra field for text marked up as HTML."""

    marked_up_text: str = Field(
        ..., description="Text marked up as HTML with highlighted spans"
    )

    @classmethod
    def from_labelled_passage(
        cls, labelled_passage: LabelledPassage
    ) -> "LabelledPassageWithMarkup":
        """Create a LabelledPassageWithMarkup from a LabelledPassage."""
        return cls(
            marked_up_text=labelled_passage.get_highlighted_text(
                start_pattern='<span class="prediction-highlight">',
                end_pattern="</span>",
            ),
            **labelled_passage.model_dump(),
        )


def get_bucket_name_from_ssm() -> str:
    """Fetch bucket name from AWS Systems Manager Parameter Store."""
    return get_aws_ssm_param("/vibe-checker/bucket-name", region_name=aws_region)


logger = get_logger()


def get_s3_client() -> S3Client:
    """Get a configured S3 client."""
    aws_env = AwsEnv(os.environ["AWS_ENV"]) if "AWS_ENV" in os.environ else None
    return cast(S3Client, cloud_get_s3_client(aws_env=aws_env, region_name=aws_region))


def get_object_bytes_from_s3(s3_client: S3Client, key: str) -> bytes:
    """Load bytes from S3 object."""
    bucket_name = get_bucket_name_from_ssm()
    return s3_client.get_object(Bucket=bucket_name, Key=key)["Body"].read()


def push_object_bytes_to_s3(s3_client: S3Client, key: str | Path, data: bytes) -> None:
    """Push bytes to S3 object."""
    bucket_name = get_bucket_name_from_ssm()
    s3_client.put_object(Bucket=bucket_name, Key=str(key), Body=data)


@task(retries=3, retry_delay_seconds=5)
def load_passages_dataset(
    passages_dataset_file_name: str = "passages_dataset.feather",
) -> pd.DataFrame:
    """Load the passages dataset from S3."""
    s3_client = get_s3_client()
    bytes_from_s3 = get_object_bytes_from_s3(s3_client, passages_dataset_file_name)
    try:
        dataset = pd.read_feather(io.BytesIO(bytes_from_s3))
        if dataset.empty:
            raise ValueError("The dataset is empty")
    except Exception as e:
        raise ValueError("Failed to load dataset") from e

    # keep only the useful columns (matching the combined_dataset schema from
    # the build_dataset Prefect flow / s3://cpr-kg-feather-files)
    dataset = dataset[
        [
            "text_block.text",
            "document_id",
            "document_name",
            "document_slug",
            "family_slug",
            "translated",
            "publication_ts",
            "document_metadata.corpus_type_name",
            "world_bank_region",
        ]
    ]
    assert isinstance(dataset, pd.DataFrame)
    return dataset


@task(retries=3, retry_delay_seconds=5)
def load_embeddings(
    embeddings_file_name: str = "passages_embeddings.npy",
) -> np.ndarray:
    """Load the passages embeddings from S3."""
    s3_client = get_s3_client()
    bytes_from_s3 = get_object_bytes_from_s3(s3_client, embeddings_file_name)
    return np.load(io.BytesIO(bytes_from_s3))


@task(retries=3, retry_delay_seconds=5)
def load_embeddings_metadata(
    embeddings_metadata_file_name: str = "passages_embeddings_metadata.json",
) -> dict:
    """Load the passages embeddings metadata from S3."""
    s3_client = get_s3_client()
    bytes_from_s3 = get_object_bytes_from_s3(s3_client, embeddings_metadata_file_name)
    return json.load(io.BytesIO(bytes_from_s3))


@task(retries=2, retry_delay_seconds=10, cache_policy=NO_CACHE)
async def process_single_concept(
    wikibase_id: WikibaseID,
    passages_dataset: pd.DataFrame,
    passages_embeddings: np.ndarray,
    embedding_model: SentenceTransformer,
    wikibase_config: WikibaseConfig,
    argilla_config: ArgillaConfig,
    s3_client: Any,
    aws_env: AwsEnv,
    track_and_upload: bool = True,
) -> dict:
    """
    Process inference for a single concept.

    This task is designed to be isolated - if it fails, it won't affect the other
    concept processing tasks.
    """
    try:
        wikibase = WikibaseSession(
            username=wikibase_config.username,
            password=wikibase_config.password.get_secret_value(),
            url=str(wikibase_config.url),
        )
        concept = await wikibase.get_concept_async(wikibase_id)
        logger.info(f"Loaded concept: {concept}")

        concept_embedding = embedding_model.encode(
            concept.to_markdown(), normalize_embeddings=True
        )

        # Ensure embeddings and concept embedding have compatible dimensions
        if len(passages_embeddings) != len(passages_dataset):
            raise ValueError(
                f"Mismatch between embeddings ({len(passages_embeddings)}) "
                f"and dataset ({len(passages_dataset)}) lengths"
            )

        similarities = passages_embeddings @ concept_embedding  # Shape: (n_passages,)

        similarity_threshold = 0.65
        min_passages = 10_000
        max_passages = 100_000

        # Sort indices by similarity descending without copying the full DataFrame
        sorted_indices = np.argsort(similarities)[::-1]
        sorted_similarities = similarities[sorted_indices]

        above_count = int((sorted_similarities > similarity_threshold).sum())
        below_count = len(sorted_similarities) - above_count

        logger.info(
            f"Found {above_count} passages above threshold {similarity_threshold}"
        )
        logger.info(f"Found {below_count} passages below threshold")

        # Selection strategy:
        # 1. If we have enough above threshold, take all of them, up to max_passages.
        # 2. If we don't have enough above threshold, take everything which is above
        #    the threshold, and then supplement the list with the passages which are
        #    closest to the threshold to reach min_passages.
        if above_count >= min_passages:
            selected_indices = sorted_indices[: min(above_count, max_passages)]
        else:
            # sorted_indices[above_count:] are below-threshold, already ordered closest
            # to threshold first (since sorted desc by similarity)
            remaining_needed = min(min_passages - above_count, below_count)
            selected_indices = sorted_indices[: above_count + remaining_needed]
            selected_indices = selected_indices[:max_passages]

        # Copy only the selected subset (up to max_passages rows, not the full dataset)
        selected_passages = passages_dataset.iloc[selected_indices].copy()
        selected_passages = selected_passages.assign(
            similarity=similarities[selected_indices]
        )
        selected_passages = selected_passages.reset_index(drop=True)

        logger.info(f"Selected {len(selected_passages)} passages")
        max_similarity = max(selected_passages["similarity"])
        min_similarity = min(selected_passages["similarity"])
        logger.info(f"Similarity range: {min_similarity:.3f}-{max_similarity:.3f}")

        # Get or create classifier from W&B (with force=False, this will fetch existing
        # or train new if missing, ensuring we always have the latest model)
        classifier = await run_training(
            wikibase_id=wikibase_id,
            track_and_upload=track_and_upload,
            aws_env=aws_env,
            wikibase_config=wikibase_config,
            argilla_config=argilla_config,
            s3_client=s3_client,
            force=False,  # Fetch if exists, train if missing
            # We don't need to run evaluation during the vibe checking stage. Doing so
            # before the concept has a proper labelled dataset in argilla will also lead
            # to the task crashing.
            evaluate=False,
        )
        logger.info(f"Loaded/trained classifier: {classifier}")

        classifier_metadata = {
            "id": classifier.id,
            "name": str(classifier),
            "date": datetime.now().date().isoformat(),
        }

        # Run inference for the concept
        logger.info(f"Running inference for {classifier}")

        assert isinstance(selected_passages, pd.DataFrame)
        texts = selected_passages["text_block.text"].astype(str).tolist()

        logger.info(f"Making predictions for {len(texts)} passages")
        predicted_spans_list = classifier.predict(texts, show_progress=False)

        metadata_records = selected_passages.astype(str).to_dict(orient="records")
        labelled_passages: list[LabelledPassage] = [
            LabelledPassage(text=text, spans=predicted_spans, metadata=record)
            for text, predicted_spans, record in zip(
                texts, predicted_spans_list, metadata_records
            )
        ]

        logger.info(f"Generated {len(labelled_passages)} labelled passages")

        # before uploading the passages, we should shuffle them
        random.shuffle(labelled_passages)

        bucket_name = get_bucket_name_from_ssm()
        output_prefix = Path(wikibase_id) / classifier.id
        logger.info(f"Outputs will be stored in s3://{bucket_name}/{output_prefix}")

        # Push results for this concept to S3
        passages_with_markup = [
            LabelledPassageWithMarkup.from_labelled_passage(labelled_passage)
            for labelled_passage in labelled_passages
        ]
        jsonl_string = serialise_pydantic_list_as_jsonl(passages_with_markup)

        logger.info(f"Pushing predictions to S3: {output_prefix / 'predictions.jsonl'}")
        push_object_bytes_to_s3(
            s3_client=s3_client,
            key=output_prefix / "predictions.jsonl",
            data=jsonl_string.encode("utf-8"),
        )

        logger.info(f"Pushing concept data to S3: {output_prefix / 'concept.json'}")
        push_object_bytes_to_s3(
            s3_client=s3_client,
            key=output_prefix / "concept.json",
            data=concept.model_dump_json().encode("utf-8"),
        )

        logger.info(
            f"Pushing classifier metadata to S3: {output_prefix / 'classifier.json'}"
        )
        push_object_bytes_to_s3(
            s3_client=s3_client,
            key=output_prefix / "classifier.json",
            data=json.dumps(classifier_metadata).encode("utf-8"),
        )

        n_positive_passages = sum(1 for passage in labelled_passages if passage.spans)
        n_passages = len(labelled_passages)

        result = {
            "concept_id": wikibase_id,
            "preferred_label": concept.preferred_label,
            "n_passages": n_passages,
            "n_positive_passages": n_positive_passages,
            "output_prefix": str(output_prefix),
            "status": "success",
        }

        logger.info(
            f"Completed processing {wikibase_id} ({n_positive_passages}/{len(labelled_passages)} positive)"
        )
        return result

    except Exception as e:
        logger.error(f"Failed to process concept {wikibase_id}: {str(e)}")
        # Return failure result instead of raising exception
        # This prevents one concept failure from stopping others
        return {
            "concept_id": wikibase_id,
            "status": "failed",
            "error": str(e),
            "n_passages": len(passages_dataset),
            "n_positive_passages": 0,
            "output_prefix": "",
        }


@flow(  # pyright: ignore[reportCallIssue, reportReturnType]
    timeout_seconds=None,
    task_runner=ThreadPoolTaskRunner(max_workers=3),  # pyright: ignore[reportArgumentType]
)
async def vibe_check_inference(
    wikibase_ids: Optional[list[str]] = None,
) -> list[dict[str, str]]:
    """
    Run inference on a set of concepts and push predictions to S3.

    For each concept, this flow:
    - Loads the passages dataset and pre-computed embeddings from S3
    - Samples passages most relevant to the concept based on their semantic similarity
    - Gets the latest classifier from W&B (or trains a new one if one doesn't exist)
    - Runs inference on the selected passages
    - Pushes predictions and concept/classifier metadata to S3

    Concepts are processed in parallel. If any concept fails, it should not affect the
    processing of other concepts.

    :param wikibase_ids: Optional list of Wikibase IDs to process. If not provided,
    the flow will load the default list from vibe-checker/config.yml.
    """
    config = await Config.create()
    _, wikibase_config, argilla_config, s3_client = await _set_up_training_environment(
        config=config, aws_env=config.aws_env
    )

    if not wikibase_ids:
        logger.info("Loading wikibase IDs from config...")
        sorted_wikibase_ids = sorted(
            load_wikibase_ids_from_config("vibe-checker/config.yml")
        )
    else:
        sorted_wikibase_ids = sorted([WikibaseID(id) for id in wikibase_ids])

    logger.info(f"Running inference for {len(sorted_wikibase_ids)} concept(s)...")

    logger.info("Loading dataset...")
    passages_dataset = load_passages_dataset()
    logger.info(f"Loaded {len(passages_dataset)} passages from the dataset")

    logger.info("Loading embeddings...")
    passages_embeddings = load_embeddings()
    logger.info(f"Loaded {passages_embeddings.shape[0]} embeddings")

    logger.info("Loading embeddings metadata...")
    passages_embeddings_metadata = load_embeddings_metadata()
    logger.info("Loaded embeddings generation metadata")

    embedding_model_name = passages_embeddings_metadata["embedding_model_name"]
    logger.info(f"Loading embedding model: {embedding_model_name}")
    embedding_model = SentenceTransformer(embedding_model_name)
    logger.info(f"Loaded embedding model: {embedding_model_name}")

    # Submit a separate inference task for each of the concepts, and then wait for all
    logger.info(
        f"Starting parallel inference of {len(sorted_wikibase_ids)} concepts..."
    )
    concept_futures = []
    for wikibase_id in sorted_wikibase_ids:
        future = process_single_concept.submit(
            wikibase_id=wikibase_id,
            passages_dataset=passages_dataset,
            passages_embeddings=passages_embeddings,
            embedding_model=embedding_model,
            wikibase_config=wikibase_config,
            argilla_config=argilla_config,
            s3_client=s3_client,
            aws_env=config.aws_env,
            track_and_upload=True,
        )
        concept_futures.append(future)

    logger.info("Waiting for all concept inference tasks to complete...")
    wait(concept_futures)

    # Track completion and collect results
    collected_results = []
    for future in concept_futures:
        try:
            result = future.result()
            collected_results.append(result)
        except Exception as e:
            logger.error(f"Unexpected error collecting result: {str(e)}")
            continue

    # Log summary results
    logger.info("Completed processing all concepts")
    successful_results = [r for r in collected_results if r.get("status") == "success"]
    failed_results = [r for r in collected_results if r.get("status") == "failed"]

    # Log successful results
    for result in sorted(successful_results, key=lambda x: x["concept_id"]):
        percentage = (
            (result["n_positive_passages"] / result["n_passages"] * 100)
            if result["n_passages"] > 0
            else 0.0
        )
        logger.info(
            f"✓ {result['concept_id']}: "
            f"{result['n_positive_passages']}/{result['n_passages']} "
            f"({percentage:.2f}%) - {result['output_prefix']}"
        )

    # Log failed results
    if failed_results:
        logger.warning(f"⚠️  {len(failed_results)} concepts failed to process")
        for failed in failed_results:
            logger.error(
                f"✗ {failed['concept_id']}: {failed.get('error', 'Unknown error')}"
            )

    logger.info(
        f"Successfully processed {len(successful_results)}/{len(collected_results)} concepts"
    )

    return collected_results
