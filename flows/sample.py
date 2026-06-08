from io import BytesIO
from typing import Annotated, Optional

import pandas as pd
import wandb
from prefect import flow, task
from prefect.logging import get_run_logger
from pydantic import Field, SecretStr

from flows.build_dataset_flow import COMBINED_S3_KEY, SAMPLED_S3_KEY
from flows.config import Config
from knowledge_graph.cloud import AwsEnv, get_async_session, get_aws_ssm_param
from knowledge_graph.identifiers import WikibaseID
from scripts.sample import run_sampling


@task
async def load_dataset_from_s3(
    dataset_name: str,
    config: Config,
    aws_env: AwsEnv,
) -> pd.DataFrame:
    logger = get_run_logger()

    if dataset_name == "balanced":
        dataset_filename = SAMPLED_S3_KEY
    elif dataset_name == "combined":
        dataset_filename = COMBINED_S3_KEY
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    logger.info(
        f"Loading {dataset_name} dataset from s3://{config.dataset_s3_bucket}/{dataset_filename}"
    )
    try:
        session = get_async_session(aws_env=aws_env, region_name=config.bucket_region)
        async with session.client("s3") as s3_client:
            response = await s3_client.get_object(
                Bucket=config.dataset_s3_bucket, Key=dataset_filename
            )
            body = await response["Body"].read()
    except Exception as e:
        logger.error(
            f"Failed to load dataset from s3://{config.dataset_s3_bucket}/{dataset_filename}: {e}"
        )
        raise

    try:
        dataset = pd.read_feather(BytesIO(body))
    except Exception as e:
        logger.error(f"Failed to parse feather file from S3: {e}")
        raise

    logger.info(f"Loaded {len(dataset):,} passages")
    return dataset


@task
async def run_sampling_task(
    wikibase_id: WikibaseID,
    dataset: pd.DataFrame,
    dataset_name: str,
    sample_size: int,
    min_negative_proportion: float,
    corpus_types_include: list[str] | None,
    corpus_types_exclude: list[str] | None,
    max_size_to_sample_from: int,
    max_negative_proportion: float | None,
    track_and_upload: bool,
    concept_override: list[str] | None,
    wikibase_username: str | None,
    wikibase_password: str | None,
    wikibase_url: str | None,
) -> str | None:
    import asyncio

    logger = get_run_logger()
    logger.info(
        f"Sampling {sample_size} passages for {wikibase_id} "
        f"(dataset={dataset_name}, track_and_upload={track_and_upload})"
    )
    try:
        return await asyncio.to_thread(
            run_sampling,
            wikibase_id=wikibase_id,
            dataset=dataset,
            dataset_name=dataset_name,
            sample_size=sample_size,
            min_negative_proportion=min_negative_proportion,
            corpus_types_include=corpus_types_include,
            corpus_types_exclude=corpus_types_exclude,
            max_size_to_sample_from=max_size_to_sample_from,
            max_negative_proportion=max_negative_proportion,
            track_and_upload=track_and_upload,
            concept_override=concept_override,
            wikibase_username=wikibase_username,
            wikibase_password=wikibase_password,
            wikibase_url=wikibase_url,
        )
    except Exception as e:
        logger.error(f"Sampling failed for {wikibase_id}: {e}")
        raise


@task
def login_to_wandb(aws_env: AwsEnv) -> None:
    logger = get_run_logger()
    logger.info("Logging in to W&B")
    try:
        wandb_api_key = SecretStr(get_aws_ssm_param("WANDB_API_KEY", aws_env=aws_env))
        wandb.login(key=wandb_api_key.get_secret_value())
    except Exception as e:
        logger.error(f"W&B login failed: {e}")
        raise
    logger.info("W&B login successful")


@flow
async def sample(
    wikibase_id: Annotated[
        WikibaseID,
        Field(description="The Wikibase ID of the concept to sample passages for"),
    ],
    dataset_name: Annotated[
        str,
        Field(
            description="Dataset to use",
            json_schema_extra={"enum": ["balanced", "combined"]},
        ),
    ] = "balanced",
    sample_size: Annotated[
        int,
        Field(description="The number of passages to sample"),
    ] = 130,
    min_negative_proportion: Annotated[
        float,
        Field(description="The minimum proportion of negative samples to take"),
    ] = 0.1,
    corpus_types_include: Annotated[
        Optional[list[str]],
        Field(
            description="Corpus types to include. Can be specified multiple times. If not set, all types are included.",
        ),
    ] = None,
    corpus_types_exclude: Annotated[
        Optional[list[str]],
        Field(
            description="Corpus types to exclude. Can be specified multiple times.",
        ),
    ] = None,
    max_size_to_sample_from: Annotated[
        int,
        Field(
            description="Maximum number of passages to load from the dataset before sampling"
        ),
    ] = 500_000,
    max_negative_proportion: Annotated[
        Optional[float],
        Field(
            description="Maximum proportion of the sample that can be negative. If not set, fills remaining sample_size after positives."
        ),
    ] = None,
    track_and_upload: Annotated[
        bool,
        Field(
            description="Whether to track the run and upload the labelled passages to W&B"
        ),
    ] = True,
    concept_override: Annotated[
        Optional[list[str]],
        Field(
            description="Concept property overrides in key=value format. Can be specified multiple times.",
        ),
    ] = None,
    aws_env: AwsEnv = AwsEnv.production,
    config: Optional[Config] = None,
) -> str | None:
    """
    Evenly sample passages for concepts from a dataset stored in S3.

    Wraps scripts.sample.run_sampling, handling S3 dataset loading and
    credential setup from AWS SSM.

    Returns the W&B artifact path (e.g. 'climatepolicyradar/Q123/labelled-passages:v3')
    if track_and_upload is True, otherwise None.
    """
    if not config:
        config = await Config.create()

    if track_and_upload:
        login_to_wandb(aws_env=aws_env)

    dataset = await load_dataset_from_s3(
        dataset_name=dataset_name,
        config=config,
        aws_env=aws_env,
    )

    return await run_sampling_task(
        wikibase_id=wikibase_id,
        dataset=dataset,
        dataset_name=dataset_name,
        sample_size=sample_size,
        min_negative_proportion=min_negative_proportion,
        corpus_types_include=corpus_types_include,
        corpus_types_exclude=corpus_types_exclude,
        max_size_to_sample_from=max_size_to_sample_from,
        max_negative_proportion=max_negative_proportion,
        track_and_upload=track_and_upload,
        concept_override=concept_override,
        wikibase_username=config.wikibase_username,
        wikibase_password=config.wikibase_password.get_secret_value()
        if config.wikibase_password
        else None,
        wikibase_url=config.wikibase_url,
    )
