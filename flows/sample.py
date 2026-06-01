from io import BytesIO
from typing import Annotated, Optional

import pandas as pd
import wandb
from prefect import flow
from pydantic import Field, SecretStr

from flows.config import Config
from knowledge_graph.cloud import AwsEnv, get_async_session, get_aws_ssm_param
from knowledge_graph.identifiers import WikibaseID
from scripts.sample import run_sampling


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
) -> None:
    """
    Evenly sample passages for concepts from a dataset stored in S3.

    Wraps scripts.sample.run_sampling, handling S3 dataset loading and
    credential setup from AWS SSM.
    """
    if not config:
        config = await Config.create()

    if track_and_upload:
        wandb_api_key = SecretStr(get_aws_ssm_param("WANDB_API_KEY", aws_env=aws_env))
        wandb.login(key=wandb_api_key.get_secret_value())

    if dataset_name == "balanced":
        dataset_filename = "sampled_dataset.feather"
    elif dataset_name == "combined":
        dataset_filename = "combined_dataset.feather"
    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    # Load dataset from S3
    session = get_async_session(aws_env=aws_env, region_name=config.bucket_region)
    async with session.client("s3") as s3_client:
        response = await s3_client.get_object(
            Bucket=config.dataset_s3_bucket, Key=dataset_filename
        )
        body = await response["Body"].read()

    dataset = pd.read_feather(BytesIO(body))

    run_sampling(
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
