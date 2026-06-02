"""
Prefect flow wrapping scripts/build_dataset.py.

Queries Snowflake for climate document passages, builds a combined (full corpus)
and a balanced sampled dataset, and uploads both as feather files to S3 for use
by the vibe checker and sampling flows.

Runs on a monthly schedule (see deployments.py). Data Scientists can pull the latest files
locally with: just build-dataset-download
"""

import asyncio
import io
from typing import cast

from mypy_boto3_s3.client import S3Client
from prefect import flow, task
from prefect.logging import get_run_logger
from pydantic import SecretStr

from flows.config import Config
from knowledge_graph.cloud import AwsEnv, get_aws_ssm_param, get_s3_client
from scripts.build_dataset import run_build_dataset

COMBINED_S3_KEY = "build_dataset/combined_dataset.feather"
SAMPLED_S3_KEY = "build_dataset/sampled_dataset.feather"

SNOWFLAKE_ACCOUNT_SSM = "/Snowflake/Account"
SNOWFLAKE_USER_SSM = "/Snowflake/ServiceUser/DbtBot/User"
SNOWFLAKE_PRIVATE_KEY_SSM = "/Snowflake/ServiceUser/DbtBot/PrivateKey"


async def _set_up_build_dataset_environment(
    config: Config | None,
    aws_env: AwsEnv,
) -> tuple[Config, str, str, SecretStr]:
    """
    Load shared config and Snowflake credentials from SSM.

    Returns (config, snowflake_account, snowflake_user, snowflake_private_key).
    """
    if not config:
        config = await Config.create()

    snowflake_account = get_aws_ssm_param(SNOWFLAKE_ACCOUNT_SSM, aws_env=aws_env)
    snowflake_user = get_aws_ssm_param(SNOWFLAKE_USER_SSM, aws_env=aws_env)
    snowflake_private_key = SecretStr(
        get_aws_ssm_param(SNOWFLAKE_PRIVATE_KEY_SSM, aws_env=aws_env)
    )

    return config, snowflake_account, snowflake_user, snowflake_private_key


@task
async def run_build_dataset_task(
    sampled_dataset_target_num_rows: int,
    snowflake_user: str,
    snowflake_private_key: str,
    snowflake_account: str,
):
    return await asyncio.to_thread(
        run_build_dataset,
        n=sampled_dataset_target_num_rows,
        snowflake_user=snowflake_user,
        snowflake_private_key=snowflake_private_key,
        snowflake_account=snowflake_account,
    )


@flow(
    name="kg-build-dataset",
)
async def build_dataset_flow(
    sampled_dataset_target_num_rows: int = 10_000,
    aws_env: AwsEnv = AwsEnv.production,
    config: Config | None = None,
) -> None:
    """
    Build combined and sampled passage datasets from Snowflake and upload to S3.

    :param sampled_dataset_target_num_rows: Target number of rows in the sampled dataset. Defaults to 10,000.
    :param aws_env: AWS environment for SSM and S3 access.
    :param config: Optional pre-built Config. Created from SSM if not provided.
    """
    logger = get_run_logger()

    (
        config,
        snowflake_account,
        snowflake_user,
        snowflake_private_key,
    ) = await _set_up_build_dataset_environment(config=config, aws_env=aws_env)

    logger.info(f"Building dataset (n={sampled_dataset_target_num_rows})")

    combined_df, sampled_df = await run_build_dataset_task(
        sampled_dataset_target_num_rows=sampled_dataset_target_num_rows,
        snowflake_user=snowflake_user,
        snowflake_private_key=snowflake_private_key.get_secret_value(),
        snowflake_account=snowflake_account,
    )

    logger.info(
        f"Built datasets: combined={len(combined_df):,} rows, "
        f"sampled={len(sampled_df):,} rows"
    )

    for column in [
        "world_bank_region",
        "document_metadata.corpus_type_name",
        "translated",
    ]:
        if column in sampled_df.columns:
            vc = sampled_df[column].value_counts()
            counts = ", ".join(f"{val}: {cnt:,}" for val, cnt in vc.items())
            logger.info(f"Sampled dataset — {column}: {counts}")

    s3_client = cast(S3Client, get_s3_client(aws_env, config.bucket_region))

    for df, key in [
        (combined_df, COMBINED_S3_KEY),
        (sampled_df, SAMPLED_S3_KEY),
    ]:
        buffer = io.BytesIO()
        df.to_feather(buffer)
        buffer.seek(0)
        s3_client.put_object(
            Bucket=config.dataset_s3_bucket, Key=key, Body=buffer.read()
        )
        logger.info(f"Uploaded to s3://{config.dataset_s3_bucket}/{key}")
