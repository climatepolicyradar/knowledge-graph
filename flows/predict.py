import asyncio
import base64
import os

import wandb
from prefect import flow, task
from pydantic import SecretStr

from flows.build_dataset_flow import (
    SNOWFLAKE_ACCOUNT_SSM,
    SNOWFLAKE_PRIVATE_KEY_SSM,
    SNOWFLAKE_USER_SSM,
)
from flows.config import Config
from knowledge_graph.cloud import AwsEnv, get_aws_ssm_param
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from scripts.predict import load_passages_from_snowflake, run_prediction


async def _set_up_prediction_environment(
    config: Config | None,
    aws_env: AwsEnv,
) -> Config:
    """
    Set up the common config for classifier prediction.

    :param config: Optional pre-configured Config object
    :param aws_env: AWS environment to use for creating the S3 client
    """
    if not config:
        config = await Config.create()

    if config.wandb_api_key:
        wandb.login(key=config.wandb_api_key.get_secret_value())

    openrouter_api_key = SecretStr(
        get_aws_ssm_param("/OpenRouter/KGApiKey", aws_env=aws_env)
    )
    os.environ["OPENROUTER_API_KEY"] = openrouter_api_key.get_secret_value()

    return config


async def _get_snowflake_credentials(aws_env: AwsEnv) -> tuple[str, str, SecretStr]:
    """
    Load Snowflake key-pair credentials from SSM.

    Returns (snowflake_account, snowflake_user, snowflake_private_key).
    """
    snowflake_account = get_aws_ssm_param(SNOWFLAKE_ACCOUNT_SSM, aws_env=aws_env)
    snowflake_user = get_aws_ssm_param(SNOWFLAKE_USER_SSM, aws_env=aws_env)
    snowflake_private_key = SecretStr(
        base64.b64decode(
            get_aws_ssm_param(SNOWFLAKE_PRIVATE_KEY_SSM, aws_env=aws_env)
        ).decode("utf-8")
    )
    return snowflake_account, snowflake_user, snowflake_private_key


@task
async def load_passages_from_snowflake_task(
    document_ids: list[str],
    snowflake_user: str,
    snowflake_private_key: str,
    snowflake_account: str,
) -> list[LabelledPassage]:
    return await asyncio.to_thread(
        load_passages_from_snowflake,
        document_ids=document_ids,
        snowflake_user=snowflake_user,
        snowflake_private_key=snowflake_private_key,
        snowflake_account=snowflake_account,
    )


@flow()
async def predict_documents(
    document_ids: list[str],
    wikibase_id: WikibaseID,
    classifier_wandb_path: str,
    track_and_upload: bool = True,
    batch_size: int = 15,
    limit: int | None = None,
    deduplicate_inputs: bool = True,
    exclude_training_data: bool = True,
    prediction_threshold: float | None = None,
    stop_after_n_positives: int | None = None,
    restart_from_wandb_run: str | None = None,
    aws_env: AwsEnv = AwsEnv.production,
    config: Config | None = None,
) -> None:
    """Load passages for specific document IDs from Snowflake and run a classifier."""
    await _set_up_prediction_environment(config=config, aws_env=aws_env)

    (
        snowflake_account,
        snowflake_user,
        snowflake_private_key,
    ) = await _get_snowflake_credentials(aws_env)

    passages = await load_passages_from_snowflake_task(
        document_ids=document_ids,
        snowflake_user=snowflake_user,
        snowflake_private_key=snowflake_private_key.get_secret_value(),
        snowflake_account=snowflake_account,
    )

    return await run_prediction(
        wikibase_id=wikibase_id,
        classifier_wandb_path=classifier_wandb_path,
        input_passages=passages,
        track_and_upload=track_and_upload,
        batch_size=batch_size,
        limit=limit,
        deduplicate_inputs=deduplicate_inputs,
        exclude_training_data=exclude_training_data,
        prediction_threshold=prediction_threshold,
        stop_after_n_positives=stop_after_n_positives,
        restart_from_wandb_run=restart_from_wandb_run,
        aws_env=aws_env,
    )


@flow()
async def predict_adhoc(
    wikibase_id: WikibaseID,
    classifier_wandb_path: str,
    labelled_passages_wandb_path: str,
    track_and_upload: bool = True,
    batch_size: int = 15,
    limit: int | None = None,
    deduplicate_inputs: bool = True,
    exclude_training_data: bool = True,
    prediction_threshold: float | None = None,
    stop_after_n_positives: int | None = None,
    restart_from_wandb_run: str | None = None,
    aws_env: AwsEnv = AwsEnv.production,
    config: Config | None = None,
) -> None:
    """Run prediction on a single classifier using labelled passages from W&B."""
    await _set_up_prediction_environment(config=config, aws_env=aws_env)

    return await run_prediction(
        wikibase_id=wikibase_id,
        classifier_wandb_path=classifier_wandb_path,
        labelled_passages_wandb_path=labelled_passages_wandb_path,
        track_and_upload=track_and_upload,
        batch_size=batch_size,
        limit=limit,
        deduplicate_inputs=deduplicate_inputs,
        exclude_training_data=exclude_training_data,
        prediction_threshold=prediction_threshold,
        stop_after_n_positives=stop_after_n_positives,
        restart_from_wandb_run=restart_from_wandb_run,
        aws_env=aws_env,
    )
