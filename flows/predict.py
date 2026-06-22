"""
Ad-hoc prediction flows: Prefect orchestration over the prediction operation.

Two flows, both resolving credentials/config and then calling into
`knowledge_graph.operations.predict`:
- `predict_adhoc` — predict on passages from a W&B artifact (or a local .jsonl file). This
  is the deployed flow.
- `predict_document_passages` — load passages for specific document IDs from Snowflake, then
  predict. Resolves Snowflake key-pair credentials from SSM (so it can run in the cloud).
  Registered as an on-demand deployment in deployments.py.

The reusable logic (`run_prediction`, `load_passages_from_snowflake`,
`deduplicate_labelled_passages`) lives in `knowledge_graph.operations.predict` and can be
called directly for programmatic / ad-hoc use without Prefect. For local runs, `just predict`
and `just predict-documents` wrap these without going through Prefect.
"""

import os
from pathlib import Path

import wandb
from prefect import flow

from flows.config import Config
from knowledge_graph.cloud import AwsEnv
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.operations.predict import (
    load_passages_from_snowflake,
    run_prediction,
)
from knowledge_graph.operations.snowflake import get_snowflake_credentials


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

    if config.openrouter_api_key:
        os.environ["OPENROUTER_API_KEY"] = config.openrouter_api_key.get_secret_value()

    return config


@flow()
async def predict_adhoc(
    wikibase_id: WikibaseID,
    classifier_wandb_path: str,
    labelled_passages_wandb_path: str | None = None,
    labelled_passages_path: Path | None = None,
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
    """
    Run prediction on a single classifier.

    Passages come from a W&B artifact (`labelled_passages_wandb_path`, used by the
    deployment) or a local .jsonl file (`labelled_passages_path`, local runs only).
    """
    await _set_up_prediction_environment(config=config, aws_env=aws_env)

    return await run_prediction(
        wikibase_id=wikibase_id,
        classifier_wandb_path=classifier_wandb_path,
        labelled_passages_wandb_path=labelled_passages_wandb_path,
        labelled_passages_path=labelled_passages_path,
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
async def predict_document_passages(
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
    """Load passages for specific document IDs from Snowflake and run a classifier on them."""
    await _set_up_prediction_environment(config=config, aws_env=aws_env)

    snowflake_account, snowflake_user, snowflake_private_key = (
        get_snowflake_credentials(aws_env)
    )
    passages = load_passages_from_snowflake(
        document_ids,
        snowflake_user=snowflake_user,
        snowflake_private_key=snowflake_private_key.get_secret_value(),
        snowflake_account=snowflake_account,
    )
    await run_prediction(
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
