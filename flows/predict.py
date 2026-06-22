import os

import wandb
from prefect import flow

from flows.config import Config
from knowledge_graph.cloud import AwsEnv
from knowledge_graph.identifiers import WikibaseID
from scripts.predict import run_prediction


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
