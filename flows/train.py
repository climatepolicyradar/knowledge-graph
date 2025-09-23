import boto3
from prefect import flow

import wandb
from flows.config import Config
from knowledge_graph.cloud import AwsEnv
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.wikibase import WikibaseConfig
from scripts.train import run_training


@flow(log_prints=True)
async def train_on_gpu(
    wikibase_id: WikibaseID,
    track_and_upload: bool = False,
    aws_env: AwsEnv = AwsEnv.labs,
    evaluate: bool = True,
    config: Config | None = None,
):
    """Trigger the training script in prefect using coiled."""
    if not config:
        config = await Config.create()

    if (
        not config.wandb_api_key
        or not config.wikibase_username
        or not config.wikibase_password
        or not config.wikibase_url
    ):
        raise ValueError("Missing values in config.")

    wandb.login(key=config.wandb_api_key.get_secret_value())

    wikibase_config = WikibaseConfig(
        username=config.wikibase_username,
        password=config.wikibase_password,
        url=config.wikibase_url,
    )

    s3_client = boto3.client("s3", region_name=config.bucket_region)

    return await run_training(
        wikibase_id=wikibase_id,
        track_and_upload=track_and_upload,
        aws_env=aws_env,
        wikibase_config=wikibase_config,
        s3_client=s3_client,
        evaluate=evaluate,
    )
