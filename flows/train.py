import asyncio
from pathlib import Path
from typing import Any, Optional

import boto3
import wandb
import yaml
from prefect import flow

from flows.config import Config
from flows.utils import get_logger
from knowledge_graph.cloud import AwsEnv
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelling import ArgillaConfig
from knowledge_graph.wikibase import WikibaseConfig
from scripts.train import run_training


async def _setup_training_environment(
    config: Config | None,
    aws_env: AwsEnv,
) -> tuple[Config, WikibaseConfig, ArgillaConfig, Any]:
    """
    Set up the common config for classifier training

    :param config: Optional pre-configured Config object
    :param aws_env: AWS environment to use for creating the S3 client
    """
    if not config:
        config = await Config.create()

    if (
        not config.wandb_api_key
        or not config.wikibase_username
        or not config.wikibase_password
        or not config.wikibase_url
        or not config.argilla_api_key
        or not config.argilla_api_url
    ):
        raise ValueError("Missing values in config.")

    wandb.login(key=config.wandb_api_key.get_secret_value())

    wikibase_config = WikibaseConfig(
        username=config.wikibase_username,
        password=config.wikibase_password,
        url=config.wikibase_url,
    )

    argilla_config = ArgillaConfig(
        api_key=config.argilla_api_key,
        url=config.argilla_api_url,
    )

    session = boto3.session.Session(
        profile_name=aws_env.value,
        region_name=config.bucket_region,
    )
    s3_client = session.client("s3")

    return config, wikibase_config, argilla_config, s3_client


@flow()
async def train_on_gpu(
    wikibase_id: WikibaseID,
    track_and_upload: bool = False,
    aws_env: AwsEnv = AwsEnv.labs,
    evaluate: bool = True,
    classifier_type: Optional[str] = None,
    classifier_kwargs: Optional[dict[str, Any]] = None,
    concept_overrides: Optional[dict[str, Any]] = None,
    training_data_wandb_path: Optional[str] = None,
    limit_training_samples: Optional[int] = None,
    config: Config | None = None,
):
    """Trigger the training script in prefect using coiled."""
    _, wikibase_config, argilla_config, s3_client = await _setup_training_environment(
        config=config, aws_env=aws_env
    )

    return await run_training(
        wikibase_id=wikibase_id,
        track_and_upload=track_and_upload,
        aws_env=aws_env,
        wikibase_config=wikibase_config,
        argilla_config=argilla_config,
        s3_client=s3_client,
        evaluate=evaluate,
        classifier_type=classifier_type,
        classifier_kwargs=classifier_kwargs,
        concept_overrides=concept_overrides,
        training_data_wandb_path=training_data_wandb_path,
        limit_training_samples=limit_training_samples,
    )


def _load_wikibase_ids_from_config(
    config_file_path: str = "vibe-checker/config.yml",
) -> list[WikibaseID]:
    """Load concept IDs from the configuration file."""
    logger = get_logger()
    config_path = Path(config_file_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_file_path}")

    try:
        with open(config_file_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        # make sure the contents are a list of valid Wikibase IDs
        wikibase_ids = [WikibaseID(id) for id in config]
        # make sure the list of wikibase ids is unique
        wikibase_ids = list(set(wikibase_ids))
    except yaml.YAMLError as e:
        raise ValueError(
            "The config file should be valid YAML containing a list of Wikibase IDs"
        ) from e
    except Exception as e:
        raise ValueError(f"Failed to load config file: {e}") from e

    if not wikibase_ids:
        raise ValueError("No concepts found in the config")

    logger.info(f"Loaded {len(wikibase_ids)} Wikibase IDs from config")
    return wikibase_ids


@flow()
async def train_from_config(
    track_and_upload: bool = True,
    aws_env: AwsEnv = AwsEnv.labs,
    config_file_path: str = "vibe-checker/config.yml",
    config: Config | None = None,
    force: bool = False,
    concurrency_limit: int = 3,
) -> list[Any]:
    """
    Train classifiers for all concepts listed in the `vibe-checker/config.yml` file.

    Reads Wikibase IDs from the config file and runs training for each concept in
    parallel. Training results and evaluation metrics are uploaded to W&B.

    :param track_and_upload: Whether to track training runs and upload artifacts to W&B
    :param aws_env: AWS environment to use for S3 uploads
    :param config_file_path: Path to the config file containing Wikibase IDs
    :param config: Optional pre-configured Config object. If not provided, will be created.
    :param force: If True, force re-training even if classifier already exists in W&B
    :param concurrency_limit: Maximum number of concurrent training tasks (default: 3)
    :return: List of trained classifiers
    """
    logger = get_logger()
    logger.info("Starting training from config file")

    _, wikibase_config, argilla_config, s3_client = await _setup_training_environment(
        config=config, aws_env=aws_env
    )

    # Load Wikibase IDs from config file
    wikibase_ids = _load_wikibase_ids_from_config(config_file_path)
    logger.info(
        f"Training classifiers for {len(wikibase_ids)} concepts "
        f"(max {concurrency_limit} at a time)"
    )

    # Create a semaphore to limit concurrent training tasks
    semaphore = asyncio.Semaphore(concurrency_limit)

    async def run_training_with_concurrency_limit(wikibase_id: WikibaseID) -> Any:
        """Wrapper to run training with concurrency limit."""
        async with semaphore:
            return await run_training(
                wikibase_id=wikibase_id,
                track_and_upload=track_and_upload,
                aws_env=aws_env,
                wikibase_config=wikibase_config,
                argilla_config=argilla_config,
                s3_client=s3_client,
                evaluate=True,
                classifier_type=None,  # Use ClassifierFactory default
                classifier_kwargs=None,
                concept_overrides=None,
                training_data_wandb_path=None,
                limit_training_samples=None,
                force=force,
            )

    # Create tasks with concurrency limit
    training_tasks = [
        run_training_with_concurrency_limit(wikibase_id) for wikibase_id in wikibase_ids
    ]

    results = await asyncio.gather(*training_tasks, return_exceptions=True)

    # Log results
    successful = [r for r in results if not isinstance(r, Exception)]
    failed = [r for r in results if isinstance(r, Exception)]

    logger.info(f"Successfully trained {len(successful)}/{len(results)} classifiers")
    if failed:
        logger.warning(f"Failed to train {len(failed)} classifiers")
        for i, error in enumerate(failed):
            logger.error(f"Training error {i + 1}: {error}")

    return results
