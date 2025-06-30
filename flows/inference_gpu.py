import os
from pathlib import Path

import coiled
from prefect import flow, task
from prefect.logging import get_run_logger
from pydantic import SecretStr

import wandb

# Set the aws env to sandbox
os.environ["AWS_ENV"] = "sandbox"

from flows.inference import Config
from src.classifier import Classifier
from src.span import Span


@task(log_prints=True)
@coiled.function(memory="16 GiB", gpu=True)
def text_inference(text: str, classifier: Classifier) -> list[Span]:
    """Run classifier inference on a single text passage."""
    return classifier.predict(text)


@flow(log_prints=True)
async def run_classifier_inference_on_a_gpu__coiled_spike(
    batch: list[str],
    classifier_name: str,
    classifier_alias: str,
    config: Config | None = None,
) -> list[tuple[str, list[Span]]]:
    """
    Run classifier inference on a batch of documents.

    This reflects the unit of work that should be run in one of many paralellised
    docker containers.
    """
    logger = get_run_logger()

    if not config:
        config = await Config.create()

    config_json = config.to_json()

    config_json["wandb_api_key"] = (
        SecretStr(config_json["wandb_api_key"])
        if config_json["wandb_api_key"]
        else None
    )
    config_json["local_classifier_dir"] = Path(config_json["local_classifier_dir"])
    config = Config(**config_json)
    assert config.wandb_api_key

    _ = wandb.login(key=config.wandb_api_key.get_secret_value())
    run = wandb.init(
        entity=config.wandb_entity,
        job_type="concept_inference",
    )

    logger.info(
        f"Loading classifier with name: {classifier_name}, and alias: {classifier_alias}"
    )
    artifact = run.use_artifact(
        f"climatepolicyradar/{classifier_name}/TargetClassifier:{classifier_alias}",
        type="model",
    )
    artifact_dir = artifact.download()
    classifier = Classifier.load(os.path.join(artifact_dir, "model.pickle"))

    logger.info(
        f"Loaded classifier with name: {classifier_name}, and alias: {classifier_alias}"
    )

    results: list[tuple[str, list[Span]]] = []
    for idx, text in enumerate(batch):
        spans: list[Span] = text_inference(text, classifier) or []
        print(f"Passage {idx}: '{text}' -> {spans}")
        results.append((text, spans))

    return results
