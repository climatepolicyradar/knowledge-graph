import importlib.metadata
import os
from pathlib import Path

import coiled
import wandb
from prefect import flow, task
from prefect.blocks.system import JSON
from prefect.docker.docker_image import DockerImage
from prefect.logging import get_run_logger
from pydantic import SecretStr

from flows.inference import Config, load_classifier
from scripts.cloud import PROJECT_NAME, AwsEnv, generate_deployment_name
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
    classifier = await load_classifier(
        run,
        config,
        classifier_name,
        classifier_alias,
    )
    logger.info(
        f"Loaded classifier with name: {classifier_name}, and alias: {classifier_alias}"
    )

    results: list[tuple[str, list[Span]]] = []
    for idx, text in enumerate(batch):
        spans: list[Span] = text_inference(text, classifier) or []
        print(f"Passage {idx}: '{text}' -> {spans}")
        results.append((text, spans))

    return results


# Coiled inference deployment (for testing)
sandbox_aws_env = AwsEnv("sandbox")
docker_registry = os.environ["DOCKER_REGISTRY"]
docker_repository = os.getenv("DOCKER_REPOSITORY", PROJECT_NAME)
version = importlib.metadata.version(PROJECT_NAME)
image_name = os.path.join(docker_registry, docker_repository)
default_variables = JSON.load(
    f"default-job-variables-prefect-mvp-{sandbox_aws_env}"
).value
job_variables = {**default_variables, **{}}

_ = run_classifier_inference_on_a_gpu__coiled_spike.deploy(
    generate_deployment_name(
        run_classifier_inference_on_a_gpu__coiled_spike.name, sandbox_aws_env
    ),
    work_pool_name="example-coiled-pool",
    version=version,
    image=DockerImage(
        name=image_name,
        tag=version,
        dockerfile="Dockerfile",
    ),
    job_variables=job_variables,
    tags=[f"repo:{docker_repository}", f"awsenv:{sandbox_aws_env}"],
    description="Run concept classifier inference on a GPU for a batch of documents",
    build=False,
    push=False,
)
