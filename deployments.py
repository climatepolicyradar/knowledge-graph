"""
Prefect deployment script

Used to create server side representation of prefect flows, triggers, config, etc.
See: https://docs-2.prefect.io/latest/concepts/deployments/
"""

import importlib.metadata
import os
from typing import Any

from prefect.blocks.system import JSON
from prefect.deployments.runner import DeploymentImage
from prefect.flows import Flow

from flows.inference import classifier_inference

MEGABYTES_PER_GIGABYTE = 1024


def create_deployment(
    project_name: str, flow: Flow, description: str, flow_variables: dict[str, Any]
) -> None:
    """Create a deployment for the specified flow"""
    aws_env = os.getenv("AWS_ENV", "sandbox")
    version = importlib.metadata.version(project_name)
    flow_name = flow.name
    docker_registry = os.environ["DOCKER_REGISTRY"]
    docker_repository = os.getenv("DOCKER_REPOSITORY", project_name)
    image_name = os.path.join(docker_registry, docker_repository)

    default_variables = JSON.load(f"default-job-variables-prefect-mvp-{aws_env}").value
    job_variables = {**default_variables, **flow_variables}

    _ = classifier_inference.deploy(
        f"{project_name}-{flow_name}-{aws_env}",
        work_pool_name=f"mvp-{aws_env}-ecs",
        version=version,
        image=DeploymentImage(
            name=image_name,
            tag=version,
            dockerfile="Dockerfile",
        ),
        work_queue_name=f"mvp-{aws_env}",
        job_variables=job_variables,
        tags=[f"repo:{docker_repository}", f"awsenv:{aws_env}"],
        description=description,
        build=False,
        push=False,
    )


# Inference
create_deployment(
    project_name="knowledge-graph",
    flow=classifier_inference,
    description="Run concept classifier inference on document passages",
    flow_variables={
        "cpu": MEGABYTES_PER_GIGABYTE * 4,
        "memory": MEGABYTES_PER_GIGABYTE * 16,
        "env": {
            "CACHE_BUCKET": os.environ["CACHE_BUCKET"],
            "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
        },
        "ephemeralStorage": {"sizeInGiB": 50},
    },
)
