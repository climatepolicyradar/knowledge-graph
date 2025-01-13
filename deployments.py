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

from flows.index import index_labelled_passages_from_s3_to_vespa
from flows.inference import (
    classifier_inference,
    run_classifier_inference_on_batch_of_documents,
)
from scripts.cloud import PROJECT_NAME, AwsEnv, generate_deployment_name

MEGABYTES_PER_GIGABYTE = 1024


def create_deployment(
    flow: Flow,
    description: str,
    flow_variables: dict[str, Any],
) -> None:
    """Create a deployment for the specified flow"""
    aws_env = AwsEnv(os.getenv("AWS_ENV", "sandbox"))
    version = importlib.metadata.version(PROJECT_NAME)
    flow_name = flow.name
    docker_registry = os.environ["DOCKER_REGISTRY"]
    docker_repository = os.getenv("DOCKER_REPOSITORY", PROJECT_NAME)
    image_name = os.path.join(docker_registry, docker_repository)

    default_variables = JSON.load(f"default-job-variables-prefect-mvp-{aws_env}").value
    job_variables = {**default_variables, **flow_variables}

    _ = flow.deploy(
        generate_deployment_name(flow_name, aws_env),
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
    flow=classifier_inference,
    description="Run concept classifier inference on document passages",
    flow_variables={
        "cpu": MEGABYTES_PER_GIGABYTE * 4,
        "memory": MEGABYTES_PER_GIGABYTE * 16,
        "ephemeralStorage": {"sizeInGiB": 50},
    },
)

# Inference - Batch of Documents
create_deployment(
    flow=run_classifier_inference_on_batch_of_documents,
    description="Run concept classifier inference on a batch of documents",
    flow_variables={
        "cpu": MEGABYTES_PER_GIGABYTE * 4,
        "memory": MEGABYTES_PER_GIGABYTE * 16,
        "ephemeralStorage": {"sizeInGiB": 50},
    },
)

# Index
create_deployment(
    flow=index_labelled_passages_from_s3_to_vespa,
    description="Run partial updates of labelled passages stored in s3 into Vespa",
    flow_variables={
        "cpu": MEGABYTES_PER_GIGABYTE * 4,
        "memory": MEGABYTES_PER_GIGABYTE * 16,
        "ephemeralStorage": {"sizeInGiB": 50},
    },
)
