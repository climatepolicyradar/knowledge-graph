"""
Prefect deployment script

Used to create server side representation of prefect flows, triggers, config, etc.
See: https://docs-2.prefect.io/latest/concepts/deployments/
"""

import importlib.metadata
import os
from typing import Any, ParamSpec, TypeVar

from prefect.blocks.system import JSON
from prefect.docker.docker_image import DockerImage
from prefect.flows import Flow
from prefect.schedules import Cron, Schedule

from flows.aggregate import (
    aggregate,
    aggregate_batch_of_documents,
)
from flows.data_backup import data_backup
from flows.deploy_static_sites import deploy_static_sites
from flows.full_pipeline import full_pipeline
from flows.index import (
    index,
    index_batch_of_documents,
)
from flows.inference import (
    inference,
    inference_batch_of_documents_cpu,
    inference_batch_of_documents_gpu,
)
from flows.utils import JsonDict
from flows.wikibase_to_s3 import wikibase_to_s3
from src.cloud import PROJECT_NAME, AwsEnv, generate_deployment_name

MEGABYTES_PER_GIGABYTE = 1024
DEFAULT_FLOW_VARIABLES = {
    "cpu": MEGABYTES_PER_GIGABYTE * 4,
    "memory": MEGABYTES_PER_GIGABYTE * 16,
    "ephemeralStorage": {"sizeInGiB": 50},
}


def get_schedule_for_env(
    aws_env: AwsEnv,
    env_schedules: dict[AwsEnv, str] | None,
    env_parameters: dict[AwsEnv, JsonDict],
) -> Schedule | None:
    """Creates a cron schedule from a env schedule mapping"""
    if not env_schedules:
        return None

    if env_schedule := env_schedules.get(aws_env):
        return Cron(
            env_schedule,
            timezone="Europe/London",
            active=True,
            parameters=env_parameters.get(aws_env),
        )
    else:
        return None


# Match what Prefect uses for Flows:
#
# > .. we use the generic type variables `P` and `R` for "Parameters"
# > and "Returns" respectively.
P = ParamSpec("P")
R = TypeVar("R")


def create_deployment(
    flow: Flow[P, R],
    description: str,
    gpu: bool = False,
    flow_variables: dict[str, Any] = DEFAULT_FLOW_VARIABLES,
    env_schedules: dict[AwsEnv, str] | None = None,
    extra_tags: list[str] = [],
    env_parameters: dict[AwsEnv, JsonDict] = {},
) -> None:
    """Create a deployment for the specified flow"""
    aws_env = AwsEnv(os.environ["AWS_ENV"])
    version = importlib.metadata.version(PROJECT_NAME)
    flow_name = flow.name
    docker_registry = os.environ["DOCKER_REGISTRY"]
    docker_repository = os.getenv("DOCKER_REPOSITORY", PROJECT_NAME)
    image_name = os.path.join(docker_registry, docker_repository)

    if gpu:
        if aws_env == AwsEnv.production:
            aws_env_str = AwsEnv.production.name
        else:
            aws_env_str = str(aws_env)

        work_pool_name = f"coiled-{aws_env_str}"
        work_queue_name = "default"
        default_job_variables = JSON.load(
            f"coiled-default-job-variables-prefect-mvp-{aws_env}"
        ).value
    else:
        work_pool_name = f"mvp-{aws_env}-ecs"
        work_queue_name = f"mvp-{aws_env}"
        default_job_variables = JSON.load(
            f"default-job-variables-prefect-mvp-{aws_env}"
        ).value

    job_variables = {**default_job_variables, **flow_variables}
    tags = [f"repo:{docker_repository}", f"awsenv:{aws_env}"] + extra_tags
    schedule = get_schedule_for_env(
        aws_env,
        env_schedules,
        env_parameters,
    )

    _ = flow.deploy(
        generate_deployment_name(flow_name, aws_env),
        work_pool_name=work_pool_name,
        work_queue_name=work_queue_name,
        version=version,
        image=DockerImage(
            name=image_name,
            tag=version,
            dockerfile="Dockerfile",
        ),
        job_variables=job_variables,
        tags=tags,
        description=description,
        schedule=schedule,
        build=False,
        push=False,
    )


# Inference

create_deployment(
    flow=inference,
    description="Run concept classifier inference on document passages",
    extra_tags=["type:entry"],
)

create_deployment(
    flow=inference_batch_of_documents_cpu,
    description="Run concept classifier inference on a batch of documents with CPU compute",
    extra_tags=["type:sub"],
    gpu=False,
)

create_deployment(
    flow=inference_batch_of_documents_gpu,
    description="Run concept classifier inference on a batch of documents with GPU compute",
    extra_tags=["type:sub"],
    gpu=True,
)

# Aggregate inference results

create_deployment(
    flow=aggregate_batch_of_documents,
    description="Aggregate inference results for a batch of documents",
    flow_variables={
        "cpu": MEGABYTES_PER_GIGABYTE * 16,
        "memory": MEGABYTES_PER_GIGABYTE * 64,
    },
    extra_tags=["type:sub"],
)

create_deployment(
    flow=aggregate,
    description="Aggregate inference results, through coordinating batches of documents",
    extra_tags=["type:entry"],
)

# Index

create_deployment(
    flow=index_batch_of_documents,
    description="Run passage indexing for a batch of documents from S3 to Vespa",
    extra_tags=["type:sub"],
)

create_deployment(
    flow=index,
    description="Run passage indexing for documents from S3 to Vespa",
    extra_tags=["type:entry"],
)

# Orchestrate full pipeline

create_deployment(
    flow=full_pipeline,
    description="Run the full Knowledge Graph Pipeline",
    extra_tags=["type:end_to_end"],
    env_schedules={
        # Run it daily, on work days, to validate it works, or to
        # surface issues.
        AwsEnv.staging: "0 7 * * MON-THU",
    },
    env_parameters={
        AwsEnv.staging: JsonDict(
            {
                "document_ids": [
                    "AF.document.061MCLAR.n0000_translated_en",
                    "CCLW.executive.10512.5360",
                ],
                "classifier_specs": [
                    # CPU-based
                    {"name": "Q708", "alias": "v6"},
                    # GPU-based
                    {"name": "Q1651", "alias": "v1"},
                ],
            }
        ),
    },
)

# Wikibase

create_deployment(
    flow=wikibase_to_s3,
    description="Upload concepts from Wikibase to S3",
    # Temporarily disabled for stability
    #     env_schedules={
    #         AwsEnv.production: "0 9 * * TUE,THU",
    #         AwsEnv.staging: "0 15 2 * *",
    #         AwsEnv.sandbox: "0 15 1 * *",
    #         # Not needed in labs
    #         # AwsEnv.labs: "0 15 3 * *",
    #     },
)

# Deploy static sites

create_deployment(
    flow=deploy_static_sites,
    description="Deploy our static sites to S3",
    env_schedules={
        AwsEnv.labs: "0 0 * * *",  # Every day at midnight
    },
)

# Data backup

create_deployment(
    flow=data_backup,
    description="Deploy all Argilla datasets to Huggingface",
    env_schedules={
        AwsEnv.labs: "0 0 * * *",  # Every day at midnight
    },
)
