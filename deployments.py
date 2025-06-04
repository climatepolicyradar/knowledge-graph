"""
Prefect deployment script

Used to create server side representation of prefect flows, triggers, config, etc.
See: https://docs-2.prefect.io/latest/concepts/deployments/
"""

import importlib.metadata
import os
from typing import Any

from prefect.blocks.system import JSON
from prefect.client.schemas.actions import DeploymentScheduleCreate
from prefect.client.schemas.schedules import construct_schedule
from prefect.deployments.schedules import create_deployment_schedule_create
from prefect.docker.docker_image import DockerImage
from prefect.flows import Flow

import flows.boundary as boundary
import flows.deindex as deindex
from flows.aggregate_inference_results import aggregate_inference_results
from flows.count_family_document_concepts import (
    count_family_document_concepts,
    load_update_document_concepts_counts,
)
from flows.data_backup import data_backup
from flows.deploy_static_sites import deploy_static_sites
from flows.index import (
    index_labelled_passages_from_s3_to_vespa,
)
from flows.index_from_aggregate_results import (
    run_indexing_from_aggregate_results,
)
from flows.inference import (
    classifier_inference,
    run_classifier_inference_on_batch_of_documents,
)
from flows.wikibase_to_s3 import wikibase_to_s3
from scripts.cloud import PROJECT_NAME, AwsEnv, generate_deployment_name

MEGABYTES_PER_GIGABYTE = 1024
DEFAULT_FLOW_VARIABLES = {
    "cpu": MEGABYTES_PER_GIGABYTE * 4,
    "memory": MEGABYTES_PER_GIGABYTE * 16,
    "ephemeralStorage": {"sizeInGiB": 50},
}


def get_schedule_for_env(
    aws_env: AwsEnv, env_schedules: dict[AwsEnv, str] | None
) -> list[DeploymentScheduleCreate] | None:
    """Creates a cron schedule from a env schedule mapping"""
    if not env_schedules:
        return None

    if env_schedules.get(aws_env):
        return [
            create_deployment_schedule_create(
                construct_schedule(
                    cron=env_schedules.get(aws_env),
                    timezone="Europe/London",
                ),
                active=True,
            )
        ]
    else:
        return None


def create_deployment(
    flow: Flow,
    description: str,
    flow_variables: dict[str, Any] = DEFAULT_FLOW_VARIABLES,
    env_schedules: dict[AwsEnv, str] | None = None,
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

    schedule = get_schedule_for_env(aws_env, env_schedules)

    _ = flow.deploy(
        generate_deployment_name(flow_name, aws_env),
        work_pool_name=f"mvp-{aws_env}-ecs",
        version=version,
        image=DockerImage(
            name=image_name,
            tag=version,
            dockerfile="Dockerfile",
        ),
        work_queue_name=f"mvp-{aws_env}",
        job_variables=job_variables,
        tags=[f"repo:{docker_repository}", f"awsenv:{aws_env}"],
        description=description,
        schedules=schedule,
        build=False,
        push=False,
    )


# Inference

create_deployment(
    flow=classifier_inference,
    description="Run concept classifier inference on document passages",
)

create_deployment(
    flow=run_classifier_inference_on_batch_of_documents,
    description="Run concept classifier inference on a batch of documents",
)

# Aggregate inference results

create_deployment(
    flow=aggregate_inference_results,
    description="Aggregate inference results",
    flow_variables={
        "cpu": MEGABYTES_PER_GIGABYTE * 16,
        "memory": MEGABYTES_PER_GIGABYTE * 64,
    },
)

# Boundary

create_deployment(
    flow=boundary.run_partial_updates_of_concepts_for_batch,
    description="Run partial updates of labelled passages stored in S3 into Vespa for a batch of documents",
)

# Index

create_deployment(
    flow=boundary.run_partial_updates_of_concepts_for_document_passages,
    description="Co-ordinate updating inference results for concepts in Vespa",
)

create_deployment(
    flow=index_labelled_passages_from_s3_to_vespa,
    description="Run partial updates of labelled passages stored in S3 into Vespa",
)

create_deployment(
    flow=run_indexing_from_aggregate_results,
    description="Index aggregated inference results from S3 into Vespa",
)

# De-index

create_deployment(
    flow=deindex.run_cleanup_objects_for_batch,
    description="Clean-up a concept's versions for a document",
)

create_deployment(
    flow=deindex.deindex_labelled_passages_from_s3_to_vespa,
    description="Run partial updates of labelled passages stored in S3 into Vespa",
)

# Concepts counting

create_deployment(
    flow=count_family_document_concepts,
    description="Update family documents in Vespa to include concepts' counts from S3",
)

create_deployment(
    flow=load_update_document_concepts_counts,
    description="Update 1 family document in Vespa to include concepts' counts from S3",
)

# Wikibase

create_deployment(
    flow=wikibase_to_s3,
    description="Upload concepts from Wikibase to S3",
    env_schedules={
        AwsEnv.production: "0 9 * * TUE,THU",
        AwsEnv.staging: "0 15 2 * *",
        AwsEnv.sandbox: "0 15 1 * *",
        # AwsEnv.labs: "0 15 3 * *",
    },
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
