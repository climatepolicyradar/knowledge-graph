"""
Prefect deployment script

Used to create server side representation of prefect flows, triggers, config, etc.
See: https://docs-2.prefect.io/latest/concepts/deployments/
"""

import importlib.metadata
import logging
import os
import subprocess
from typing import Any, ParamSpec, TypeVar

from prefect.client.schemas.objects import (
    ConcurrencyLimitConfig,
    ConcurrencyLimitStrategy,
)
from prefect.docker.docker_image import DockerImage
from prefect.flows import Flow
from prefect.schedules import Cron, Schedule
from prefect.variables import Variable

from flows.aggregate import aggregate, aggregate_batch_of_documents
from flows.classifiers_profiles import sync_classifiers_profiles
from flows.data_backup import data_backup
from flows.deploy_static_sites import deploy_static_sites
from flows.full_pipeline import full_pipeline
from flows.index import index, index_batch_of_documents
from flows.inference import (
    inference,
    inference_batch_of_documents_cpu,
    inference_batch_of_documents_gpu,
)
from flows.sync_concepts import sync_concepts
from flows.train import train_for_vibe_checker, train_on_gpu
from flows.update_neo4j import update_concepts
from flows.utils import JsonDict, get_logger
from flows.vibe_check import vibe_check_inference
from flows.wikibase_to_s3 import wikibase_to_s3
from knowledge_graph.cloud import PROJECT_NAME, AwsEnv, generate_deployment_name

MEGABYTES_PER_GIGABYTE = 1024
DEFAULT_FLOW_VARIABLES = {
    "cpu": MEGABYTES_PER_GIGABYTE * 4,
    "memory": MEGABYTES_PER_GIGABYTE * 16,
    "ephemeralStorage": {"sizeInGiB": 50},
    "match_latest_revision_in_family": True,
}

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler and set level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Add ch to logger
logger.addHandler(ch)


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
    concurrency_limit: int | ConcurrencyLimitConfig | None = None,
) -> None:
    """Create a deployment for the specified flow"""
    logger = get_logger()

    aws_env = AwsEnv(os.environ["AWS_ENV"])
    version = _version()
    flow_name = flow.name
    docker_registry = os.environ["DOCKER_REGISTRY"]
    docker_repository = os.getenv("DOCKER_REPOSITORY", PROJECT_NAME)
    image_name = os.path.join(docker_registry, docker_repository)
    image = f"{image_name}:{version}"
    if gpu:
        if aws_env == AwsEnv.production:
            aws_env_str = AwsEnv.production.name
        else:
            aws_env_str = str(aws_env)

        work_pool_name = f"coiled-{aws_env_str}"
        default_job_variables_name = (
            f"coiled-default-job-variables-prefect-mvp-{aws_env}"
        )
        default_job_variables: dict[str, Any] = Variable.get(default_job_variables_name)  # pyright: ignore[reportAssignmentType]
        if default_job_variables is None:
            raise ValueError(
                f"Variable '{default_job_variables_name}' not found in Prefect"
            )
        default_job_variables["image"] = image
        # Using a single host with a GPU, see:
        # https://docs.coiled.io/user_guide/prefect.html#configure-hardware
        default_job_variables.update(
            {
                "container": image,
                "gpu": True,
                "threads_per_worker": -1,
            }
        )

    else:
        work_pool_name = f"mvp-{aws_env}-ecs"
        default_job_variables_name = f"ecs-default-job-variables-prefect-mvp-{aws_env}"
        default_job_variables: dict[str, Any] = Variable.get(default_job_variables_name)  # pyright: ignore[reportAssignmentType]
        if default_job_variables is None:
            raise ValueError(
                f"Variable '{default_job_variables_name}' not found in Prefect"
            )

    job_variables = {**default_job_variables, **flow_variables}
    tags = [f"repo:{docker_repository}", f"awsenv:{aws_env}"] + extra_tags
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, check=True
        )
        if commit_sha := result.stdout.decode().strip():
            tags.append(f"sha:{commit_sha}")
    except Exception as e:
        logger.error(f"failed to get commit SHA: {e}")

    try:
        branch = os.environ.get("GIT_BRANCH")
        if not branch:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                check=True,
            )
            branch = result.stdout.decode().strip()

        if branch:
            tags.append(f"branch:{branch}")
    except Exception as e:
        logger.error(f"failed to get branch: {e}")

    schedule = get_schedule_for_env(
        aws_env,
        env_schedules,
        env_parameters,
    )

    _ = flow.deploy(
        generate_deployment_name(flow_name, aws_env),
        work_pool_name=work_pool_name,
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
        concurrency_limit=concurrency_limit,
    )


def _version() -> str:
    return importlib.metadata.version(PROJECT_NAME)


if __name__ == "__main__":
    logger.info(f"using version: {_version()}")

    # Train
    create_deployment(
        flow=train_on_gpu,
        description="Train concept classifiers with GPU compute",
        gpu=True,
        flow_variables={},
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
        flow_variables={
            "cpu": MEGABYTES_PER_GIGABYTE * 8,
            "memory": MEGABYTES_PER_GIGABYTE * 32,
            "match_latest_revision_in_family": True,
            "ephemeralStorage": {"sizeInGiB": 50},
        },
    )

    create_deployment(
        flow=inference_batch_of_documents_gpu,
        description="Run concept classifier inference on a batch of documents with GPU compute",
        extra_tags=["type:sub"],
        gpu=True,
        flow_variables={
            "cpu": 8,
            "memory": "32 GiB",
            "match_latest_revision_in_family": True,
        },
    )

    # Aggregate inference results

    create_deployment(
        flow=aggregate_batch_of_documents,
        description="Aggregate inference results for a batch of documents",
        flow_variables={
            "cpu": MEGABYTES_PER_GIGABYTE * 16,
            "memory": MEGABYTES_PER_GIGABYTE * 64,
            "match_latest_revision_in_family": True,
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
                    ]
                }
            ),
        },
        concurrency_limit=ConcurrencyLimitConfig(
            limit=1,
            collision_strategy=ConcurrencyLimitStrategy.ENQUEUE,
        ),
    )

    # Wikibase

    create_deployment(
        flow=sync_concepts,
        description="Upload concepts from Wikibase to Vespa",
        concurrency_limit=ConcurrencyLimitConfig(
            limit=1,
            collision_strategy=ConcurrencyLimitStrategy.ENQUEUE,
        ),
        env_schedules={
            AwsEnv.production: "0 8 * * MON-THU",
            AwsEnv.staging: "0 9 * * MON-THU",
            AwsEnv.sandbox: "0 10 * * MON-THU",
        },
    )

    create_deployment(
        flow=wikibase_to_s3,
        description="Upload concepts from Wikibase to S3",
        # required for application topics in concepts-api
        env_schedules={
            AwsEnv.production: "0 9 * * TUE,THU",
            AwsEnv.staging: "0 15 2 * *",
            AwsEnv.sandbox: "0 15 1 * *",
            # Not needed in labs
        },
    )

    # Classifiers Profiles Lifecycle

    create_deployment(
        flow=sync_classifiers_profiles,
        description="Compare Wikibase classifiers profiles with classifiers specs",
        env_schedules={
            AwsEnv.staging: "0 10 * * MON-THU",  # staging run 1x per day
            AwsEnv.production: "0 10,17 * * MON-THU",
        },
        env_parameters={
            AwsEnv.staging: JsonDict(
                {
                    "upload_to_wandb": False,  # staging env should never update wandb
                    "upload_to_vespa": True,
                    "automerge_classifier_specs_pr": True,
                    "auto_train": False,
                    "enable_slack_notifications": False,
                }
            ),
            AwsEnv.production: JsonDict(
                {
                    "upload_to_wandb": True,
                    "upload_to_vespa": True,
                    "automerge_classifier_specs_pr": True,
                    "auto_train": True,
                    "enable_slack_notifications": True,
                }
            ),
        },
    )

    # Sync Neo4j with Wikibase

    create_deployment(
        flow=update_concepts,
        description="Refresh Neo4j with the latest concept graph",
        env_schedules={
            AwsEnv.labs: "0 3 * * MON-THU",
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

    # Vibe Check

    create_deployment(
        flow=vibe_check_inference,  # pyright: ignore[reportArgumentType]
        description="Run vibe check inference on a set of concepts and push the results to s3. Optionally provide a custom list of Wikibase IDs to process.",
        env_schedules={
            # Once a week, at midday on Sunday
            AwsEnv.labs: "0 12 * * SUN",
        },
    )

    create_deployment(
        flow=train_for_vibe_checker,
        description="Train classifiers for all concepts listed in vibe-checker/config.yml. Runs training in parallel and uploads results to Weights & Biases.",
        gpu=False,
        env_schedules={
            AwsEnv.labs: "0 8 * * MON-THU",  # Every working day at 8am
        },
    )
