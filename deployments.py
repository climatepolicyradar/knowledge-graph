"""
Prefect deployment script

Used to create server side representation of prefect flows, triggers, config, etc.
See: https://docs-2.prefect.io/latest/concepts/deployments/
"""

import asyncio
import importlib.metadata
import logging
import os
from typing import Any

import httpx
import prefect
from prefect.blocks.system import JSON
from prefect.client.schemas.actions import ConcurrencyLimitCreate
from prefect.client.schemas.objects import (
    ConcurrencyLimit,
)
from prefect.deployments.runner import DeploymentImage
from prefect.flows import Flow

from flows.inference import (
    CLASSIFIER_INFERENCE_START_CONCURRENCY_LIMIT_NAME,
    classifier_inference,
)

logger = logging.getLogger(__name__)


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


async def read_concurrency_limit_by_name(name: str) -> ConcurrencyLimit:
    """
    Read a concurrency limit by name.

    [1] https://github.com/PrefectHQ/prefect/blob/main/src/prefect/client/orchestration.py#L824  # noqa: E501
    """
    client = prefect.get_client()

    try:
        response = await client._client.get(
            f"/concurrency_limits/{name}",
        )
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise prefect.exceptions.ObjectNotFound(http_exc=e) from e
        else:
            raise

    concurrency_limit_id = response.json().get("id")

    if not concurrency_limit_id:
        raise httpx.RequestError(f"Malformed response: {response}")

    return ConcurrencyLimit.model_validate(response.json())


async def delete_concurrency_limit_by_name(name: str) -> None:
    """
    Delete a concurrency limit by name.

    [1] https://github.com/PrefectHQ/prefect/blob/main/src/prefect/client/orchestration.py#L919C1-L942C22  # noqa: E501
    """
    client = prefect.get_client()

    try:
        await client._client.delete(
            f"/concurrency_limits/{name}",
        )
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise prefect.exceptions.ObjectNotFound(http_exc=e) from e
        else:
            raise


async def create_concurrency_limit(name, limit=1) -> ConcurrencyLimit:
    client = prefect.get_client()

    concurrency_limit_create = ConcurrencyLimitCreate(
        name=name,
        limit=limit,
    )

    response = await client._client.post(
        "/concurrency_limits/",
        json=concurrency_limit_create.dict(json_compatible=True),
    )

    return ConcurrencyLimit.model_validate(response.json())


async def create_replace_global_concurrency_limit(name, limit=1) -> ConcurrencyLimit:
    """
    Create a global concurrency limit by name.

    This checks if it already exists first, and if so, deletes it.

    Prefect don't have a "nice" wrapper around this, as they do other things
    (like `Deployment`s, above).

    Instead, we'll use the same approach they take [1][2].

    [1] https://docs-2.prefect.io/latest/api-ref/prefect/client/orchestration/?h=get_client#prefect.client.orchestration.PrefectClient.create_concurrency_limit  # noqa: E501
    [2] https://github.com/PrefectHQ/prefect/blob/main/src/prefect/client/orchestration.py#L788  # noqa: E501
    """
    try:
        await read_concurrency_limit_by_name(name)
        await delete_concurrency_limit_by_name(name)
    except prefect.exceptions.ObjectNotFound:
        # If it doesn't exist yet, that's fine
        pass
    except Exception as e:
        # Re-raise any other exceptions
        raise e

    return await create_concurrency_limit(name, limit)


async def main():
    # Inference
    concurrency_limit = await create_replace_global_concurrency_limit(
        CLASSIFIER_INFERENCE_START_CONCURRENCY_LIMIT_NAME,
    )

    logger.info(
        f"Created concurrency limit (ID: {concurrency_limit.id}, Name: {CLASSIFIER_INFERENCE_START_CONCURRENCY_LIMIT_NAME})"
    )

    # create_deployment(
    #     project_name="knowledge-graph",
    #     flow=classifier_inference,
    #     description="Run concept classifier inference on document passages",
    #     flow_variables={
    #         "cpu": MEGABYTES_PER_GIGABYTE * 4,
    #         "memory": MEGABYTES_PER_GIGABYTE * 16,
    #         "ephemeralStorage": {"sizeInGiB": 50},
    #     },
    # )


if __name__ == "__main__":
    asyncio.run(main())
