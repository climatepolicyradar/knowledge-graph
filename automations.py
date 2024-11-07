import asyncio
import os
from typing import List
from uuid import UUID

import prefect.events.schemas.automations as automations
from prefect.client import get_client
from prefect.client.schemas.objects import Deployment
from prefect.events.actions import RunDeployment

from flows.inference import classifier_inference


async def read_automations_by_name(client, name: str) -> List[dict]:
    response = await client._client.post(
        "/automations/filter",
        json={
            "automations": {
                "operator": "and_",
                "name": {"any_": [name]},
            },
        },
    )

    response.raise_for_status()

    return response.json()


async def update_automation(client, automation_id: UUID, automation):
    """Updates an automation in Prefect Cloud."""

    # The first step in preparing this to be serialised to JSON
    json = automation.dict()

    # Delete as it's null
    del json["owner_resource"]
    # Delete these as they're all sets which aren't serialisable, and
    # also they're unused
    del json["trigger"]["expect"]
    del json["trigger"]["after"]
    del json["trigger"]["for_each"]

    # Transform these types to be serialisable
    json["trigger"]["within"] = int(json["trigger"]["within"].total_seconds())
    # This lens matches the structure returned by `prime()`.
    json["actions"][0]["deployment_id"] = str(json["actions"][0]["deployment_id"])

    response = await client._client.put(
        f"/automations/{automation_id}",
        json=json,
    )
    response.raise_for_status


def prime(
    navigator_data_s3_backup_deployment: Deployment,
    classifier_inference_deployment: Deployment,
    aws_env: str,
) -> automations.Automation:
    """Return a new copy of the target Automation."""
    return automations.Automation(
        name=f"{classifier_inference_deployment.name}-trigger",
        description="Start concept store inference with classifiers.",
        trigger=automations.EventTrigger(
            match={
                "prefect.resource.id": "prefect.flow-run.*",
                "prefect.resource.name": ["navigator-data-s3-backup"],
            },
            match_related=automations.ResourceSpecification(
                __root__={
                    # This is the Deployment, for the specific AWS environment.
                    "prefect.resource.id": str(navigator_data_s3_backup_deployment.id),
                }
            ),
            posture="Proactive",
            threshold=1,
            within=0,
        ),
        expect=["prefect.flow-run.Completed"],
        actions=[
            RunDeployment(
                # Requires passing the Deployment ID [1]
                #
                # [1] https://github.com/PrefectHQ/prefect/blob/ab964c1c4b52fd9ae61bc8d816505ac89df7a8f8/src/prefect/events/actions.py#L32
                source="selected",
                deployment_id=classifier_inference_deployment.id,
                parameters={"use_new_and_updated": True},
            ),
        ],
    )


async def main() -> None:
    """Create or update the automation for triggering inference."""
    aws_env = os.getenv("AWS_ENV")
    if aws_env is None or aws_env == "":
        raise ValueError("AWS_ENV is missing")

    project_name = "knowledge-graph"

    client = get_client()

    navigator_data_s3_backup_flow_name = "navigator-data-s3-backup"
    navigator_data_s3_backup_deployment_name = f"{navigator_data_s3_backup_flow_name}/navigator-data-s3-backup-pipeline-cache-{aws_env}"

    print(f"loading Deployment: {navigator_data_s3_backup_deployment_name}")

    navigator_data_s3_backup_deployment = await client.read_deployment_by_name(
        name=navigator_data_s3_backup_deployment_name
    )

    print(
        f"loaded Deployment: name={navigator_data_s3_backup_deployment.name}, flow_id={navigator_data_s3_backup_deployment.flow_id}"
    )

    classifier_inference_flow_name = classifier_inference.name
    classifier_inference_deployment_name = f"{classifier_inference_flow_name}/{project_name}-{classifier_inference.name}-{aws_env}"

    print(f"loading Deployment: {classifier_inference_deployment_name}")

    classifier_inference_deployment = await client.read_deployment_by_name(
        name=classifier_inference_deployment_name
    )

    print(
        f"loaded Deployment: name={classifier_inference_deployment.name}, flow_id={classifier_inference_deployment.flow_id}"
    )

    original = prime(
        navigator_data_s3_backup_deployment,
        classifier_inference_deployment,
        aws_env,
    )

    automations = await read_automations_by_name(
        client=client,
        name=original.name,
    )

    match len(automations):
        case 0:
            print("Automation doesn't exist already, creating it")

            automation_id = await client.create_automation(automation=original)

            print(f"Created automation with id=`{automation_id}`")
        case 1:
            automation = automations[0]

            print("Automation exists already, updating it")
            print(
                (
                    "Read automation with "
                    f'"id=`{automation["id"]}`, '
                    f'name=`{automation["name"]}`'
                )
            )

            await update_automation(
                client=client,
                automation_id=UUID(automation["id"]),
                automation=original,
            )

            print("Updated automation")
        case _:
            names = [auto["name"] for auto in automations]

            raise ValueError(
                f"Found multiple automations with name {original.name}: {names}"
            )


if __name__ == "__main__":
    asyncio.run(main())
