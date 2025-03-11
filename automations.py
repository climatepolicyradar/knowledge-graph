"""Create automations in Prefect Cloud."""

import asyncio
import logging
import os
from typing import List
from uuid import UUID

import prefect.events.schemas.automations as automations
from prefect.client import get_client
from prefect.client.schemas.objects import Deployment
from prefect.events.actions import RunDeployment
from prefect.exceptions import ObjectNotFound

from flows.count_family_document_concepts import count_family_document_concepts
from flows.index import index_labelled_passages_from_s3_to_vespa
from flows.inference import classifier_inference
from scripts.cloud import PROJECT_NAME, AwsEnv, generate_deployment_name

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler and set level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Add ch to logger
logger.addHandler(ch)


async def read_automations_by_name(client, name: str) -> List[dict]:
    """Read a list of automations from Prefect Cloud."""
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
    """Update an automation in Prefect Cloud."""
    # The first step in preparing this to be serialised to JSON
    json = automation.dict()

    # Delete as it's null
    del json["owner_resource"]
    # Delete these as they're all sets which aren't serialisable, and
    # also they're unused
    del json["trigger"]["after"]
    del json["trigger"]["for_each"]

    # Transform these types to be serialisable
    json["trigger"]["within"] = int(json["trigger"]["within"].total_seconds())
    json["trigger"]["expect"] = list(json["trigger"]["expect"])
    # This lens matches the structure returned in the target automation.
    json["actions"][0]["deployment_id"] = str(json["actions"][0]["deployment_id"])

    response = await client._client.put(
        f"/automations/{automation_id}",
        json=json,
    )
    response.raise_for_status


def create_target_automation(
    a_flow_name: str,
    a_deployment: Deployment,
    b_deployment: Deployment,
    description: str,
    parameters: dict,
    enabled: bool,
) -> automations.Automation:
    """
    Create a copy of the `Automation` that triggers another `Deployment`.

    This is the "target" in the sense that it's used for creating the
    `Automation` if it doesn't exist yet or the `Automation` has changed
    and we want to update the existing `Automation`s in Prefect Cloud to
    reflect the updated approach.

    The `Automation` will trigger `Deployment` B to run after
    `Deployment` A has completed successfully. The ~Automation`
    watches for completion events from `Flow` A `Run`s and
    specifically from `Deployment` A, then launches `Deployment` B
    with the provided parameters.

    Args:
        a_flow_name: The name of the `Flow` that needs to complete first
        a_deployment: The `Deployment` of `Flow` A that needs to complete
        b_deployment: The `Deployment` to trigger after A completes
        parameters: Parameters to pass to `Deployment` B when it's triggered
        enabled: Whether the automation should run or not
        description: A helpful description of what the `Automation` does

    Returns:
        A Prefect Automation object configured to trigger B after A completes
    """
    return automations.Automation(
        name=f"trigger-{b_deployment.name}",
        description=description,
        enabled=enabled,
        # This is based on crafting the trigger in the web UI.
        trigger=automations.EventTrigger(
            match={
                # The `*` is to match on any ID
                "prefect.resource.id": "prefect.flow-run.*",
            },
            match_related=automations.ResourceSpecification(
                __root__={
                    # This is the Deployment, for the specific AWS
                    # environment, that started the `flow-run`.
                    "prefect.resource.id": f"prefect.deployment.{a_deployment.id}",
                    "prefect.resource.role": "deployment",
                    "prefect.resource.name": a_deployment.name,
                }
            ),
            # > Reactive automations respond to the presence of the
            # > expected events ...
            posture="Reactive",
            # > The number of events required for this Automation to
            # > trigger (for Reactive automations), or the number of
            # > events expected (for Proactive automations)
            threshold=1,
            # > The time period over which the events must occur. For
            # > Reactive triggers, this may be as low as 0 seconds,
            # > but must be at least 10 seconds for Proactive triggers.
            within=0,
            # > The event(s) this automation is expecting to see. If
            # > empty, this automation will evaluate any matched
            # > event.
            #
            # NB: Currently a set due to the Prefect Python class used
            expect=set(["prefect.flow-run.Completed"]),
        ),
        actions=[
            RunDeployment(
                # Requires passing the Deployment ID [1]
                #
                # [1] https://github.com/PrefectHQ/prefect/blob/ab964c1c4b52fd9ae61bc8d816505ac89df7a8f8/src/prefect/events/actions.py#L32
                source="selected",
                deployment_id=b_deployment.id,
                parameters=parameters,
            ),
        ],
    )


def flow_deployment_names_id(flow_name, deployment_name):
    return f"{flow_name}/{deployment_name}"


async def a_triggers_b(
    a_flow_name: str,
    a_deployment_name: str,
    b_flow_name: str,
    b_deployment_name: str,
    b_parameters: dict,
    enabled: bool,
    description: str,
    ignore: list[AwsEnv],
    aws_env: AwsEnv,
) -> None:
    """Automation to after Deployment A completes trigger Deployment B."""
    client = get_client()

    logger.info(
        "Loading Deployment: %s",
        a_deployment_name,
    )

    try:
        a_deployment = await client.read_deployment_by_name(
            name=flow_deployment_names_id(a_flow_name, a_deployment_name)
        )
    except ObjectNotFound:
        if aws_env in ignore:
            logger.info(
                f"Deployment not found in {aws_env} environment, skipping automation creation"
            )
            return

        raise

    logger.info(
        "Loaded Deployment: name=%s, flow_id=%s",
        a_deployment.name,
        a_deployment.flow_id,
    )

    logger.info("Loading Deployment: %s", b_deployment_name)

    b_deployment = await client.read_deployment_by_name(
        name=flow_deployment_names_id(b_flow_name, b_deployment_name)
    )

    logger.info(
        "Loaded Deployment: name=%s, flow_id=%s",
        b_deployment.name,
        b_deployment.flow_id,
    )

    target = create_target_automation(
        a_flow_name=a_flow_name,
        a_deployment=a_deployment,
        b_deployment=b_deployment,
        description=description,
        parameters=b_parameters,
        enabled=enabled,
    )

    automations = await read_automations_by_name(
        client=client,
        name=target.name,
    )

    match len(automations):
        case 0:
            logger.info("Automation doesn't exist already, creating it")

            automation_id = await client.create_automation(automation=target)

            logger.info("Created automation with id='%s'", automation_id)
        case 1:
            automation = automations[0]

            logger.info("Automation exists already, updating it")
            logger.info(
                "Read automation with id='%s', name='%s'",
                automation["id"],
                automation["name"],
            )

            await update_automation(
                client=client,
                automation_id=UUID(automation["id"]),
                automation=target,
            )

            logger.info("Updated automation")
        case _:
            names = [auto["name"] for auto in automations]

            raise ValueError(
                f"Found multiple automations with name {target.name}: {names}"
            )


async def main() -> None:
    """Create or update the automation for triggering inference."""
    aws_env = AwsEnv(os.getenv("AWS_ENV"))

    await a_triggers_b(
        a_flow_name="navigator-data-s3-backup",
        a_deployment_name=f"navigator-data-s3-backup-pipeline-cache-{aws_env}",
        b_flow_name=classifier_inference.name,
        b_deployment_name=f"{PROJECT_NAME}-{classifier_inference.name}-{aws_env}",
        b_parameters={"use_new_and_updated": True},
        description="Start concept store inference with classifiers.",
        enabled=True,
        aws_env=aws_env,
        ignore=[AwsEnv.labs],
    )

    await a_triggers_b(
        a_flow_name=classifier_inference.name,
        a_deployment_name=generate_deployment_name(classifier_inference.name, aws_env),
        b_flow_name=index_labelled_passages_from_s3_to_vespa.name,
        b_deployment_name=generate_deployment_name(
            index_labelled_passages_from_s3_to_vespa.name, aws_env
        ),
        b_parameters={},
        enabled=False,
        description="Start concept store indexing with classifiers.",
        aws_env=aws_env,
        ignore=[],
    )

    await a_triggers_b(
        a_flow_name=index_labelled_passages_from_s3_to_vespa.name,
        a_deployment_name=generate_deployment_name(
            index_labelled_passages_from_s3_to_vespa.name, aws_env
        ),
        b_flow_name=count_family_document_concepts.name,
        b_deployment_name=generate_deployment_name(
            count_family_document_concepts.name, aws_env
        ),
        b_parameters={},
        enabled=True,
        description="Start concepts counting and indexing (loading) into Vespa",
        aws_env=aws_env,
        ignore=[],
    )


if __name__ == "__main__":
    asyncio.run(main())
