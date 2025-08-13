"""Create automations in Prefect Cloud."""

import asyncio
import logging
import os
import re
from datetime import timedelta

import prefect.events.schemas.automations as automations
import prefect.events.schemas.events as events
from prefect.automations import Automation
from prefect.client.orchestration import get_client
from prefect.client.schemas.responses import DeploymentResponse
from prefect.events.actions import RunDeployment
from prefect.exceptions import ObjectNotFound

from flows.full_pipeline import full_pipeline
from scripts.cloud import AwsEnv, generate_deployment_name

# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler and set level
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)

# Add ch to logger
logger.addHandler(ch)


def create_target_automation(
    a_flow_name: str,
    a_deployment: DeploymentResponse,
    b_deployment: DeploymentResponse,
    description: str,
    parameters: dict,
    enabled: bool,
) -> Automation:
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
    """
    return Automation(
        name=f"trigger-{b_deployment.name}",
        description=description,
        enabled=enabled,
        # This is based on crafting the trigger in the web UI.
        trigger=automations.EventTrigger(
            match=events.ResourceSpecification(
                # The `*` is to match on any ID
                root={"prefect.resource.id": "prefect.flow-run.*"},
            ),
            match_related={
                # This is the Deployment, for the specific AWS
                # environment, that started the `flow-run`.
                "prefect.resource.id": f"prefect.deployment.{a_deployment.id}",
                "prefect.resource.role": "deployment",
                "prefect.resource.name": a_deployment.name,
            },
            # > Reactive automations respond to the presence of the
            # > expected events ...
            posture=automations.Posture.Reactive,
            # > The number of events required for this Automation to
            # > trigger (for Reactive automations), or the number of
            # > events expected (for Proactive automations)
            threshold=1,
            # > The time period over which the events must occur. For
            # > Reactive triggers, this may be as low as 0 seconds,
            # > but must be at least 10 seconds for Proactive triggers.
            within=timedelta(seconds=0),
            # > The event(s) this automation is expecting to see. If
            # > empty, this automation will evaluate any matched
            # > event.
            #
            # NB: Currently a set due to the Prefect Python class used
            expect=set(["prefect.flow-run.Running"]),
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


async def delete_if_exists(name: str) -> None:
    """
    Delete the Automation, if it exists.

    Since there isn't an overwrite argument, like with Blocks, we
    handle this ourselves.
    """
    try:
        automation = await Automation.read(name=name)

        print("Automation exists, deleting it")

        assert await automation.delete()

        print("Automation deleted")

        return None
    except ValueError as e:
        error_message = str(e)
        # From https://prefect-python-sdk-docs.netlify.app/prefect/automations/#prefect.automations.Automation.read
        pattern = r"Automation with.*not found"

        if re.match(pattern, error_message):
            print("automation doesn't exist already, so nothing to do")
            return None
        else:
            # It was a real problem, re-raise the excpetion
            raise


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

    print("deleting, if already exists...")
    await delete_if_exists(name=target.name)

    print("creating...")
    automation = await target.acreate()
    print(f"created Automation with id=`{automation.id}`, name=`{automation.name}`")


async def main() -> None:
    """Create or update the automation for triggering inference."""
    aws_env = AwsEnv(os.getenv("AWS_ENV"))

    await a_triggers_b(
        a_flow_name="navigator-data-s3-backup",
        a_deployment_name=f"navigator-data-s3-backup-pipeline-cache-{aws_env}",
        b_flow_name=full_pipeline.name,
        b_deployment_name=generate_deployment_name(full_pipeline.name, aws_env),
        b_parameters={"inference_use_new_and_updated": True},
        description="Start the knowledge graph full pipeline.",
        enabled=False,
        aws_env=aws_env,
        ignore=[AwsEnv.labs],
    )


if __name__ == "__main__":
    asyncio.run(main())
