import asyncio
import os
import re

from prefect.automations import Automation
from prefect.deployments import Deployment
from prefect.events.schemas.automations import EventTrigger
from prefect.server.events.actions import RunDeployment

from flows.inference import classifier_inference


def prime(
    navigator_data_s3_backup_deployment: Deployment,
    classifier_inference_deployment: Deployment,
    aws_env: str,
) -> Automation:
    """Return a new copy of the target Automation."""
    return Automation(
        name=f"{classifier_inference_deployment.name}-trigger",
        trigger=EventTrigger(
            match={
                "prefect.resource.id": "prefect.flow-run.*",
                "prefect.resource.name": ["navigator-data-s3-backup"],
            },
            match_related={
                # This is the Deployment, for the specific AWS environment.
                "prefect.resource.id": navigator_data_s3_backup_deployment.id,
            },
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

    project_name = ("knowledge-graph",)

    navigator_data_s3_backup_deployment = Deployment(
        name=f"navigator-data-s3-backup-pipeline-cache-{aws_env}",
    ).load()

    classifier_inference_deployment = Deployment(
        name=f"{project_name}-{classifier_inference.name}-{aws_env}"
    ).load()

    original = prime(
        navigator_data_s3_backup_deployment,
        classifier_inference_deployment,
        aws_env,
    )

    try:
        automation = await Automation.read(name=original.name)
        print("Automation exists already, updating it")
        print(
            (
                "Read automation with "
                f"id=`{automation.id}`, "
                f"name=`{automation.name}`"
            )
        )

        # Set the attributes that will be updated
        automation.trigger = original.trigger
        automation.expect = original.expect
        automation.actions = original.actions

        automation = await automation.update()
        print("Updated automation")
    except ValueError as e:
        error_message = str(e)
        # From https://docs-2.prefect.io/latest/api-ref/prefect/automations/#prefect.automations.Automation.read
        pattern = r"Automation with.*not found"

        if re.match(pattern, error_message):
            print("Automation doesn't exist already, creating it")

            automation = await original.create()
            print(
                (
                    "Created automation with "
                    f"id=`{automation.id}`, "
                    f"name=`{automation.name}`"
                )
            )
        else:
            # It was a real problem, re-raise the exception
            raise


if __name__ == "__main__":
    asyncio.run(main())
