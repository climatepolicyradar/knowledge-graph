from typing import Optional

import typer
from prefect.client.schemas.objects import FlowRun
from prefect.deployments import run_deployment  # type: ignore
from prefect.settings import PREFECT_UI_URL
from pydantic import ValidationError
from rich.console import Console
from typing_extensions import Annotated

from flows.classifier_specs.spec_interface import ClassifierSpec
from flows.inference import inference
from knowledge_graph.cloud import (
    AwsEnv,
    generate_deployment_name,
    parse_aws_env,
)

app = typer.Typer()
console = Console()


def convert_classifier_specs(
    requested_classifiers: list[str] | None,
) -> list[ClassifierSpec] | None:
    """
    Prepare the requested classifiers.

    Validates the classifier parameter and converts it to json ready
    to submit to prefect cloud
    """
    if not requested_classifiers:
        return []

    specs = []
    for i, classifier in enumerate(requested_classifiers, 1):
        try:
            specs.append(ClassifierSpec.model_validate_json(classifier))
        except ValidationError as e:
            raise typer.BadParameter(
                f"Incorrect classifier specification for item {i}: {classifier}: {e}"
            )
    return specs


async def _trigger_deployment(
    deployment_name: str,
    classifiers: list[ClassifierSpec] | None,
    documents: list[str] | None,
) -> FlowRun:
    try:
        flow_run = await run_deployment(  # type: ignore[misc]
            name=deployment_name,
            parameters={
                "classifier_specs": classifiers,
                "document_ids": documents,
            },
            # Don't wait for the flow to finish before continuing script
            timeout=0,
        )
        return flow_run
    except Exception as e:
        console.log(f"[red]Error running deployment: {e}[/red]")
        raise e


@app.command(
    help="""Run classifier inference on documents.

        This triggers the deployed inference flow to run against documents in a
        pipeline cache bucket and save the labelled passage results back to s3.
        """
)
def main(
    aws_env: Annotated[
        AwsEnv,
        typer.Option(
            help="The AWS environment to use to find and store documents",
            parser=parse_aws_env,
        ),
    ],
    classifiers: Annotated[
        list[str] | None,
        typer.Option(
            "--classifier",
            "-c",
            help=(
                "JSON string specifying a classifier. The JSON should contain "
                "classifier specification fields like wikibase_id, classifier_id, "
                "and wandb_registry_version. Add more of this option to run "
                "multiple classifiers. For example: "
                '--classifier \'{"wikibase_id": "Q787", "classifier_id": "393fk3km", "wandb_registry_version": "v2"}\' '
                '--classifier \'{"wikibase_id": "Q123", "classifier_id": "aaaabbbb", "wandb_registry_version": "v1"}\''
            ),
            callback=convert_classifier_specs,
        ),
    ] = [],
    documents: Annotated[
        Optional[list[str]],
        typer.Option(
            "--document",
            "-d",
            help=(
                "The ids of the documents to run on. Add more of this option "
                "to run on multiple. For example: "
                "-d CCLW.executive.10002.4495 -d CCLW.executive.10126.4646"
            ),
        ),
    ] = [],
):
    # Set to None if empty as Typer reads it as a list
    documents_or_default = documents or None  # type: ignore
    # Set to None if empty as Typer reads it as a list

    classifiers_or_default: list[ClassifierSpec] | None = classifiers or None  # type: ignore
    console.log(f"Selected to run on: {classifiers=} & {documents=}")

    deployment_name = (
        f"{inference.name}/{generate_deployment_name(inference.name, aws_env)}"  # pyright: ignore[reportFunctionMemberAccess]
    )
    console.log(f"Starting run for deployment: {deployment_name}")
    import asyncio

    flow_run_coroutine = _trigger_deployment(
        deployment_name, classifiers_or_default, documents_or_default
    )
    # Run the coroutine
    flow_run = asyncio.run(flow_run_coroutine)

    flow_url = f"{PREFECT_UI_URL.value()}/runs/flow-run/{flow_run.id}"  # type: ignore
    console.log(f"See progress at: [green]{flow_url}[/green]")


if __name__ == "__main__":
    app()
