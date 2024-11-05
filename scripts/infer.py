from enum import Enum
from typing import Optional

import typer
from prefect.deployments import run_deployment
from prefect.settings import PREFECT_UI_URL
from rich.console import Console
from typing_extensions import Annotated

from flows.inference import (
    ClassifierSpec,
)

app = typer.Typer()
console = Console()


def convert_classifier_specs(requested_classifiers: list[str]) -> list[ClassifierSpec]:
    """
    Prepares the requested classifiers

    Validates the classifier parameter and converts it to json ready to submit to
    prefect cloud
    """
    classifier_specs = []
    for i, classifier in enumerate(requested_classifiers):
        match classifier.count(":"):
            case 0:
                spec = ClassifierSpec(name=classifier)
            case 1:
                name, alias = classifier.split(":")
                spec = ClassifierSpec(name=name, alias=alias)
            case _:
                raise typer.BadParameter(
                    f"Incorrect classifier specification for item {i}: {classifier}"
                )
        classifier_specs.append(spec.model_dump())
    return classifier_specs


class Env(str, Enum):
    """Options of which environment prefect job to use"""

    sandbox = "sandbox"
    labs = "labs"
    staging = "staging"
    prod = "prod"


@app.command(
    help="""Run classifier inference on documents.
        
        This triggers the deployed inference flow to run against documents in a
        pipeline cache bucket and save the labelled passage results back to s3.
        """
)
def main(
    environment: Annotated[
        Env,
        typer.Option(
            help="AWS env to use to find and store documents",
        ),
    ],
    classifiers: Annotated[
        Optional[list[str]],
        typer.Option(
            "--classifier",
            "-c",
            help=(
                "Select which classifiers and their aliases to run with "
                "Specify they alias by appending it after a ':', "
                "alias will default to 'latest' if left unspecified"
                "Add more of this option to run on multiple. For example: "
                "-c Q787:v0 -c Q787:v1 -c Q111"
            ),
            callback=convert_classifier_specs,
        ),
    ] = None,
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
    ] = None,
):
    documents = documents or None  # Set to None if empty as Typer reads it as a list
    console.log(f"Selected to run on: {classifiers=} & {documents=}")

    deployment_name = (
        f"classifier-inference/knowledge-graph-classifier-inference-{environment}"
    )
    console.log(f"Starting run for deployment: {deployment_name}")

    flow_run = run_deployment(
        name=deployment_name,
        timeout=0,  # Don't wait for the flow to finish before continuing script
        parameters={
            "classifier_specs": classifiers,
            "document_ids": documents,
        },
    )

    flow_url = f"{PREFECT_UI_URL.value()}/runs/flow-run/{flow_run.id}"
    console.log(f"See progress at: {flow_url}")


if __name__ == "__main__":
    app()
