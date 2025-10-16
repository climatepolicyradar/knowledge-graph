from typing import Annotated

import typer
import wandb
from dotenv import load_dotenv
from rich.console import Console

from knowledge_graph.cloud import AwsEnv, parse_aws_env
from knowledge_graph.identifiers import ClassifierID, WikibaseID

load_dotenv()

console = Console()

app = typer.Typer()


@app.command()
def main(
    wikibase_id: Annotated[
        WikibaseID,
        typer.Option(
            help="Wikibase ID of the concept",
            parser=WikibaseID,
        ),
    ],
    aws_env: Annotated[
        AwsEnv,
        typer.Option(
            help="AWS environment to promote the model artifact within",
            parser=parse_aws_env,
        ),
    ],
    classifier_id: Annotated[
        ClassifierID,
        typer.Option(
            help="Classifier ID that aligns with the Python class name",
            parser=ClassifierID,
        ),
    ],
    wandb_registry_version: Annotated[
        str,
        typer.Option(help="The version of the classifier to retrieve from wandb"),
    ],
) -> None:
    """Refreshes the classifier specs with the latest state of wandb."""
    console.log(f"Running for AWS environments: {aws_env.name}")

    artifact_id = f"{wikibase_id}:{wandb_registry_version}"
    target_path = f"wandb-registry-model/{artifact_id}"

    console.log(f"Searching for artifact_id {artifact_id}")
    run = wandb.init()

    artifact = run.use_model(f"{target_path}")
    if not artifact:
        console.log(f"No artifact found for {artifact_id}")
        return

    console.log("Finished!")


if __name__ == "__main__":
    app()
