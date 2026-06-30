"""
CLI wrapper for model demotion.

The reusable logic lives in `flows.demote.run_demotion`; this module only adds the Typer
command used by `just demote` and the `demote` console script.
"""

from typing import Annotated, Optional

import typer

from flows.demote import run_demotion
from knowledge_graph.cloud import AwsEnv, parse_aws_env
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.version import Version

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
            help="AWS environment to demote the model artifact in",
            parser=parse_aws_env,
        ),
    ],
    wandb_registry_version: Annotated[
        Optional[Version],
        typer.Option(
            help="Optional: specific registry version of the model to demote",
            parser=Version,
        ),
    ] = None,
):
    """Demote a model within an AWS environment."""
    try:
        run_demotion(
            wikibase_id=wikibase_id,
            aws_env=aws_env,
            wandb_registry_version=wandb_registry_version,
        )
    except ValueError as e:
        raise typer.BadParameter(str(e))


if __name__ == "__main__":
    app()
