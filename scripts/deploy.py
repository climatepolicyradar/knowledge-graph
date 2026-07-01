"""
CLI wrapper for the classifier deployment pipeline.

The reusable train -> promote -> refresh logic lives in `flows.deploy`; this module
only adds the Typer commands used by `just deploy-existing` / `just deploy-new`, plus
the CLI-level argument validation and exit handling.
"""

import traceback
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from flows.deploy import (
    run_deploy_existing,
    run_deploy_new,
    validate_classifiers_profiles,
)
from knowledge_graph.cloud import AwsEnv, parse_aws_env
from knowledge_graph.identifiers import WikibaseID

# Load environment variables from .env file at Git root
load_dotenv(Path(__file__).parent.parent / ".env")

app = typer.Typer()


def _validate_classifiers_profiles(
    add_classifiers_profiles: list[str] | None,
    remove_classifiers_profiles: list[str] | None = None,
) -> None:
    """Validate profiles at the CLI layer, surfacing errors as `typer.BadParameter`."""
    try:
        validate_classifiers_profiles(
            add_classifiers_profiles, remove_classifiers_profiles
        )
    except ValueError as e:
        raise typer.BadParameter(str(e))


@app.command()
def existing(
    from_aws_env: Annotated[
        AwsEnv,
        typer.Option(
            help="AWS environment to deploy from",
            parser=parse_aws_env,
        ),
    ] = AwsEnv.staging,
    to_aws_env: Annotated[
        AwsEnv,
        typer.Option(
            help="AWS environment to deploy to",
            parser=parse_aws_env,
        ),
    ] = AwsEnv.production,
    train: Annotated[bool, typer.Option(help="Whether to train models")] = True,
    promote: Annotated[bool, typer.Option(help="Whether to promote models")] = True,
    add_classifiers_profiles: Annotated[
        list[str] | None,
        typer.Option(help="Adds 1 classifiers profile."),
    ] = None,
    remove_classifiers_profiles: Annotated[
        list[str] | None,
        typer.Option(help="Removes 1 or more classifiers profiles."),
    ] = None,
):
    """Deploy existing models from one environment to another."""
    _validate_classifiers_profiles(
        add_classifiers_profiles, remove_classifiers_profiles
    )

    run_deploy_existing(
        from_aws_env=from_aws_env,
        to_aws_env=to_aws_env,
        train=train,
        promote=promote,
        add_classifiers_profiles=add_classifiers_profiles,
        remove_classifiers_profiles=remove_classifiers_profiles,
    )


@app.command()
def new(
    aws_env: Annotated[
        AwsEnv,
        typer.Option(
            help="AWS environment to deploy to",
            parser=parse_aws_env,
        ),
    ] = AwsEnv.staging,
    wikibase_ids: Annotated[
        list[WikibaseID],
        typer.Option(
            "--wikibase-id",
            help="List of Wikibase IDs to deploy (can be used multiple times)",
            parser=lambda x: WikibaseID(x),
        ),
    ] = [],
    train: Annotated[bool, typer.Option(help="Whether to train models")] = True,
    promote: Annotated[bool, typer.Option(help="Whether to promote models")] = True,
    add_classifiers_profiles: Annotated[
        list[str] | None,
        typer.Option(help="Adds 1 classifiers profile."),
    ] = None,
):
    """Deploy new models by training and promoting them."""
    _validate_classifiers_profiles(add_classifiers_profiles)

    if failed_wikibase_ids := run_deploy_new(
        aws_env=aws_env,
        wikibase_ids=wikibase_ids,
        train=train,
        promote=promote,
        add_classifiers_profiles=add_classifiers_profiles,
    ):
        for wikibase_id, e in failed_wikibase_ids:
            print(f"Failed to deploy classifier for {wikibase_id}:")
            print("".join(traceback.format_exception(e)))
            print("-" * 100 + "\n")

        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
