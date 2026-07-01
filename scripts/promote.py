"""
CLI wrapper for model promotion.

The reusable logic lives in `flows.promote.run_promotion`; this module only adds the
Typer command used by `just promote` and the `promote` console script.
"""

from typing import Annotated

import typer

from flows.promote import run_promotion
from knowledge_graph.cloud import AwsEnv, parse_aws_env
from knowledge_graph.identifiers import ClassifierID, WikibaseID

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
    classifier_id: Annotated[
        ClassifierID,
        typer.Option(
            help="Classifier ID that aligns with the Python class name",
            parser=ClassifierID,
        ),
    ],
    aws_env: Annotated[
        AwsEnv,
        typer.Option(
            help="AWS environment to promote the model artifact within",
            parser=parse_aws_env,
        ),
    ],
    add_classifiers_profiles: Annotated[
        list[str] | None,
        typer.Option(help="Adds 1 classifiers profile."),
    ] = None,
    remove_classifiers_profiles: Annotated[
        list[str] | None,
        typer.Option(help="Removes 1 or more classifiers profiles."),
    ] = None,
):
    """Promote a model to the registry so it can be used downstream."""
    try:
        run_promotion(
            wikibase_id=wikibase_id,
            classifier_id=classifier_id,
            aws_env=aws_env,
            add_classifiers_profiles=add_classifiers_profiles,
            remove_classifiers_profiles=remove_classifiers_profiles,
        )
    except ValueError as e:
        raise typer.BadParameter(str(e))


if __name__ == "__main__":
    app()
