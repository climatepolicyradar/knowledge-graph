from typing import Annotated

import typer
from rich.console import Console

from scripts.cloud import AwsEnv
from scripts.promote import main as promote
from scripts.train import main as train
from src.identifiers import WikibaseID

console = Console()
app = typer.Typer()


@app.command()
def main(
    wikibase_id: Annotated[
        WikibaseID,
        typer.Option(
            ...,
            help="The Wikibase ID of the concept classifier to train",
            parser=WikibaseID,
        ),
    ],
    aws_env: Annotated[
        AwsEnv,
        typer.Option(
            ...,
            help="AWS environment to use for S3 uploads",
        ),
    ],
) -> None:
    """
    Train and promote a modeldirectly to primary for the given AWS environment.

    :param wikibase_id: The Wikibase ID of the concept classifier to train.
    :type wikibase_id: WikibaseID
    :param aws_env: The AWS environment to use for S3 uploads.
    :type aws_env: AwsEnv
    """
    upload = track = primary = True  # because we wantmake this the primary version

    classifier = train(wikibase_id, track, upload, aws_env)
    assert classifier.version is not None, "Classifier version None, cannot promote"
    promote(
        wikibase_id=wikibase_id,
        classifier=classifier.name,
        version=classifier.version,
        within_aws_env=aws_env,
        primary=primary,
    )


if __name__ == "__main__":
    app()
