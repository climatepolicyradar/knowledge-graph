"""The whole pipeline to deploy classifier models."""

from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

import scripts.get_concept
import scripts.promote
import scripts.train
from scripts.cloud import AwsEnv, parse_aws_env, validate_transition
from scripts.update_classifier_spec import (
    get_all_available_classifiers,
    parse_spec_file,
)
from src.identifiers import WikibaseID

# Load environment variables from .env file at Git root
load_dotenv(Path(__file__).parent.parent / ".env")

app = typer.Typer()


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
    get: Annotated[bool, typer.Option(help="Whether to get concepts")] = True,
    train: Annotated[bool, typer.Option(help="Whether to train models")] = True,
    promote: Annotated[bool, typer.Option(help="Whether to promote models")] = True,
):
    """Deploy existing models from one environment to another."""
    validate_transition(from_aws_env, to_aws_env)

    specs = parse_spec_file(from_aws_env)
    print(f"loaded {len(specs)} classifier specifications")

    for spec in specs:
        print(f"\nprocessing {spec.name}:{spec.alias}")

        if get:
            print("getting concept")
            scripts.get_concept.main(wikibase_id=WikibaseID(spec.name))

        if train:
            print("training")
            classifier = scripts.train.main(
                wikibase_id=WikibaseID(spec.name),
                track=True,
                upload=True,
                aws_env=to_aws_env,
            )

            if classifier.version is None:
                print(f"classifier {classifier.name} is missing a version, so skipping")
                continue

            if promote:
                print("promoting")
                scripts.promote.main(
                    wikibase_id=WikibaseID(spec.name),
                    classifier=classifier.name,
                    version=classifier.version,
                    within_aws_env=to_aws_env,
                    primary=True,
                )

    get_all_available_classifiers([to_aws_env])


@app.command()
def new(
    to_aws_env: Annotated[
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
    get: Annotated[bool, typer.Option(help="Whether to get concepts")] = True,
    train: Annotated[bool, typer.Option(help="Whether to train models")] = True,
    promote: Annotated[bool, typer.Option(help="Whether to promote models")] = True,
):
    """Deploy new models by training and promoting them."""
    for wikibase_id in wikibase_ids:
        print(f"\nprocessing {wikibase_id}")

        if get:
            print("getting concept")
            scripts.get_concept.main(wikibase_id=wikibase_id)

        if train:
            print("training")
            classifier = scripts.train.main(
                wikibase_id=wikibase_id,
                track=True,
                upload=True,
                aws_env=to_aws_env,
            )

            if classifier.version is None:
                print(f"classifier {classifier.name} is missing a version, so skipping")
                continue

            if promote:
                print("promoting")
                scripts.promote.main(
                    wikibase_id=wikibase_id,
                    classifier=classifier.name,
                    version=classifier.version,
                    within_aws_env=to_aws_env,
                    primary=True,
                )

    get_all_available_classifiers([to_aws_env])


if __name__ == "__main__":
    app()
