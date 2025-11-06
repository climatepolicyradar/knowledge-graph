"""The whole pipeline to deploy classifier models."""

import traceback
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

import scripts.get_concept
import scripts.promote
import scripts.train
from knowledge_graph.cloud import (
    AwsEnv,
    parse_aws_env,
    parse_spec_file,
    validate_transition,
)
from knowledge_graph.identifiers import WikibaseID
from scripts.update_classifier_spec import (
    refresh_all_available_classifiers,
)

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
    validate_transition(from_aws_env, to_aws_env)

    add_class_prof: set[str] = (
        set(add_classifiers_profiles) if add_classifiers_profiles else set()
    )
    remove_class_prof: set[str] = (
        set(remove_classifiers_profiles) if remove_classifiers_profiles else set()
    )
    if dupes := add_class_prof & remove_class_prof:
        raise typer.BadParameter(
            f"duplicate values found for adding and removing classifiers profiles: `{','.join(dupes)}`"
        )

    if len(add_class_prof) > 1:
        raise typer.BadParameter(
            f"Artifact must have maximum of one classifiers profile in metadata, or you must specify 1 to remove. Provided: `{','.join(add_class_prof)}`"
        )

    specs = parse_spec_file(from_aws_env)
    print(f"loaded {len(specs)} classifier specifications")

    for spec in specs:
        print(f"\nprocessing {spec.name}:{spec.alias}")

        if train:
            print("training")
            classifier = scripts.train.main(
                wikibase_id=WikibaseID(spec.name),
                track_and_upload=True,
                aws_env=to_aws_env,
            )
            if not classifier:
                raise ValueError("No classifier returned from training.")

            if promote:
                print("promoting")
                scripts.promote.main(
                    wikibase_id=WikibaseID(spec.name),
                    classifier_id=classifier.id,
                    aws_env=to_aws_env,
                    add_classifiers_profiles=add_classifiers_profiles,
                    remove_classifiers_profiles=remove_classifiers_profiles,
                )

    refresh_all_available_classifiers([to_aws_env])


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
    failed_wikibase_ids = []
    for wikibase_id in wikibase_ids:
        try:
            print(f"\nprocessing {wikibase_id}")

            add_class_prof: set[str] = (
                set(add_classifiers_profiles) if add_classifiers_profiles else set()
            )

            if len(add_class_prof) > 1:
                raise typer.BadParameter(
                    f"Artifact must have maximum of one classifiers profile in metadata. Provided: `{','.join(add_class_prof)}`"
                )

            if train:
                print("training")
                classifier = scripts.train.main(
                    wikibase_id=wikibase_id,
                    track_and_upload=True,
                    aws_env=aws_env,
                )

                if not classifier:
                    raise ValueError("No classifier returned from training.")

                if promote:
                    print("promoting")
                    scripts.promote.main(
                        wikibase_id=wikibase_id,
                        classifier_id=classifier.id,
                        aws_env=aws_env,
                        add_classifiers_profiles=add_classifiers_profiles,
                    )
        except AttributeError as e:
            print(f"Error getting concept: {e}")
            failed_wikibase_ids.append((wikibase_id, e))
            continue

    refresh_all_available_classifiers([aws_env])

    if failed_wikibase_ids:
        for wikibase_id, e in failed_wikibase_ids:
            print(f"Failed to deploy classifier for {wikibase_id}:")
            print("".join(traceback.format_exception(e)))
            print("-" * 100 + "\n")

        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
