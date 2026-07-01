"""
Orchestration for the classifier deployment pipeline.

`run_deploy_existing` re-deploys the models already listed in an environment's spec
file; `run_deploy_new` trains and promotes a fresh set of Wikibase IDs. Both train
locally, optionally promote, then refresh the destination env's classifier specs.

Argument validation and the train -> promote -> refresh logic live here so direct
callers are guarded too; `scripts.deploy` only adds the Typer CLI surface and exit
handling, surfacing validation errors as `typer.BadParameter`. This module reaches
into operations/flows directly rather than other `scripts/` modules.
"""

import asyncio

from flows.promote import run_promotion
from flows.update_classifier_spec import refresh_all_available_classifiers
from knowledge_graph.classifier import Classifier
from knowledge_graph.cloud import (
    AwsEnv,
    parse_spec_file,
    validate_transition,
)
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.operations.train import run_training
from knowledge_graph.utils import get_logger

logger = get_logger(__name__)


def validate_classifiers_profiles(
    add_classifiers_profiles: list[str] | None,
    remove_classifiers_profiles: list[str] | None = None,
) -> None:
    """
    Validate that at most one profile is added and none is both added and removed.

    Raises ``ValueError`` on invalid input; callers facing a CLI should convert this
    into the appropriate user-facing error.
    """
    add_class_prof = (
        set(add_classifiers_profiles) if add_classifiers_profiles else set()
    )
    remove_class_prof = (
        set(remove_classifiers_profiles) if remove_classifiers_profiles else set()
    )

    if dupes := add_class_prof & remove_class_prof:
        raise ValueError(
            f"duplicate values found for adding and removing classifiers profiles: `{','.join(dupes)}`"
        )

    if len(add_class_prof) > 1:
        raise ValueError(
            f"Artifact must have maximum of one classifiers profile in metadata, or you must specify 1 to remove. Provided: `{','.join(add_class_prof)}`"
        )


def _train_locally(wikibase_id: WikibaseID, aws_env: AwsEnv) -> Classifier:
    """Train a classifier in-process and upload its artifact, mirroring `just train`."""
    return asyncio.run(
        run_training(
            wikibase_id=wikibase_id,
            track_and_upload=True,
            aws_env=aws_env,
        )
    )


def run_deploy_existing(
    from_aws_env: AwsEnv,
    to_aws_env: AwsEnv,
    train: bool = True,
    promote: bool = True,
    add_classifiers_profiles: list[str] | None = None,
    remove_classifiers_profiles: list[str] | None = None,
) -> None:
    """Deploy existing models from one environment to another."""
    validate_transition(from_aws_env, to_aws_env)
    validate_classifiers_profiles(add_classifiers_profiles, remove_classifiers_profiles)

    specs = parse_spec_file(from_aws_env)
    logger.info(f"loaded {len(specs)} classifier specifications")

    for spec in specs:
        logger.info(f"processing {spec.name}:{spec.alias}")

        if train:
            logger.info("training")
            classifier = _train_locally(WikibaseID(spec.name), to_aws_env)
            if not classifier:
                raise ValueError("No classifier returned from training.")

            if promote:
                logger.info("promoting")
                run_promotion(
                    wikibase_id=WikibaseID(spec.name),
                    classifier_id=classifier.id,
                    aws_env=to_aws_env,
                    add_classifiers_profiles=add_classifiers_profiles,
                    remove_classifiers_profiles=remove_classifiers_profiles,
                )

    refresh_all_available_classifiers([to_aws_env])


def run_deploy_new(
    aws_env: AwsEnv,
    wikibase_ids: list[WikibaseID],
    train: bool = True,
    promote: bool = True,
    add_classifiers_profiles: list[str] | None = None,
) -> list[tuple[WikibaseID, Exception]]:
    """
    Deploy new models by training and promoting them.

    Returns a list of ``(wikibase_id, error)`` for any classifiers that failed to
    deploy; the caller is responsible for reporting them and signalling failure.
    """
    validate_classifiers_profiles(add_classifiers_profiles)

    failed_wikibase_ids: list[tuple[WikibaseID, Exception]] = []
    for wikibase_id in wikibase_ids:
        try:
            logger.info(f"processing {wikibase_id}")

            if train:
                logger.info("training")
                classifier = _train_locally(wikibase_id, aws_env)
                if not classifier:
                    raise ValueError("No classifier returned from training.")

                if promote:
                    logger.info("promoting")
                    run_promotion(
                        wikibase_id=wikibase_id,
                        classifier_id=classifier.id,
                        aws_env=aws_env,
                        add_classifiers_profiles=add_classifiers_profiles,
                    )
        except AttributeError as e:
            logger.error(f"Error getting concept: {e}")
            failed_wikibase_ids.append((wikibase_id, e))
            continue

    refresh_all_available_classifiers([aws_env])
    return failed_wikibase_ids
