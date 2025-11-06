"""
Flow that updates classifiers profiles changes detected in wikibase

Assumes that the classifier model has been trained in wandb
"""

import json
from pathlib import Path

import polars as pl
from prefect import flow
from prefect.artifacts import acreate_table_artifact

from flows.classifier_specs.spec_interface import ClassifierSpec, load_classifier_specs
from flows.config import Config
from flows.result import Err, Error, Ok, Result
from flows.utils import SlackNotify, get_logger
from knowledge_graph.classifiers_profiles import (
    ClassifiersProfileMapping,
    Profile,
    validate_classifiers_profiles_mappings,
)
from knowledge_graph.cloud import AwsEnv, get_aws_ssm_param
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import ClassifierID, WikibaseID
from knowledge_graph.wikibase import WikibaseSession

# TODO: these are in config, update to use config for this
WIKIBASE_PASSWORD_SSM_NAME = "/Wikibase/Cloud/ServiceAccount/Password"
WIKIBASE_USERNAME_SSM_NAME = "/Wikibase/Cloud/ServiceAccount/Username"
WIKIBASE_URL_SSM_NAME = "/Wikibase/Cloud/URL"


def add_classifier_profile(
    wikibase_id: WikibaseID,
    classifier_id: ClassifierID,
    aws_env: AwsEnv,
    classifiers_profile: str,
):
    """Promote a classifier and add classifiers profile"""
    return


def remove_classifier_profile(
    wikibase_id: WikibaseID,
    classifier_id: ClassifierID,
    aws_env: AwsEnv,
    classifiers_profile: str,
):
    """Demote a classifier based on model registry and remove classifiers profile"""
    return


def update_classifier_profile(
    wikibase_id: WikibaseID,
    classifier_id: ClassifierID,
    aws_env: AwsEnv,
    classifiers_profile: str,
):
    """Update classifiers profile for already promoted model"""
    return


def get_wikibase_session(aws_env: AwsEnv):
    #  TODO: update to wikibaseauth
    username = get_aws_ssm_param(
        WIKIBASE_USERNAME_SSM_NAME,
        aws_env=aws_env,
    )
    password = get_aws_ssm_param(
        WIKIBASE_PASSWORD_SSM_NAME,
        aws_env=aws_env,
    )
    url = get_aws_ssm_param(
        WIKIBASE_URL_SSM_NAME,
        aws_env=aws_env,
    )

    wikibase = WikibaseSession(
        username=username,
        password=password,
        url=url,
    )
    return wikibase


async def read_concept_store_local(wikibase: WikibaseSession) -> list[Concept]:
    """Read concept store for classifier IDs"""

    # TODO: Remove, as dev only
    # Check for cached concepts to avoid network download
    cache_path = Path("./tmp/concepts_cache_q218.jsonl")
    if cache_path.exists():
        print(f"loading concepts from cache: {cache_path}")
        concepts = []
        with open(cache_path, "r") as f:
            for line in f:
                concepts.append(Concept.model_validate_json(line))
        print(f"loaded {len(concepts)} concepts from cache")
    else:
        print("getting concepts from wikibase")
        concepts = await wikibase.get_concepts_async()
        print(f"got {len(concepts)} concepts")

        # Save to cache
        print(f"saving concepts to cache: {cache_path}")
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            for concept in concepts:
                f.write(concept.model_dump_json() + "\n")
        print(f"saved {len(concepts)} concepts to cache")

    return concepts


async def read_concept_store(wikibase: WikibaseSession) -> list[Concept]:
    logger = get_logger()

    try:
        concepts = await wikibase.get_concepts_async()
    except Exception as e:
        logger.error(f"Failed to read concept store: {e}")
        raise Exception

    logger.info(f"Loaded {len(concepts)} concepts from wikibase")
    return concepts


async def get_classifiers_profiles(
    wikibase: WikibaseSession, concepts: list[Concept]
) -> tuple[list[ClassifiersProfileMapping], list[Result[WikibaseID, Error]]]:
    """
    Return valid classifiers profiles and different kids of validation errors.

    Validation errors can be invalid concepts and violated business constraints.
    """
    logger = get_logger()

    results: list[Result[WikibaseID, Error]] = []
    classifiers_profiles = []
    for concept in concepts:
        logger.info(f"getting classifier profile for concept: {concept.wikibase_id}")
        try:
            if not concept.wikibase_id:
                results.append(
                    Err(
                        Error(
                            msg="No wikibase ID for concept",
                            metadata={"preferred_label": concept.preferred_label},
                        )
                    )
                )
                continue

            concept_classifiers_profiles = await wikibase.get_classifier_ids_async(
                wikibase_id=concept.wikibase_id
            )
            # TODO: potentially remove this check
            if len(concept_classifiers_profiles) == 0:
                results.append(
                    Err(
                        Error(
                            msg="No classifier ID in Concept",
                            metadata={"wikibase_id": concept.wikibase_id},
                        )
                    )
                )
                continue

            for rank, classifier_id in concept_classifiers_profiles:
                classifiers_profiles.append(
                    ClassifiersProfileMapping(
                        wikibase_id=concept.wikibase_id,
                        classifier_id=classifier_id,
                        classifiers_profile=Profile.generate(rank),
                    )
                )

            logger.info(
                f"Got {len(concept_classifiers_profiles)} classifier profiles from wikibase {concept.wikibase_id}"
            )

        except Exception as e:
            logger.info(f"Error getting classifier ID from wikibase: {e}")
            results.append(
                Err(
                    Error(
                        msg=f"Error getting classifier ID from wikibase: {e}",
                        metadata={"wikibase_id": concept.wikibase_id},
                    )
                )
            )
            continue

    # run validation
    valid_classifiers_profiles = []
    try:
        valid_classifiers_profiles, validation_results = (
            validate_classifiers_profiles_mappings(classifiers_profiles)
        )
        results.extend(validation_results)
    except Exception as e:
        raise Exception(f"Error validating classifiers profiles {e}")

    return valid_classifiers_profiles, results


async def create_validation_artifact(results: list[Result[WikibaseID, Error]]):
    """Create an artifact with a summary of the classifiers profiles validation checks"""

    successes = [r._value for r in results if isinstance(r, Ok)]
    failures = [r._error for r in results if isinstance(r, Err)]

    total_concepts = len(results)
    successful_concepts = len(successes)
    failed_concepts = len(failures)

    overview_description = f"""# Classifiers Profiles Validation Summary
## Overview
- **Total concepts found**: {total_concepts}
- **Successful Wikibase IDs**: {successful_concepts}
- **Failed Wikibase IDs**: {failed_concepts}
"""

    cp_details = [
        {
            "Wikibase ID": str(wikibase_id),
            "Status": "✓",
            "Error": "N/A",
        }
        for wikibase_id in successes
    ] + [
        {
            "Wikibase ID": str((error.metadata or {}).get("wikibase_id", "Unknown")),
            "Status": "✗",
            "Error": (
                f"{error.msg}: {json.dumps((error.metadata or {}).get('response'))}"  # pyright: ignore[reportOptionalMemberAccess]
                if error.metadata and error.metadata.get("response")
                else error.msg
            ),
        }
        for error in failures
    ]

    await acreate_table_artifact(
        key="classifiers-profiles-validation",
        table=cp_details,
        description=overview_description,
    )


def classifier_specs_to_dataframe(
    classifier_specs: list[ClassifierSpec],
) -> pl.DataFrame:
    df = pl.DataFrame(
        [
            spec.model_dump(
                include={
                    "concept_id",
                    "wikibase_id",
                    "classifier_id",
                    "classifiers_profile",
                },
                mode="python",
            )
            for spec in classifier_specs
        ]
    )
    return df


def classifiers_profiles_to_dataframe(
    classifiers_profiles: list[ClassifiersProfileMapping],
) -> pl.DataFrame:
    df = pl.DataFrame(
        [
            cp.model_dump(
                include={
                    "wikibase_id",
                    "classifier_id",
                    "classifiers_profile",
                },
                mode="python",
            )
            for cp in classifiers_profiles
        ]
    )

    return df


def convert_to_classifier_dict(
    dataset: list[ClassifierSpec] | list[ClassifiersProfileMapping],
) -> dict:
    classifier_dict = {}

    for item in dataset:
        # get explicit values
        values = vars(item).copy()
        key = (item.wikibase_id, item.classifier_id)
        classifier_dict[key] = values

    return classifier_dict


def compare_classifiers_profiles(
    classifier_specs: list[ClassifierSpec],
    classifiers_profiles: list[ClassifiersProfileMapping],
) -> list[dict]:
    """
    Compare current classifiers specs to valid classifiers profiles from wikibase

    Classify action to take for each
    wikibase_id, classifier_id pair.
    Actions: add, remove, update, ignore.
    """

    data_current = convert_to_classifier_dict(classifier_specs)
    data_new = convert_to_classifier_dict(classifiers_profiles)

    current_keys = set(data_current.keys())
    new_keys = set(data_new.keys())

    to_remove = [
        {"key": k, **data_current[k], "status": "remove"}
        for k in (current_keys - new_keys)
    ]

    to_add = [
        {"key": k, **data_new[k], "status": "add"} for k in (new_keys - current_keys)
    ]

    common = current_keys & new_keys

    to_update = [
        {"key": k, "status": "update", "current": data_current[k], "new": data_new[k]}
        for k in common
        if data_current[k]["classifiers_profile"] != data_new[k]["classifiers_profile"]
    ]

    to_ignore = [
        {"key": k, **data_current[k], "status": "ignore"}
        for k in common
        if data_current[k]["classifiers_profile"] == data_new[k]["classifiers_profile"]
    ]

    combined_results = [to_remove, to_add, to_update, to_ignore]

    return [d for sublist in combined_results for d in sublist]


@flow(on_failure=[SlackNotify.message])
async def sync_classifiers_profiles(
    aws_env: AwsEnv,
    config: Config | None = None,
):
    """Update classifier profile for a given aws environment."""

    logger = get_logger()

    if not config:
        logger.info("No pipeline config provided, creating default...")
        config = await Config.create()

    logger.info(
        f"Running the classifiers profiles lifecycle with the config: {config}, "
    )
    classifier_specs = load_classifier_specs(aws_env)
    logger.info(
        f"Loaded {len(classifier_specs)} classifier specs for env {aws_env.name}"
    )

    wikibase = get_wikibase_session(aws_env)
    concepts = await read_concept_store(wikibase)
    classifiers_profiles, results = await get_classifiers_profiles(wikibase, concepts)

    logger.info(f"Successful classifiers {len(classifiers_profiles)}")

    updates = compare_classifiers_profiles(classifier_specs, classifiers_profiles)

    # TODO: update artifact to include successful results
    # saves all validation errors only
    await create_validation_artifact(
        results=results,
    )

    logger.info(f"Processed {len(updates)} updates")
