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
    logger = get_logger()

    results: list[Result[WikibaseID, Error]] = []
    classifiers_profiles = []
    valid_classifiers_profiles = []
    for concept in concepts:
        logger.info(f"getting classifier profile for concept: {concept.wikibase_id}")
        try:
            if concept.wikibase_id:
                concept_classifiers_profiles = await wikibase.get_classifier_ids_async(
                    wikibase_id=concept.wikibase_id
                )
                # TODO: potentially remove this check
                if len(concept_classifiers_profiles) == 0:
                    results.append(
                        Err(
                            Error(
                                msg="No classifier ID in wikibase",
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

            else:
                results.append(
                    Err(
                        Error(
                            msg="No wikibase ID for concept",
                            metadata={"preferred_label": concept.preferred_label},
                        )
                    )
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


async def create_model_changes_artifact(updates: pl.DataFrame):
    """Create an artifact with a summary of the classifiers profiles changes"""

    unchanged = updates.filter(pl.col("status") == "same").to_dicts()
    proposed_updates = updates.filter(
        pl.col("status").is_in(["add", "remove", "update"])
    ).to_dicts()
    failures = updates.filter(pl.col("status") == "unknown").to_dicts()

    total_classifiers = len(updates)
    failed_classifiers = len(failures)
    proposed_changes = len(proposed_updates)
    unchanged_classifiers = len(unchanged)

    overview_description = f"""# Classifiers Profiles Validation Summary
## Overview
- **Total classifiers found in specs and wikibase**: {total_classifiers}
- **Proposed changes**: {proposed_changes}
- **No changes needed**: {unchanged_classifiers}
- **Failed classifiers**: {failed_classifiers}
"""

    cp_details = (
        [
            {
                "Wikibase ID": classifier.get("wikibase_id"),
                "Classifier ID": classifier.get("classifier_id"),
                "Classifiers Profile": classifier.get("classifiers_profile_changes"),
                "Change": classifier.get("status"),
                "Status": "✓",
            }
            for classifier in proposed_updates
        ]
        + [
            {
                "Wikibase ID": classifier.get("wikibase_id"),
                "Classifier ID": classifier.get("classifier_id"),
                "Classifiers Profile": classifier.get("classifiers_profile"),
                "Change": classifier.get("status"),
                "Status": "✗",
            }
            for classifier in failures
        ]
        + [
            {
                "Wikibase ID": classifier.get("wikibase_id"),
                "Classifier ID": classifier.get("classifier_id"),
                "Classifiers Profile": classifier.get("classifiers_profile"),
                "Change": classifier.get("status"),
                "Status": "-",
            }
            for classifier in unchanged
        ]
    )

    await acreate_table_artifact(
        key="classifiers-profiles-changes",
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


def compare_classifiers_profiles(
    classifier_specs: list[ClassifierSpec],
    classifiers_profiles: list[ClassifiersProfileMapping],
) -> pl.DataFrame:
    specs_df = classifier_specs_to_dataframe(classifier_specs)
    cp_df = classifiers_profiles_to_dataframe(classifiers_profiles)

    updates_df = specs_df.join(
        cp_df, on=["wikibase_id", "classifier_id"], how="full", suffix="_update"
    )

    updates_df = updates_df.with_columns(
        pl.when(
            pl.col("wikibase_id").is_not_null()
            & (pl.col("wikibase_id") == pl.col("wikibase_id_update"))
            & pl.col("classifier_id").is_not_null()
            & (pl.col("classifier_id") == pl.col("classifier_id_update"))
            & pl.col("classifiers_profile").is_not_null()
            & (pl.col("classifiers_profile") == pl.col("classifiers_profile_update"))
        )
        .then(pl.lit("same"))
        .when(
            pl.col("wikibase_id").is_not_null()
            & (pl.col("wikibase_id") == pl.col("wikibase_id_update"))
            & pl.col("classifier_id").is_not_null()
            & (pl.col("classifier_id") == pl.col("classifier_id_update"))
            & pl.col("classifiers_profile").is_not_null()
            & pl.col("classifiers_profile_update").is_not_null()
            & (pl.col("classifiers_profile") != pl.col("classifiers_profile_update"))
        )
        .then(pl.lit("update"))
        .when(
            pl.col("wikibase_id").is_not_null()
            & pl.col("classifier_id").is_not_null()
            & pl.col("classifiers_profile").is_not_null()
            & pl.col("wikibase_id_update").is_null()
            & pl.col("classifier_id_update").is_null()
            & pl.col("classifiers_profile_update").is_null()
        )
        .then(pl.lit("remove"))
        .when(
            pl.col("wikibase_id").is_null()
            & pl.col("classifier_id").is_null()
            & pl.col("classifiers_profile").is_null()
            & pl.col("wikibase_id_update").is_not_null()
            & pl.col("classifier_id_update").is_not_null()
            & pl.col("classifiers_profile_update").is_not_null()
        )
        .then(pl.lit("add"))
        .otherwise(pl.lit("unknown"))  # Handle any unmatched rows
        .alias("status")
    )
    updates_df = updates_df.with_columns(
        pl.coalesce([pl.col("wikibase_id"), pl.col("wikibase_id_update")]).alias(
            "wikibase_id"
        ),
        pl.coalesce([pl.col("classifier_id"), pl.col("classifier_id_update")]).alias(
            "classifier_id"
        ),
    ).drop(["wikibase_id_update", "classifier_id_update"])

    updates_df = (
        updates_df.with_columns(
            pl.format(
                "{} → {}",
                pl.col("classifiers_profile"),
                pl.col("classifiers_profile_update"),
            ).alias("classifiers_profile_changes")
        )
        .drop(["classifiers_profile"])
        .rename({"classifiers_profile_update": "classifiers_profile"})
    )

    return updates_df


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

    updates_df = compare_classifiers_profiles(classifier_specs, classifiers_profiles)

    # TODO: combine into 1 artifact with all results
    await create_validation_artifact(
        results=results,
    )
    await create_model_changes_artifact(updates_df)

    # TODO(PLA-948):
    # * - check classifier exists in w&b
    # * - validation checks
    # 8 - run promote / demote / update
    # 9 - update classifier spec file
    # 10 - commit changes to git repo
    # 11 - trigger full-pipeline (Optional?)
    # 12 - update vespa classifiers profiles
