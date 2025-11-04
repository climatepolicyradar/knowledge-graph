# Flow that updates classifiers profiles changes detected in wikibase
# assumes that the classifier model has been trained in wandb
import json
import os
from pathlib import Path

from prefect import flow
from prefect.artifacts import acreate_table_artifact

from flows.classifier_specs.spec_interface import load_classifier_specs
from flows.config import Config
from flows.result import Err, Error, Ok, Result
from flows.utils import get_logger
from knowledge_graph.classifiers_profiles import (
    ClassifiersProfile,
    ClassifiersProfiles,
    Profile,
)
from knowledge_graph.cloud import AwsEnv, get_aws_ssm_param
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import ClassifierID, WikibaseID
from knowledge_graph.wikibase import WikibaseSession

# these are in config, update to use config for this
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


async def read_concept_store(wikibase: WikibaseSession) -> list[Concept]:
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


async def get_classifier_profiles(
    wikibase: WikibaseSession, concepts: list[Concept]
) -> tuple[ClassifiersProfiles, list[Result[WikibaseID, Error]]]:
    logger = get_logger()

    results: list[Result[WikibaseID, Error]] = []
    classifier_profiles = ClassifiersProfiles()

    for concept in concepts:
        logger.info(f"Concept wikibase id: {concept.wikibase_id}")
        try:
            wikibase_id = WikibaseID(concept.wikibase_id)
            concept_classifier_profiles = await wikibase.get_classifier_ids_async(
                wikibase_id=wikibase_id
            )
            if len(concept_classifier_profiles) == 0:
                results.append(
                    Err(
                        Error(
                            msg="No classifier ID in wikibase",
                            metadata={"wikibase_id": wikibase_id},
                        )
                    )
                )
                continue

            for rank, classifier_id in concept_classifier_profiles:
                classifier_profiles.append(
                    ClassifiersProfile(
                        wikibase_id=wikibase_id,
                        classifier_id=classifier_id,
                        classifier_profile=Profile.generate(rank),
                    )
                )
            logger.info(
                f"Got {len(concept_classifier_profiles)} classifier profiles from wikibase {concept.wikibase_id}"
            )
            results.append(Ok(wikibase_id))
        except Exception as e:
            logger.info(f"{e}")
            results.append(
                Err(
                    Error(
                        msg=str(e),
                        metadata={"wikibase_id": concept.wikibase_id},
                    )
                )
            )
            continue

    return classifier_profiles, results


async def create_validation_artifact(results: list[Result[WikibaseID, Error]]):
    """Create an artifact with a summary of the classifiers profiles changes"""

    successes = [r._value for r in results if isinstance(r, Ok)]
    failures = [r._error for r in results if isinstance(r, Err)]

    total_concepts = len(results)
    successful_cps = len(successes)
    failed_cps = len(failures)

    overview_description = f"""# Classifiers Profiles Validation Summary
## Overview
- **Total concepts found**: {total_concepts}
- **Successful wikibase IDs**: {successful_cps}
- **Failed wikibase IDs**: {failed_cps}
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


@flow(
    # on_failure=[SlackNotify.message],
)
async def classifiers_profiles_lifecycle(
    aws_env: AwsEnv = AwsEnv.staging,
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

    # 1 - read classifier specs file (NOT YET USED)
    specs = load_classifier_specs(aws_env)
    logger.info(f"Loaded {len(specs)} classifier specs for env {aws_env.name}")
    logger.info(f"AWS ENV {os.getenv('AWS_ENV')}")

    # 2 - read concept store classifier profiles
    wikibase = get_wikibase_session(aws_env)
    concepts = await read_concept_store(wikibase)

    # 2a - get classifier profiles for all concepts
    classifier_profiles, results = await get_classifier_profiles(wikibase, concepts)

    logger.info(f"Successful classifiers {len(classifier_profiles)}")
    logger.info(f"Valid profiles: {classifier_profiles}")

    # 6a - create artifact for validation success / failures
    # Create artifact with results
    await create_validation_artifact(
        results=results,
    )

    # next:
    # 4 - create dataframe to compare current vs updates
    # * - check classifier exists in w&b
    # 7 - for each row in dataframe flag for add / remove / update profile

    # later / separate PLA-948:
    # 8 - run promote / demote / update
    # 9 - update classifier spec file
    # 10 - commit changes to git repo
    # 11 - trigger full-pipeline (Optional?)
    # 12 - update vespa classifiers profiles


if __name__ == "__main__":
    classifiers_profiles_lifecycle.serve(
        name="classifiers-profiles-lifecycle", tags=["debug"]
    )
