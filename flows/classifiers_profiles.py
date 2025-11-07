"""
Flow that updates classifiers profiles changes detected in wikibase

Assumes that the classifier model has been trained in wandb
"""

import os
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypedDict

import wandb
from prefect.artifacts import acreate_table_artifact
from pydantic import AnyHttpUrl, SecretStr

from flows.classifier_specs.spec_interface import ClassifierSpec, load_classifier_specs
from flows.config import Config
from flows.result import Err, Error, Ok, Result
from flows.utils import get_logger
from knowledge_graph.classifier import ModelPath
from knowledge_graph.classifiers_profiles import (
    ClassifiersProfileMapping,
    Profile,
    validate_classifiers_profiles_mappings,
)
from knowledge_graph.cloud import AwsEnv, get_aws_ssm_param
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import ClassifierID, WikibaseID
from knowledge_graph.version import Version, get_latest_model_version
from knowledge_graph.wikibase import WikibaseAuth, WikibaseSession

WIKIBASE_PASSWORD_SSM_NAME = "/Wikibase/Cloud/ServiceAccount/Password"
WIKIBASE_USERNAME_SSM_NAME = "/Wikibase/Cloud/ServiceAccount/Username"
WIKIBASE_URL_SSM_NAME = "/Wikibase/Cloud/URL"


def log_and_return_error(logger, msg: str, metadata: dict) -> Err:
    logger.info(msg)
    # TODO remove print statement
    print(msg)
    return Err(Error(msg=msg, metadata=metadata))


def wandb_validation(
    wikibase_id: WikibaseID,
    aws_env: AwsEnv,
    classifier_id: Optional[ClassifierID] = None,
    wandb_registry_version: Optional[Version] = None,
) -> Result[WikibaseID, Error]:
    """Validate artifact for updates exists, return Result with artifact or error"""
    logger = get_logger()

    api = wandb.Api()
    artifact_path = ""

    try:
        if wikibase_id and wandb_registry_version:
            artifact_path = (
                f"wandb-registry-model/{wikibase_id}:{wandb_registry_version}"
            )

        elif wikibase_id and classifier_id:
            model_path = ModelPath(wikibase_id=wikibase_id, classifier_id=classifier_id)

            artifacts = api.artifacts(type_name="model", name=f"{model_path}")
            classifier_version = get_latest_model_version(artifacts, aws_env)
            artifact_path = f"{model_path}:{classifier_version}"

        if artifact_path == "":
            return log_and_return_error(
                logger,
                msg="Error artifact not found",
                metadata={
                    "wikibase_id": wikibase_id,
                    "classifier_id": classifier_id,
                    "wandb_registry_version": wandb_registry_version,
                },
            )

        artifact = api.artifact(artifact_path, type="model")
        result = validate_artifact_metadata_rules(artifact=artifact)

        # If validation fails, append metadata to the error
        if isinstance(result, Err) and result._error.metadata is not None:
            result._error.metadata.update(
                {
                    "wikibase_id": wikibase_id,
                    "classifier_id": classifier_id,
                    "wandb_registry_version": wandb_registry_version,
                }
            )
            return result
        else:
            return Ok(wikibase_id)

    except Exception as e:
        # urllib3.exceptions.ReadTimeoutError: HTTPSConnectionPool(host='api.wandb.ai', port=443): Read timed out. (read timeout=19)
        return log_and_return_error(
            logger,
            msg=f"Error retrieving artifact: {e}",
            metadata={
                "wikibase_id": wikibase_id,
                "classifier_id": classifier_id,
                "wandb_registry_version": wandb_registry_version,
            },
        )


def validate_artifact_metadata_rules(
    artifact: wandb.Artifact,
) -> Result[wandb.Artifact, Error]:
    """Validate against data science rules for artifact metadata, return Result"""
    logger = get_logger()

    # data science: validation rules for wandb
    restricted_classifier_names = ["LLMClassifier", "KeywordExpansion", "Embedding"]
    restricted_run_configs = {
        "experimental_concept": True,
        "experimental_model_type": True,
    }

    if artifact.metadata.get("classifier_name") in restricted_classifier_names:
        return log_and_return_error(
            logger,
            msg="Error artifact validation failed for classifier type",
            metadata={},
        )

    # get the run config metadata
    producer_run = artifact.logged_by()
    if producer_run is not None:
        run_config = producer_run.config

        for key in restricted_run_configs:
            if key in run_config and run_config.get(key) == restricted_run_configs.get(
                key
            ):
                return log_and_return_error(
                    logger,
                    msg="Error artifact validation failed for run config",
                    metadata={},
                )
    return Ok(artifact)


def handle_classifier_profile_action(
    action: str,
    wikibase_id: WikibaseID,
    aws_env: AwsEnv,
    additional_args: dict,
    action_function: Callable[..., Any],
) -> Result[Dict, Error]:
    """Run promote/demote/update based on params"""
    logger = get_logger()

    wandb_registry_version = additional_args.get("wandb_registry_version", None)
    classifier_id = additional_args.get("classifier_id", None)

    result = wandb_validation(
        wikibase_id=wikibase_id,
        classifier_id=classifier_id,
        wandb_registry_version=wandb_registry_version,
        aws_env=aws_env,
    )
    if isinstance(result, Err):
        return result

    try:
        action_function(wikibase_id, aws_env, **additional_args)

    except Exception as e:
        return log_and_return_error(
            logger,
            f"Error {action} classifier profile: {e}",
            {
                "wikibase_id": wikibase_id,
                "classifier_id": classifier_id,
                **additional_args,
            },
        )

    result = {
        "wikibase_id": wikibase_id,
        "classifier_id": classifier_id,
        **additional_args,
        "status": action,
    }
    return Ok(result)


def promote_classifier_profile(
    current_specs: ClassifierSpec | None,
    new_specs: ClassifiersProfileMapping,
    aws_env: AwsEnv,
) -> Result[Dict, Error]:
    """
    Promote a classifier and add classifiers profile.

    Use new_specs to get the details for promotion.
    """
    #     scripts.promote.main(
    #         wikibase_id=wikibase_id,
    #         classifier_id=classifier_id,
    #         aws_env=aws_env,
    #         add_classifiers_profiles=classifiers_profile,
    #     )

    return handle_classifier_profile_action(
        action="promoting",
        wikibase_id=new_specs.wikibase_id,
        aws_env=aws_env,
        additional_args={
            "classifier_id": new_specs.classifier_id,
            "classifiers_profile": [str(new_specs.classifiers_profile)],
        },
        action_function=lambda w_id, env, classifier_id, classifiers_profile: print(
            f"Promoting {w_id}, {classifier_id}, {classifiers_profile}, {env}"
        ),
    )


def demote_classifier_profile(
    current_specs: ClassifierSpec,
    new_specs: ClassifiersProfileMapping | None,
    aws_env: AwsEnv,
) -> Result[Dict, Error]:
    """
    Demote a classifier based on model registry and remove classifiers profile".

    Use current_specs to get the details for demotion.
    """
    #     scripts.demote.main(
    #         wikibase_id=wikibase_id,
    #         wandb_registry_version=wandb_registry_version,
    #         aws_env=aws_env
    #     )
    additional_args: dict = {
        "wandb_registry_version": current_specs.wandb_registry_version
    }
    # Check if classifier_id is not None and add it to additional_args
    if current_specs.classifier_id is not None:
        additional_args["classifier_id"] = current_specs.classifier_id

    # Check if classifiers_profile is not None and add it to additional_args
    if current_specs.classifiers_profile is not None:
        additional_args["classifiers_profile"] = current_specs.classifiers_profile

    return handle_classifier_profile_action(
        action="demoting",
        wikibase_id=current_specs.wikibase_id,
        aws_env=aws_env,
        additional_args=additional_args,
        action_function=lambda w_id, env, **kwargs: print(f"Demoting {w_id}, {env}"),
    )


def update_classifier_profile(
    current_specs: ClassifierSpec,
    new_specs: ClassifiersProfileMapping,
    aws_env: AwsEnv,
) -> Result[Dict, Error]:
    """
    Update classifiers profile for already promoted model.

    Use current_specs and new_specs to get the details for update.
    """
    # scripts.classifier_metadata.update(
    #     wikibase_id=wikibase_id,
    #     classifier_id=classifier_id,
    #     add_classifiers_profiles=[add_classifiers_profile],
    #     remove_classifiers_profiles=remove_classifiers_profile_value,
    #     aws_env=aws_env,
    #     update_specs = False
    # )
    return handle_classifier_profile_action(
        action="updating",
        wikibase_id=current_specs.wikibase_id,
        aws_env=aws_env,
        additional_args={
            "classifier_id": current_specs.classifier_id,
            "remove_classifiers_profile": [current_specs.classifiers_profile],
            "add_classifiers_profile": [new_specs.classifiers_profile],
        },
        action_function=lambda w_id,
        env,
        classifier_id,
        add_classifiers_profile,
        remove_classifiers_profile: print(
            f"Updating {w_id}, {env}, {classifier_id}, {str(add_classifiers_profile[0])}, {str(remove_classifiers_profile[0])}"
        ),
    )


async def read_concepts(
    wikibase_auth: WikibaseAuth,
    wikibase_cache_path: Path | None,  # path to JSONL file
    wikibase_cache_save_if_missing: bool,
) -> list[Concept]:
    """Read concepts from wikibase or specified cache"""

    logger = get_logger()

    wikibase = WikibaseSession(
        username=wikibase_auth.username,
        password=str(wikibase_auth.password),
        url=str(wikibase_auth.url),
    )

    if wikibase_cache_path and wikibase_cache_path.exists():
        logger.info(f"loading concepts from cache: {wikibase_cache_path}")
        concepts = []
        with open(wikibase_cache_path, "r") as f:
            for line in f:
                concepts.append(Concept.model_validate_json(line))
        logger.info(f"loaded {len(concepts)} concepts from cache")
    else:
        logger.info("getting concepts from Wikibase")

        try:
            concepts = await wikibase.get_concepts_async()
        except Exception as e:
            logger.error(f"Failed to read concept store: {e}")
            raise Exception(f"Failed to read concept store: {e}")

        logger.info(f"Loaded {len(concepts)} concepts from wikibase")

        # Save to cache
        if wikibase_cache_path and wikibase_cache_save_if_missing:
            logger.info(f"saving concepts to cache: {wikibase_cache_path}")
            wikibase_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(wikibase_cache_path, "w") as f:
                for concept in concepts:
                    f.write(concept.model_dump_json() + "\n")
            logger.info(f"saved {len(concepts)} concepts to cache")

    return concepts


async def get_classifiers_profiles(
    wikibase_auth: WikibaseAuth, concepts: list[Concept]
) -> list[Result[ClassifiersProfileMapping, Error]]:
    """
    Return valid classifiers profiles and different kids of validation errors.

    Validation errors can be invalid concepts and violated business constraints.
    """
    logger = get_logger()

    wikibase = WikibaseSession(
        username=wikibase_auth.username,
        password=str(wikibase_auth.password),
        url=str(wikibase_auth.url),
    )

    results: list[Result[ClassifiersProfileMapping, Error]] = []
    classifiers_profiles = []
    for concept in concepts:
        logger.info(f"getting classifier profile for concept: {concept.wikibase_id}")
        try:
            if not concept.wikibase_id:
                results.append(
                    Err(
                        Error(
                            msg=f"No wikibase ID for concept: {concept.preferred_label}",
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
    try:
        validation_results = validate_classifiers_profiles_mappings(
            classifiers_profiles
        )
        results.extend(validation_results)
    except Exception as e:
        raise Exception(f"Error validating classifiers profiles {e}")

    return results


async def create_classifiers_profiles_artifact(results: list[Result[Dict, Error]]):
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

    def format_cp_details(
        concept: dict, status: str, error: Optional[str] = None
    ) -> dict:
        return {
            "Wikibase ID": str(concept.get("wikibase_id", "Unknown")),
            "Classifier ID": str(concept.get("classifier_id", "Unknown")),
            "Classifiers Profile": (
                f"{str(concept.get('add_classifiers_profile', [None])[0])} to {str(concept.get('remove_classifiers_profile', [None])[0])} ({concept.get('status')})"
                if concept.get("add_classifiers_profile")
                else f"{str(concept.get('classifiers_profile'))} ({concept.get('status', '')})"
            ),
            "Status": status,
            "Error": error or "N/A",
        }

    cp_details = [format_cp_details(concept, "✓") for concept in successes] + [
        format_cp_details(error.metadata or {}, "✗", error.msg) for error in failures
    ]

    # TODO remove print statements
    print(overview_description)
    print(cp_details)

    await acreate_table_artifact(
        key="classifiers-profiles-validation",
        table=cp_details,
        description=overview_description,
    )


def convert_to_classifier_dict(
    dataset: list[ClassifierSpec] | list[ClassifiersProfileMapping],
) -> dict[tuple[WikibaseID, ClassifierID], dict]:
    classifier_dict = {}

    for item in dataset:
        # get explicit values
        values = vars(item).copy()
        key = (item.wikibase_id, item.classifier_id)
        classifier_dict[key] = values

    return classifier_dict


class CompareResult(TypedDict):
    """
    Class with results of comparing classifiers profiles.

    Contains current classifier specs and new classifiers profiles from wikibase.
    """

    key: tuple[WikibaseID, ClassifierID]
    status: str
    current: ClassifierSpec | None
    new: ClassifiersProfileMapping | None


def compare_classifiers_profiles(
    classifier_specs: list[ClassifierSpec],
    classifiers_profiles: list[ClassifiersProfileMapping],
) -> list[CompareResult]:
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

    # TODO: update convert_dict_to_classifier_spec to instead retrieve from ClassifierSpec
    to_remove: list[CompareResult] = [
        {
            "key": k,
            "status": "remove",
            "current": convert_dict_to_classifier_spec(data_current[k]),
            "new": None,
        }
        for k in (current_keys - new_keys)
    ]

    to_add: list[CompareResult] = [
        {
            "key": k,
            "status": "add",
            "current": None,
            "new": convert_dict_to_classifiers_profile_mapping(data_new[k]),
        }
        for k in (new_keys - current_keys)
    ]

    common = current_keys & new_keys

    to_update: list[CompareResult] = [
        {
            "key": k,
            "status": "update",
            "current": convert_dict_to_classifier_spec(data_current[k]),
            "new": convert_dict_to_classifiers_profile_mapping(data_new[k]),
        }
        for k in common
        if data_current[k]["classifiers_profile"] != data_new[k]["classifiers_profile"]
    ]

    combined_results = [to_remove, to_add, to_update]  # not including no changes

    return [d for sublist in combined_results for d in sublist]


def convert_dict_to_classifier_spec(
    data: dict,
) -> ClassifierSpec:
    # check required fields are in data
    required_fields = ["wikibase_id", "classifier_id", "wandb_registry_version"]
    for field in required_fields:
        if field not in data:
            raise ValueError(
                f"Error converting dict to ClassifierSpec, missing required field: {field}"
            )

    return ClassifierSpec(
        wikibase_id=WikibaseID(data.get("wikibase_id")),
        classifier_id=ClassifierID(data.get("classifier_id")),
        wandb_registry_version=str(data.get("wandb_registry_version")),  # type: ignore
        classifiers_profile=data.get("classifiers_profile", None),
    )


def convert_dict_to_classifiers_profile_mapping(
    data: dict,
) -> ClassifiersProfileMapping:
    # check required fields are in data
    required_fields = ["wikibase_id", "classifier_id", "classifiers_profile"]
    for field in required_fields:
        if field not in data:
            raise ValueError(
                f"Error converting dict to ClassifiersProfileMapping, missing required field: {field}"
            )

    return ClassifiersProfileMapping(
        wikibase_id=WikibaseID(data.get("wikibase_id")),
        classifier_id=ClassifierID(data.get("classifier_id")),
        classifiers_profile=Profile(data.get("classifiers_profile")),
    )


# @flow(on_failure=[SlackNotify.message])
async def sync_classifiers_profiles(
    aws_env: AwsEnv,
    config: Config | None = None,
    wikibase_auth: WikibaseAuth | None = None,
    wikibase_cache_path: Path | None = None,
    wikibase_cache_save_if_missing: bool = False,
):
    """Update classifier profile for a given aws environment."""

    logger = get_logger()

    print("Wikibase Cache Path:", wikibase_cache_path)
    if not config:
        logger.info("No pipeline config provided, creating default...")
        config = await Config.create()

    if wikibase_auth is None:
        wikibase_password = SecretStr(get_aws_ssm_param(WIKIBASE_PASSWORD_SSM_NAME))
        wikibase_username = get_aws_ssm_param(WIKIBASE_USERNAME_SSM_NAME)
        wikibase_url = get_aws_ssm_param(WIKIBASE_URL_SSM_NAME)
        # Set as env var so Concept.wikibase_url property can access it
        os.environ["WIKIBASE_URL"] = wikibase_url
        wikibase_auth = WikibaseAuth(
            username=wikibase_username,
            password=wikibase_password,
            url=AnyHttpUrl(wikibase_url),
        )

    if config.wandb_api_key is None:
        raise ValueError("Wandb API key is not set in the config.")

    logger.info(
        f"Running the classifiers profiles lifecycle with the config: {config}, "
    )
    classifier_specs = load_classifier_specs(aws_env)
    logger.info(
        f"Loaded {len(classifier_specs)} classifier specs for env {aws_env.name}"
    )

    concepts = await read_concepts(
        wikibase_auth, wikibase_cache_path, wikibase_cache_save_if_missing
    )

    # returns Result with valid classifiers profiles and validation errors
    results = await get_classifiers_profiles(wikibase_auth, concepts)

    validation_errors: list[Result[Dict, Error]] = [
        Err(Error(msg=r._error.msg, metadata=r._error.metadata))
        for r in results
        if isinstance(r, Err)
    ]
    classifiers_profiles: list[ClassifiersProfileMapping] = [
        r._value for r in results if isinstance(r, Ok)
    ]

    logger.info(f"Valid concept retrieved from wikibase: {len(classifiers_profiles)}")

    print(f"Validation errors: {len(validation_errors)}")
    print(f"Successful classifiers profiles: {len(classifiers_profiles)}")

    updates = compare_classifiers_profiles(classifier_specs, classifiers_profiles)
    logger.info(f"Identified {len(updates)} updates")

    wandb.login(key=config.wandb_api_key.get_secret_value())

    wandb_results: list[Result[Dict, Error]] = []
    UPDATE_HANDLERS = {
        "add": promote_classifier_profile,
        "remove": demote_classifier_profile,
        "update": update_classifier_profile,
    }

    for update in updates:
        if status := update.get("status"):
            if handler := UPDATE_HANDLERS.get(status):
                wandb_results.append(
                    handler(
                        current_specs=update.get("current", None),
                        new_specs=update.get("new", None),
                        aws_env=aws_env,
                    )
                )
            else:
                print(f"Unhandled status: {status}")

    # update classifiers specs
    # refresh_all_available_classifiers([aws_env])

    # combine validation errors with wandb errors and successes
    final_results: list[Result[Dict, Error]] = validation_errors + wandb_results

    # create artifact with summary
    await create_classifiers_profiles_artifact(
        results=final_results,
    )


if __name__ == "__main__":
    # sync_classifiers_profiles.serve(name="classifiers-profiles-lifecycle")
    import asyncio

    asyncio.run(
        sync_classifiers_profiles(
            aws_env=AwsEnv.staging,
            wikibase_cache_path=Path("./tmp/concepts_cache_q218.jsonl"),
            wikibase_cache_save_if_missing=True,
        )
    )
