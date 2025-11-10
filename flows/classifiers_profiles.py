"""
Flow that updates classifiers profiles changes detected in wikibase

Assumes that the classifier model has been trained in wandb
"""

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional, TypedDict

import wandb
from prefect import flow
from prefect.artifacts import acreate_table_artifact
from prefect.context import FlowRunContext, get_run_context
from pydantic import AnyHttpUrl, SecretStr

from flows.classifier_specs.spec_interface import ClassifierSpec, load_classifier_specs
from flows.config import Config
from flows.result import Err, Error, Ok, Result
from flows.utils import get_logger, get_slack_client
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
from scripts.update_classifier_spec import refresh_all_available_classifiers

WIKIBASE_PASSWORD_SSM_NAME = "/Wikibase/Cloud/ServiceAccount/Password"
WIKIBASE_USERNAME_SSM_NAME = "/Wikibase/Cloud/ServiceAccount/Username"
WIKIBASE_URL_SSM_NAME = "/Wikibase/Cloud/URL"


def log_and_return_error(logger, msg: str, metadata: dict) -> Err:
    logger.info(msg)
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
    logger = get_logger()
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
        action_function=lambda w_id,
        env,
        classifier_id,
        classifiers_profile: logger.info(
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
    logger = get_logger()

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
        action_function=lambda w_id, env, **kwargs: logger.info(
            f"Demoting {w_id}, {env}"
        ),
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
    logger = get_logger()

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
        remove_classifiers_profile: logger.info(
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

    combined_results = [to_remove, to_add, to_update]  # not including unchanged

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


async def send_classifiers_profile_slack_alert(
    validation_errors: list[Error],
    other_errors: list[Error],
    successes: list[Dict],
    aws_env: AwsEnv,
):
    """
    Send slack alert with failures from the classifiers profiles lifecycle sync.

    Posts a summary message to slack channel and thread with table of failures.
    """
    logger = get_logger()
    slack_client = await get_slack_client()

    total_concepts = len(successes) + len(validation_errors) + len(other_errors)

    try:
        channel = f"alerts-platform-{aws_env}"
        # TODO: change channel once CP data populated
        # channel = "alerts-concept-store"
        if len(validation_errors) > 0:
            channel = channel
            main_response = await _post_errors_main(
                slack_client=slack_client,
                channel=channel,
                total_concepts=total_concepts,
                validation_errors=validation_errors,
                other_errors=other_errors,
            )

            if not main_response.get("ok"):
                logger.error(f"Slack API response: {main_response}")
                raise Exception(
                    f"failed to send main response to Slack channel #{channel}: {main_response}"
                )

            logger.info(
                f"sent main alert to Slack channel #{channel}: {main_response['ok']}"
            )

            # Get thread_ts for threading replies
            if thread_ts := main_response.get("ts"):
                # Post issues to thread
                if validation_errors:
                    await _post_errors_thread(
                        slack_client=slack_client,
                        channel=channel,
                        thread_ts=thread_ts,
                        errors=validation_errors,
                        error_type="Data Quality Issues",
                    )
            else:
                raise ValueError(
                    f"no thread TS in main response for Data Quality Issues: {main_response}"
                )

        if len(other_errors) > 0:
            channel = f"alerts-platform-{aws_env}"

            main_response = await _post_errors_main(
                slack_client=slack_client,
                channel=channel,
                total_concepts=total_concepts,
                validation_errors=validation_errors,
                other_errors=other_errors,
            )

            if not main_response.get("ok"):
                logger.error(f"Slack API response: {main_response}")
                raise Exception(
                    f"failed to send main response to Slack channel #{channel}: {main_response}"
                )

            logger.info(
                f"sent main alert to Slack channel #{channel}: {main_response['ok']}"
            )

            # Get thread_ts for threading replies
            if thread_ts := main_response.get("ts"):
                if other_errors:
                    await _post_errors_thread(
                        slack_client=slack_client,
                        # channel="alerts-concept-store",
                        channel="alerts-platform-staging",
                        thread_ts=thread_ts,
                        errors=other_errors,
                        error_type="System Errors",
                    )
            else:
                raise ValueError(
                    f"no thread TS in main response for System Errors: {main_response}"
                )
    except Exception as e:
        logger.error(f"failed to send Slack alerts: {e}")
    return


async def _post_errors_main(
    slack_client,
    channel: str,
    total_concepts: int,
    validation_errors: list[Error],
    other_errors: list[Error],
):
    get_logger()
    failures = len(validation_errors) + len(other_errors)

    # Create summary blocks
    summary_blocks: list[dict[str, Any]] = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"{failures} of {total_concepts} classifiers profiles failed with wikibase validation errors.",
            },
        },
        {
            "type": "section",
            "fields": [
                {
                    "type": "mrkdwn",
                    "text": f"*Data Quality Issues*\n{len(validation_errors)}",
                },
                {"type": "mrkdwn", "text": f"*System Errors*\n{len(other_errors)}"},
            ],
        },
    ]

    run_context = get_run_context()
    # Set a default, just in case, to prioritise getting an alert out
    flow_run_name = "unknown"
    if isinstance(run_context, FlowRunContext) and run_context.flow_run:
        flow_run_name = run_context.flow_run.name

    # Add context footer
    timestamp = int(datetime.now(timezone.utc).timestamp())

    summary_blocks.append(
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"Flow Run: `{flow_run_name}` | <!date^{timestamp}^{{date_num}} {{time_secs}}|{datetime.now(timezone.utc).isoformat()}>",
                },
            ],
        }
    )

    failure_rate = (failures / total_concepts) * 100 if total_concepts > 0 else 0
    # Determine colour based on failure rate
    if failure_rate >= 50:
        color = "#e01e5a"  # Red
    else:
        color = "#ecb22e"  # Orange

    return await slack_client.chat_postMessage(
        channel=channel,
        text="Classifiers Profile Sync Summary",
        attachments=[
            {
                "color": color,
                "blocks": summary_blocks,
            }
        ],
    )


async def _post_errors_thread(
    slack_client,
    channel: str,
    thread_ts: str,
    errors: list[Error],
    error_type: str,
):
    """Post errors details as a thread with table of wikibase_id, classifier_id and error message."""
    logger = get_logger()

    # Build table with header row
    table_rows = [
        [
            {
                "type": "rich_text",
                "elements": [
                    {
                        "type": "rich_text_section",
                        "elements": [
                            {
                                "type": "text",
                                "text": "Wikibase ID",
                                "style": {"bold": True},
                            }
                        ],
                    }
                ],
            },
            {
                "type": "rich_text",
                "elements": [
                    {
                        "type": "rich_text_section",
                        "elements": [
                            {
                                "type": "text",
                                "text": "Classifier ID",
                                "style": {"bold": True},
                            }
                        ],
                    }
                ],
            },
            {
                "type": "rich_text",
                "elements": [
                    {
                        "type": "rich_text_section",
                        "elements": [
                            {
                                "type": "text",
                                "text": "Issue",
                                "style": {"bold": True},
                            }
                        ],
                    }
                ],
            },
        ]
    ]

    # Helper function to convert sets to comma-separated strings
    def convert_set_to_string(value: Any) -> str:
        if isinstance(value, set):
            return ", ".join(map(str, value))  # Convert set to comma-separated string
        return str(value)

    # Add data rows
    for error in errors:
        # Get Wikibase ID from metadata
        if error.metadata and "wikibase_id" in error.metadata:
            wikibase_id = convert_set_to_string(error.metadata.get("wikibase_id"))
        else:
            logger.warning(f"error metadata was missing Wikibase ID: {error.metadata}")
            continue

        if error.metadata and "classifier_id" in error.metadata:
            classifier_id = convert_set_to_string(error.metadata.get("classifier_id"))
        else:
            classifier_id = "N/A"

        table_rows.append(
            [
                {
                    "type": "rich_text",
                    "elements": [
                        {
                            "type": "rich_text_section",
                            "elements": [
                                {
                                    "type": "text",
                                    "text": wikibase_id,
                                },
                            ],
                        }
                    ],
                },
                {
                    "type": "rich_text",
                    "elements": [
                        {
                            "type": "rich_text_section",
                            "elements": [
                                {
                                    "type": "text",
                                    "text": classifier_id,
                                }
                            ],
                        }
                    ],
                },
                {
                    "type": "rich_text",
                    "elements": [
                        {
                            "type": "rich_text_section",
                            "elements": [
                                {
                                    "type": "text",
                                    "text": error.msg,
                                }
                            ],
                        }
                    ],
                },
            ]
        )

    validation_blocks: list[dict[str, Any]] = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*{error_type}* ({len(errors)} total)",
            },
        },
        {"type": "table", "rows": table_rows},
    ]

    try:
        await slack_client.chat_postMessage(
            channel=channel,
            thread_ts=thread_ts,
            text=f"{error_type}: {len(errors)} issues found",
            blocks=validation_blocks,
        )
        logger.info(f"posted {error_type} thread")
    except Exception as e:
        logger.error(f"failed to post {error_type} thread: {e}")


@flow()
async def sync_classifiers_profiles(
    aws_env: AwsEnv,
    config: Config | None = None,
    wikibase_auth: WikibaseAuth | None = None,
    wikibase_cache_path: Path | None = None,
    wikibase_cache_save_if_missing: bool = False,
):
    """Update classifier profile for a given aws environment."""

    logger = get_logger()

    logger.info(f"Wikibase Cache Path: {wikibase_cache_path}")
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

    logger.info(f"Valid concepts retrieved from wikibase: {len(classifiers_profiles)}")
    logger.info(f"Validation errors from wikibase: {len(validation_errors)}")

    classifiers_profiles_to_update = compare_classifiers_profiles(
        classifier_specs, classifiers_profiles
    )
    logger.info(f"Identified {len(classifiers_profiles_to_update)} updates for syncing")

    wandb.login(key=config.wandb_api_key.get_secret_value())

    wandb_results: list[Result[Dict, Error]] = []
    UPDATE_HANDLERS = {
        "add": promote_classifier_profile,
        "remove": demote_classifier_profile,
        "update": update_classifier_profile,
    }

    for classifier_profile in classifiers_profiles_to_update:
        if status := classifier_profile.get("status"):
            if handler := UPDATE_HANDLERS.get(status):
                wandb_results.append(
                    handler(
                        current_specs=classifier_profile.get("current", None),
                        new_specs=classifier_profile.get("new", None),
                        aws_env=aws_env,
                    )
                )
            else:
                logger.info(f"Unhandled status: {status}")

    successes = [r._value for r in results if isinstance(r, Ok)]
    logger.info(
        f"Successfully synced {len(successes)} classifier profile updates to wandb"
    )

    # update classifiers specs yaml file
    refresh_all_available_classifiers([aws_env])

    # combine validation errors with wandb errors and successes
    final_results: list[Result[Dict, Error]] = validation_errors + wandb_results

    try:
        await send_classifiers_profile_slack_alert(
            validation_errors=[
                r._error for r in validation_errors if isinstance(r, Err)
            ],
            other_errors=[r._error for r in wandb_results if isinstance(r, Err)],
            successes=[r._value for r in final_results if isinstance(r, Ok)],
            aws_env=aws_env,
        )
    except Exception as e:
        logger.error(f"failed to send validation alert: {e}")

    # create artifact with summary
    await create_classifiers_profiles_artifact(
        results=final_results,
    )
