"""
Flow that updates classifiers profiles changes detected in wikibase

Assumes that the classifier model has been trained in wandb
"""

import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import httpx
import wandb
from cpr_sdk.models.search import ClassifiersProfile as VespaClassifiersProfile
from cpr_sdk.models.search import ClassifiersProfiles as VespaClassifiersProfiles
from cpr_sdk.models.search import (
    ConceptV2DocumentFilter,
    SearchParameters,
)
from cpr_sdk.models.search import WikibaseId as VespaWikibaseId
from cpr_sdk.search_adaptors import VespaSearchAdapter
from prefect import flow
from prefect.artifacts import acreate_table_artifact
from prefect.context import FlowRunContext, get_run_context
from prefect.events import Event, emit_event
from prefect.settings import PREFECT_UI_URL
from pydantic import AnyHttpUrl, SecretStr
from vespa.application import VespaAsync
from vespa.io import VespaResponse

import flows.create_classifiers_specs_pr as create_classifiers_specs_pr
import scripts.classifier_metadata
import scripts.demote
import scripts.promote
from flows.boundary import get_vespa_search_adapter_from_aws_secrets
from flows.classifier_specs.spec_interface import (
    ClassifierSpec,
    determine_spec_file_path,
    load_classifier_specs,
)
from flows.result import Err, Error, Ok, Result, is_err, is_ok, unwrap_err, unwrap_ok
from flows.utils import (
    JsonDict,
    SlackNotify,
    get_logger,
    get_run_name,
    get_slack_client,
    total_milliseconds,
)
from knowledge_graph.classifier import ModelPath
from knowledge_graph.classifiers_profiles import (
    ClassifiersProfileMapping,
    Profile,
    validate_classifiers_profiles_mappings,
)
from knowledge_graph.cloud import AwsEnv, get_aws_ssm_param
from knowledge_graph.compare_result_operation import (
    CompareResultOperation,
    Demote,
    Promote,
    Update,
)
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import ClassifierID, Identifier, WikibaseID
from knowledge_graph.version import Version, get_latest_model_version
from knowledge_graph.wikibase import WikibaseAuth, WikibaseSession
from scripts.update_classifier_spec import refresh_all_available_classifiers

VESPA_MAX_TIMEOUT_MS: int = total_milliseconds(timedelta(minutes=5))
VESPA_CONNECTION_POOL_SIZE: int = 5

WIKIBASE_PASSWORD_SSM_NAME = "/Wikibase/Cloud/ServiceAccount/Password"
WIKIBASE_USERNAME_SSM_NAME = "/Wikibase/Cloud/ServiceAccount/Username"
WIKIBASE_URL_SSM_NAME = "/Wikibase/Cloud/URL"

SYNC_FINISHED_EVENT_NAME = "sync-classifiers_profiles.finished"
SYNC_RESOURCE_ID = "sync-classifiers-profiles"


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
        # using wandb_registry_version is directly accessing artifact and takes priority
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
                msg="Error artifact not found, check input parameters",
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
            msg=f"Error artifact validation failed: classifier name {artifact.metadata.get('classifier_name')} violates classifier name rules",
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
                    msg=f"Error artifact validation failed: run config {key} violates run config rules",
                    metadata={},
                )
    return Ok(artifact)


def handle_classifier_profile_action(
    action: str,
    wikibase_id: WikibaseID,
    aws_env: AwsEnv,
    action_function: Callable[..., Any],
    upload_to_wandb: bool,
    **kwargs,
) -> Result[Dict, Error]:
    """
    Run wandb validation and execute action function based on params

    The action function should be one of promote, demote, update classifier profile functions.
    Returns a Result with the action and parameters or an error.
    """
    logger = get_logger()

    wandb_registry_version = kwargs.get("wandb_registry_version", None)
    classifier_id = kwargs.get("classifier_id", None)

    result = wandb_validation(
        wikibase_id=wikibase_id,
        classifier_id=classifier_id,
        wandb_registry_version=wandb_registry_version,
        aws_env=aws_env,
    )
    if isinstance(result, Err):
        return result

    # Remove keys with None values from kwargs for action function
    kwargs = {k: v for k, v in kwargs.items() if v is not None}

    try:
        action_function(
            wikibase_id=wikibase_id,
            aws_env=aws_env,
            upload_to_wandb=upload_to_wandb,
            **kwargs,
        )

    except Exception as e:
        return log_and_return_error(
            logger,
            f"Error {action} classifier profile: {e}",
            {
                "wikibase_id": wikibase_id,
                "classifier_id": classifier_id,
                **kwargs,
            },
        )

    result = {
        "wikibase_id": wikibase_id,
        "classifier_id": classifier_id,
        **kwargs,
        "status": action,
    }
    return Ok(result)


def promote_classifier_profile(
    wikibase_id: WikibaseID,
    classifier_id: ClassifierID,
    classifiers_profile: Profile,
    aws_env: AwsEnv,
    upload_to_wandb: bool,
):
    """
    Promote a classifier and add classifiers profile

    This promotes a classifier that exists in wandb projects into
    wandb registry. It tags the artifact with the aws_env and
    sets the classifiers profile in the metadata.
    The projects and registry artifacts are linked,
    any changes applied to one artifact will be reflected in the other.
    When refreshed, tagged artifacts in wandb registry return in the classifiers specs file.
    """
    logger = get_logger()

    logger.info(
        f"Promoting {wikibase_id}, {classifier_id}, {classifiers_profile}, {aws_env}"
    )

    if not upload_to_wandb:
        logger.info("Dry run, not uploading to wandb.")
    else:
        scripts.promote.main(
            wikibase_id=wikibase_id,
            classifier_id=classifier_id,
            aws_env=aws_env,
            add_classifiers_profiles=[classifiers_profile.value],
        )


def demote_classifier_profile(
    wikibase_id: WikibaseID,
    aws_env: AwsEnv,
    wandb_registry_version: Version,
    upload_to_wandb: bool,
    classifier_id: Optional[ClassifierID] = None,
    classifiers_profile: Optional[Profile] = None,
):
    """
    Demote a classifier based on model registry and remove classifiers profile

    This removes the tag and the classifiers profile from the wandb registry artifact.
    When refreshed, the classifiers specs file will no longer include this classifier.
    The link between the project and wandb registry artifacts is retained.
    """
    logger = get_logger()

    logger.info(
        f"Demoting {wikibase_id}, {aws_env}, {classifier_id}, {wandb_registry_version}, {classifiers_profile}"
    )
    if not upload_to_wandb:
        logger.info("Dry run, not uploading to wandb.")
    else:
        scripts.demote.main(
            wikibase_id=wikibase_id,
            wandb_registry_version=wandb_registry_version,
            aws_env=aws_env,
        )


def update_classifier_profile(
    wikibase_id: WikibaseID,
    classifier_id: ClassifierID,
    add_classifiers_profiles: list[Profile],
    remove_classifiers_profiles: list[str],
    aws_env: AwsEnv,
    upload_to_wandb: bool,
):
    """
    Update classifiers profile for already promoted model

    This modifies the metadata of a classifier already in the wandb registry.
    It adds and optionally removes classifiers profiles as specified by the parameters.
    """
    logger = get_logger()

    logger.info(
        f"Updating {wikibase_id}, {aws_env}, {classifier_id}, {str(add_classifiers_profiles[0])}, {str(remove_classifiers_profiles[0])}"
    )

    if not upload_to_wandb:
        logger.info("Dry run, not uploading to wandb.")
    else:
        scripts.classifier_metadata.update(
            wikibase_id=wikibase_id,
            classifier_id=classifier_id,
            add_classifiers_profiles=[
                profile.value for profile in add_classifiers_profiles
            ],
            remove_classifiers_profiles=remove_classifiers_profiles,
            aws_env=aws_env,
            update_specs=False,
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
    wikibase_auth: WikibaseAuth, concepts: list[Concept], debug: bool = False
) -> list[Result[ClassifiersProfileMapping, Error]]:
    """
    Return valid classifiers profiles and different kids of validation errors.

    Validation errors can be invalid concepts and violated business constraints.
    Debug mode: when enabled helps identify classifiers in wikibase without any classifier IDs.
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
            # only apply this check in debug mode
            if len(concept_classifiers_profiles) == 0 and debug:
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


async def create_classifiers_profiles_artifact(
    validation_errors: list[Error],
    wandb_errors: list[Error],
    vespa_errors: list[Error],
    successes: list[Dict],
    aws_env: AwsEnv,
    pr_number: int | None,
):
    """Create an artifact with a summary of the classifiers profiles validation checks"""

    # vespa errors can be per vespa request or per concept and are excluded from total concepts count
    total_concepts = len(successes) + len(validation_errors) + len(wandb_errors)
    successful_concepts = len(successes)

    all_failures = validation_errors + wandb_errors + vespa_errors
    failed_concepts = len(all_failures)

    pr_details = ""
    if pr_number and pr_number > 0:
        pr_url = (
            f"https://github.com/climatepolicyradar/knowledge-graph/pull/{pr_number}"
        )
        pr_details = f"- **Classifiers Specs PR**: [#{pr_number}]({pr_url})\n"
    else:
        pr_details = "- **Classifiers Specs PR**: No PR created\n"

    overview_description = f"""# Classifiers Profiles Validation Summary
## Overview
- **Total concepts found**: {total_concepts}
- **Successful Wikibase IDs**: {successful_concepts}
- **Failed Wikibase IDs**: {failed_concepts}
- **WandB Errors**: {len(wandb_errors)}
- **Validation Errors**: {len(validation_errors)}
- **Vespa Errors**: {len(vespa_errors)}
- {pr_details}
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
        format_cp_details(error.metadata or {}, "✗", error.msg)
        for error in all_failures
    ]

    await acreate_table_artifact(
        key=f"classifiers-profiles-validation-{aws_env.value}",
        table=cp_details,
        description=overview_description,
    )


def compare_classifiers_profiles(
    classifier_specs: list[ClassifierSpec],
    classifiers_profile_mappings: list[ClassifiersProfileMapping],
) -> list[CompareResultOperation]:
    """
    Compare current classifiers specs to valid classifiers profile mappings from wikibase

    Classify action to take for each classifier based on wikibase_id, classifier_id.
    Actions: add, remove, update, ignore.
    """

    results: list[CompareResultOperation] = []

    # Iterate over classifier_specs and classifiers_profiles
    for spec in classifier_specs:
        # Check if the classifier spec exists in classifiers_profile_mappings
        # comparing wikibase_id and classifier_id
        # concept_id is not required
        # classifier_id should be sufficient for matching as it uses the concept parameters to generate the canonical ID
        if matching_classifier := next(
            (
                mapping
                for mapping in classifiers_profile_mappings
                if mapping.wikibase_id == spec.wikibase_id
                and mapping.classifier_id == spec.classifier_id
            ),
            None,
        ):
            # If the classifier exists in both, check whether classifiers profile matches (ignore) or not (update)
            if (
                spec.classifiers_profile
                != matching_classifier.classifiers_profile.value
            ):
                results.append(
                    Update(
                        classifier_spec=spec,
                        classifiers_profile_mapping=matching_classifier,
                    )
                )
            else:
                # Skipping for now
                # results.append(Ignore(classifier_spec=spec))
                pass  # ignores are not returned
        else:
            # If the classifier does not exist in classifiers_profile_mappings but is in classifier_specs, it should be removed (demote)
            results.append(Demote(classifier_spec=spec))

    # Check for mappings that are in classifiers_profile_mappings but not in classifier_specs (promote)
    for mapping in classifiers_profile_mappings:
        if not any(
            spec.wikibase_id == mapping.wikibase_id
            and spec.classifier_id == mapping.classifier_id
            for spec in classifier_specs
        ):
            results.append(Promote(classifiers_profile_mapping=mapping))

    return results


async def send_classifiers_profile_slack_alert(
    validation_errors: list[Error],
    wandb_errors: list[Error],
    vespa_errors: list[Error],
    successes: list[Dict],
    aws_env: AwsEnv,
    upload_to_wandb: bool,
    upload_to_vespa: bool,
    event: Result[Event | None, Error],
):
    """
    Send slack alert with failures from the classifiers profiles lifecycle sync.

    Posts a summary message to slack channel and thread with table of failures.
    """
    logger = get_logger()
    slack_client = await get_slack_client()

    # vespa_errors can be per vespa request or per concept and are excluded from total concepts count
    total_concepts = len(successes) + len(validation_errors) + len(wandb_errors)

    event_errors = [unwrap_err(event)] if is_err(event) else []

    other_errors = wandb_errors + vespa_errors + event_errors
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
                errors=len(validation_errors),
                upload_to_wandb=upload_to_wandb,
                upload_to_vespa=upload_to_vespa,
                error_type="Data Quality Issues",
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
                        # channel = "alerts-concept-store"
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
                errors=len(other_errors),
                upload_to_wandb=upload_to_wandb,
                upload_to_vespa=upload_to_vespa,
                error_type="System Errors",
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
                if wandb_errors:
                    await _post_errors_thread(
                        slack_client=slack_client,
                        channel="alerts-platform-staging",
                        thread_ts=thread_ts,
                        errors=wandb_errors,
                        error_type="WandB Errors",
                    )
                if vespa_errors:
                    await _post_errors_thread(
                        slack_client=slack_client,
                        channel="alerts-platform-staging",
                        thread_ts=thread_ts,
                        errors=vespa_errors,
                        error_type="Vespa Errors",
                    )
                if event_errors:
                    await _post_errors_thread(
                        slack_client=slack_client,
                        channel=channel,
                        thread_ts=thread_ts,
                        errors=event_errors,
                        error_type="Event Errors",
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
    errors: int,
    upload_to_wandb: bool,
    upload_to_vespa: bool,
    error_type: str,
):
    # Create summary blocks
    summary_blocks: list[dict[str, Any]] = [
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"<{error_type}>: {errors} of {total_concepts} classifiers profiles failed.",
            },
        },
    ]

    flow_run_name = get_run_name() or "unknown"

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

    failure_rate = (errors / total_concepts) * 100 if total_concepts > 0 else 0
    # Determine colour based on failure rate
    if failure_rate >= 50:
        color = "#e01e5a"  # Red
    else:
        color = "#ecb22e"  # Orange

    header = "Classifiers Profile Sync Summary:"
    if upload_to_wandb:
        header += " uploading to wandb"
    else:
        header += " (dry run, not uploading to wandb)"
    if upload_to_vespa:
        header += " uploading to vespa"
    else:
        header += " (dry run, not uploading to vespa)"

    return await slack_client.chat_postMessage(
        channel=channel,
        text=header,
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
            # not be set for vespa sync errors
            wikibase_id = "N/A"

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


def create_vespa_profile_mappings(
    classifier_specs: list[ClassifierSpec],
) -> list[VespaClassifiersProfile.Mapping]:
    """Create VespaClassifiersProfile.Mapping objects from classifier specs."""

    mappings = []
    errors = []
    for spec in classifier_specs:
        try:
            mappings.append(
                VespaClassifiersProfile.Mapping(
                    concept_id=str(spec.concept_id),
                    concept_wikibase_id=VespaWikibaseId(str(spec.wikibase_id)),
                    classifier_id=str(spec.classifier_id),
                )
            )

        except Exception as e:
            errors.append(f"Failed to create mapping for {spec.wikibase_id}: {e}")

    if errors:
        raise ValueError(f"Errors creating VespaClassifiersProfile.Mapping: {errors}")

    return mappings


def create_vespa_classifiers_profile(
    name: Profile,
    mappings: list[VespaClassifiersProfile.Mapping],
) -> VespaClassifiersProfile:
    """Create VespaClassifiersProfile object from mappings."""
    try:
        id = Identifier.generate(name.value, mappings)
        vespa_profile = VespaClassifiersProfile(
            id=str(id),
            name=name.value,
            mappings=mappings,
            multi=(name == Profile.RETIRED),
            response_raw={},
        )
        return vespa_profile

    except Exception as e:
        raise ValueError(
            f"Failed to create VespaClassifiersProfile for {name.value} with mappings {mappings}: {e}"
        )


async def update_vespa_with_classifiers_profiles(
    classifier_specs: list[ClassifierSpec],
    vespa_connection_pool: VespaAsync,
    upload_to_vespa: bool = True,
) -> list[Result[None, Error]]:
    """
    Update Vespa with the latest classifiers profiles from classifier specs

    Returns a list of Result indicating success or failure for syncing to vespa

    ClassifierSpec are used instead of ClassifiersProfileMapping to include
    all unchanged classifiers as well as those that have been promoted/demoted/updated.
    ClassifiersProfileMapping also doesn't include concept_id which is required for Vespa mappings.
    """
    logger = get_logger()
    results: list[Result[None, Error]] = []

    logger.info(
        f"Processing {len(classifier_specs)} classifier specs for Vespa update."
    )

    try:
        # create VespaClassifiersProfiles.Mappings from classifier specs split by classifier profiles: primary, experimental, retired
        vespa_classifiers_profile = {}
        for profile in [Profile.PRIMARY, Profile.EXPERIMENTAL, Profile.RETIRED]:
            profile_mappings = create_vespa_profile_mappings(
                [
                    spec
                    for spec in classifier_specs
                    if spec.classifiers_profile == profile.value
                ],
            )

            logger.info(
                f"Created VespaClassifiersProfile.Mapping object for {profile.value}, with {len(profile_mappings)} mappings"
            )

            # convert to VespaClassifiersProfile
            # skip creating VespaClassifiersProfile if no mappings
            if len(profile_mappings) == 0:
                logger.info(
                    f"No mappings for profile {profile.value}, skipping VespaClassifiersProfile creation."
                )
                continue

            vespa_classifiers_profile[profile.value] = create_vespa_classifiers_profile(
                profile, profile_mappings
            )

            logger.info(f"Created VespaClassifiersProfile object for {profile.value}")

        # create VespaClassifiersProfiles from VespaClassifiersProfile objects
        # create mappings dynamically using vespa_classifiers_profile
        mappings = {}
        for profile_name, profile in vespa_classifiers_profile.items():
            mappings[profile_name] = f"{profile.name}.{profile.id}"

        vespa_classifiers_profiles = VespaClassifiersProfiles(
            id="default",
            mappings=mappings,
            response_raw={},
        )

    except Exception as e:
        results.append(
            log_and_return_error(
                logger,
                msg=f"Error creating VespaClassifiersProfile(s) objects: {e}",
                metadata={},
            )
        )
        return results

    # sync to vespa
    if not upload_to_vespa:
        logger.info("Upload to Vespa is not enabled. Skipping upload step.")
        results.append(Ok(None))
    else:
        logger.info("Syncing classifiers profiles to Vespa...")
        # sync VespaClassifiersProfile to vespa
        for profile_name, profile in vespa_classifiers_profile.items():
            fields = JsonDict(profile.model_dump(mode="json", exclude={"response_raw"}))
            doc_id = f"{profile.name}.{profile.id}"

            response: VespaResponse = await vespa_connection_pool.update_data(
                schema="classifiers_profile",
                namespace="doc_search",
                data_id=doc_id,
                create=True,  # create document if it doesn't exist
                fields=fields,
            )

            if not response.is_successful():
                results.append(
                    log_and_return_error(
                        logger,
                        msg=f"Error syncing VespaClassifiersProfile {profile_name} to Vespa",
                        metadata={
                            "response": response.get_json(),
                            "classifiers_profile": profile_name,
                        },
                    )
                )
                return results

            logger.info(
                f"Synced VespaClassifiersProfile {profile_name} to Vespa with {len(profile.mappings)} mappings for doc id {doc_id}"
            )

        # sync VespaClassifiersProfiles to Vespa with default ID
        fields = JsonDict(
            vespa_classifiers_profiles.model_dump(
                mode="json", exclude={"response_raw", "id"}
            )
        )
        doc_id = "default"

        response: VespaResponse = await vespa_connection_pool.update_data(
            schema="classifiers_profiles",
            namespace="doc_search",
            data_id=doc_id,
            create=True,  # create document if it doesn't exist
            fields=fields,
        )

        if not response.is_successful():
            results.append(
                log_and_return_error(
                    logger,
                    msg="Error syncing VespaClassifiersProfiles to Vespa",
                    metadata={"response": response.get_json(), "fields": fields},
                )
            )
            return results

        logger.info(
            f"Synced VespaClassifiersProfiles to doc id {doc_id}, with mappings: {vespa_classifiers_profiles.mappings} and fields {fields}"
        )

        results.append(Ok(None))

    return results


def emit_finished(
    promotions: list[Promote],
    aws_env: AwsEnv,
) -> Result[Event | None, Error]:
    """Emit an event indicating the pipeline finished."""

    logger = get_logger()

    if not promotions:
        logger.info("no promotions, skipping emitting finished event")
        return Ok(None)

    event = SYNC_FINISHED_EVENT_NAME
    resource = {
        "prefect.resource.id": SYNC_RESOURCE_ID,
        "awsenv": aws_env,
    }
    payload = {
        # Not currently used by the trigger, but including just in
        # case it'll be helpful.
        "promotions": list(
            map(
                lambda p: p.model_dump(mode="json"),
                promotions,
            )
        )
    }

    try:
        if event := emit_event(
            event=event,
            resource=resource,
            payload=payload,
        ):
            logger.info(f"finished event emitted (`{event.id}`)")
            return Ok(event)
        else:
            return Err(
                Error(
                    msg="emitting event returned `None`, indicating it wasn't sent",
                    metadata={
                        "event": event,
                        "resource": resource,
                        "payload": payload,
                    },
                )
            )
    except Exception as e:
        return Err(
            Error(
                msg="failed to emit event",
                metadata={
                    "event": event,
                    "resource": resource,
                    "payload": payload,
                    "exception": str(e),
                },
            )
        )


def maybe_allow_retiring(
    op: Promote | Update,
    vespa_search_adapter: VespaSearchAdapter,
    wandb_results: list[Result[Dict, Error]],
) -> tuple[
    bool,
    list[Result[Dict, Error]],
]:
    """If the operation is for a retiring profile, check for some results in Vespa."""
    logger = get_logger()

    if op.classifiers_profile_mapping.classifiers_profile != Profile.RETIRED:
        return True, wandb_results

    match concept_present_in_vespa(
        wikibase_id=op.classifiers_profile_mapping.wikibase_id,
        classifier_id=op.classifiers_profile_mapping.classifier_id,
        vespa_search_adapter=vespa_search_adapter,
    ):
        case Ok(True):
            logger.info(
                f"{op.classifiers_profile_mapping.wikibase_id}, {op.classifiers_profile_mapping.classifier_id} has results in Vespa, and can be retired"
            )
            return True, wandb_results
        case Ok(False):
            logger.info(
                f"{op.classifiers_profile_mapping.wikibase_id}, {op.classifiers_profile_mapping.classifier_id} has no results in Vespa, and can't be retired"
            )
            wandb_results.append(
                Err(
                    Error(
                        msg="no results found in Vespa, so can't retire",
                        metadata=op.classifiers_profile_mapping.model_dump(mode="json"),
                    )
                )
            )
            return False, wandb_results
        case Err(e):
            logger.info(
                f"{op.classifiers_profile_mapping.wikibase_id}, {op.classifiers_profile_mapping.classifier_id} failed to be checked for in Vespa: {str(e)}"
            )
            e.msg = e.msg + ". Failed to check for results in Vespa, so can't retire"
            wandb_results.append(Err(e))
            return False, wandb_results

    wandb_results.append(
        Err(
            Error(
                msg="failed to check for concept being present in Vespa",
                metadata=op.classifiers_profile_mapping.model_dump(mode="json"),
            )
        )
    )
    return False, wandb_results


def concept_present_in_vespa(
    wikibase_id: WikibaseID,
    classifier_id: ClassifierID,
    vespa_search_adapter: VespaSearchAdapter,
) -> Result[bool, Error]:
    """Check if a Concept<>Classifier has some results in Vespa."""
    try:
        response = vespa_search_adapter.search(
            SearchParameters(
                concept_v2_document_filters=[
                    ConceptV2DocumentFilter(
                        concept_wikibase_id=wikibase_id,
                        classifier_id=classifier_id,
                    )
                ],
                documents_only=True,
                # Use the presence of 1 as proof that data is there
                limit=1,
            )
        )

        return Ok(len(response.results) > 0)
    except Exception as e:
        return Err(
            Error(
                msg="failed to search Vespa for results",
                metadata={
                    "concept_wikibase_id": wikibase_id,
                    "classifier_id": classifier_id,
                    "exception": str(e),
                },
            )
        )


@flow(on_failure=[SlackNotify.message], on_crashed=[SlackNotify.message])
async def sync_classifiers_profiles(
    wandb_api_key: SecretStr | None = None,
    wikibase_auth: WikibaseAuth | None = None,
    wikibase_cache_path: Path | None = None,
    wikibase_cache_save_if_missing: bool = False,
    vespa_search_adapter: VespaSearchAdapter | None = None,
    upload_to_wandb: bool = False,  # set to False for dry run by default
    upload_to_vespa: bool = True,
    automerge_classifier_specs_pr: bool = False,
    auto_train: bool = False,
    debug_wikibase_validation: bool = False,
):
    """Update classifier profile for a given AWS environment."""

    logger = get_logger()

    # Pull it from the environment, as is our approach in Prefect, and
    # thus elsewhere, since this is run Prefect-first.
    aws_env = AwsEnv(os.environ["AWS_ENV"])

    if not vespa_search_adapter:
        logger.info("no Vespa search adapter provided, getting one from AWS secrets")
        temp_dir = tempfile.TemporaryDirectory()
        vespa_search_adapter = get_vespa_search_adapter_from_aws_secrets(
            cert_dir=temp_dir.name,
            vespa_private_key_param_name="VESPA_PRIVATE_KEY_FULL_ACCESS",
            vespa_public_cert_param_name="VESPA_PUBLIC_CERT_FULL_ACCESS",
            aws_env=aws_env,
        )

    logger.info(f"Wikibase Cache Path: {wikibase_cache_path}")
    if not wandb_api_key:
        logger.info("no W&B API key provided, getting one from AWS secrets")
        wandb_api_key = SecretStr(
            get_aws_ssm_param(
                "WANDB_API_KEY",
                aws_env=aws_env,
            )
        )

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

    github_token = SecretStr(
        get_aws_ssm_param(
            "GITHUB_TOKEN",
            aws_env=aws_env,
        )
    )

    if not upload_to_wandb:
        logger.warning(
            f"upload_to_wandb is set to {upload_to_wandb}. Using dry run mode for wandb."
        )

    if not upload_to_vespa:
        logger.warning(
            f"upload_to_vespa is set to {upload_to_vespa}. Using dry run mode for vespa."
        )

    if not automerge_classifier_specs_pr:
        logger.warning(
            f"automerge_classifier_specs_pr is set to {automerge_classifier_specs_pr}. Classifier specs PRs will not be auto-merged."
        )

    classifier_specs = load_classifier_specs(aws_env)
    logger.info(
        f"Loaded {len(classifier_specs)} classifier specs for env {aws_env.name}"
    )

    concepts = await read_concepts(
        wikibase_auth, wikibase_cache_path, wikibase_cache_save_if_missing
    )

    # returns Result with valid classifiers profiles and validation errors
    results = await get_classifiers_profiles(
        wikibase_auth, concepts, debug=debug_wikibase_validation
    )

    # retrieve validation errors and valid classifiers profiles
    validation_errors: list[Error] = [
        unwrap_err(r) for r in results if isinstance(r, Err)
    ]
    classifiers_profiles_mappings: list[ClassifiersProfileMapping] = [
        unwrap_ok(r) for r in results if isinstance(r, Ok)
    ]

    logger.info(
        f"Valid concepts retrieved from wikibase: {len(classifiers_profiles_mappings)}"
    )
    logger.info(f"Validation errors from wikibase: {len(validation_errors)}")

    # compare current specs to valid classifiers profiles from wikibase to identify changes
    classifiers_to_update = compare_classifiers_profiles(
        classifier_specs, classifiers_profiles_mappings
    )
    logger.info(f"Identified {len(classifiers_to_update)} updates for syncing")

    wandb.login(key=wandb_api_key.get_secret_value())

    wandb_results: list[Result[Dict, Error]] = []

    promotions: list[Promote] = []

    for classifiers in classifiers_to_update:
        match classifiers:
            case Promote() as a:
                logger.info(
                    f"promote called for {a.classifiers_profile_mapping.wikibase_id}, {a.classifiers_profile_mapping.classifier_id}, {a.classifiers_profile_mapping.classifiers_profile}"
                )

                new_spec = a.classifiers_profile_mapping
                wandb_result = handle_classifier_profile_action(
                    action="promoting",
                    wikibase_id=new_spec.wikibase_id,
                    aws_env=aws_env,
                    action_function=promote_classifier_profile,
                    upload_to_wandb=upload_to_wandb,
                    classifier_id=new_spec.classifier_id,
                    classifiers_profile=[str(new_spec.classifiers_profile)],
                )
                wandb_results.append(wandb_result)
                if is_ok(wandb_result):
                    promotions.append(a)

            case Update() as u:
                allow_retiring_or_other_action, wandb_results = maybe_allow_retiring(
                    u,
                    vespa_search_adapter,
                    wandb_results,
                )
                if not allow_retiring_or_other_action:
                    continue

                logger.info(
                    f"update called for {u.classifier_spec.wikibase_id}, {u.classifier_spec.classifier_id}, {u.classifier_spec.classifiers_profile} to {u.classifiers_profile_mapping.classifiers_profile}"
                )

                current_spec = u.classifier_spec
                new_spec = u.classifiers_profile_mapping

                wandb_results.append(
                    handle_classifier_profile_action(
                        action="updating",
                        wikibase_id=current_spec.wikibase_id,
                        aws_env=aws_env,
                        action_function=update_classifier_profile,
                        upload_to_wandb=upload_to_wandb,
                        classifier_id=current_spec.classifier_id,
                        remove_classifiers_profiles=[current_spec.classifiers_profile],
                        add_classifiers_profiles=[new_spec.classifiers_profile],
                    )
                )

            case Demote() as r:
                logger.info(
                    f"demote called for {r.classifier_spec.wikibase_id}, {r.classifier_spec.classifier_id}, {r.classifier_spec.classifiers_profile}"
                )

                current_spec = r.classifier_spec

                wandb_results.append(
                    handle_classifier_profile_action(
                        action="demoting",
                        wikibase_id=current_spec.wikibase_id,
                        aws_env=aws_env,
                        action_function=demote_classifier_profile,
                        upload_to_wandb=upload_to_wandb,
                        wandb_registry_version=current_spec.wandb_registry_version,
                        classifier_id=current_spec.classifier_id,
                        classifiers_profile=current_spec.classifiers_profile,
                    )
                )

    successes = [unwrap_ok(r) for r in wandb_results if isinstance(r, Ok)]
    wandb_errors = [unwrap_err(r) for r in wandb_results if isinstance(r, Err)]
    logger.info(
        f"Total concepts processed: {len(wandb_results) + len(validation_errors)}"
    )
    logger.info(
        f"Successful updates: {len(successes)}, Validation errors: {len(validation_errors)}, Wandb errors: {len(wandb_errors)}"
    )

    # if there were changes to wandb
    vespa_results = []
    cs_pr_results = []
    if len(successes) > 0:
        logger.info(
            f"Changes made to wandb: {len(successes)}, updating Vespa with the latest classifiers profiles..."
        )
        # update classifiers specs yaml file
        refresh_all_available_classifiers([aws_env])

        # reload classifier specs to confirm updates
        updated_classifier_specs = load_classifier_specs(aws_env)

        # create PR with updated classifier specs
        spec_file = str(determine_spec_file_path(aws_env))
        run_context = get_run_context()
        flow_run_name = get_run_name() or "unknown"
        flow_run_url = "unknown"
        if isinstance(run_context, FlowRunContext) and run_context.flow_run:
            flow_run_url = (
                f"{PREFECT_UI_URL.value()}/flow-runs/flow-run/{run_context.flow_run.id}"
            )
        cs_pr_results = await create_classifiers_specs_pr.create_and_merge_pr(
            spec_file=spec_file,
            aws_env=aws_env,
            flow_run_name=flow_run_name,
            flow_run_url=flow_run_url,
            github_token=github_token,
            auto_merge=automerge_classifier_specs_pr,
        )

        async with vespa_search_adapter.client.asyncio(
            connections=VESPA_CONNECTION_POOL_SIZE,
            timeout=httpx.Timeout(VESPA_MAX_TIMEOUT_MS / 1_000),
        ) as vespa_connection_pool:
            vespa_results = await update_vespa_with_classifiers_profiles(
                updated_classifier_specs, vespa_connection_pool, upload_to_vespa
            )

    vespa_errors = [unwrap_err(r) for r in vespa_results if isinstance(r, Err)]

    # retrieve PR number if PR was created successfully, otherwise set to -1
    pr_number = unwrap_ok(cs_pr_results[0]) if isinstance(cs_pr_results[0], Ok) else -1
    pr_errors = [unwrap_err(r) for r in cs_pr_results if isinstance(r, Err)]

    # The default, assuming there were no Vespa successes
    event: Result[Event | None, Error] = Ok(None)
    if any(map(is_ok, vespa_results)) and auto_train:
        logger.info("found at least 1 Vespa success, emitting finished event")
        event = emit_finished(
            # This is a vague check, since all of the promotions may
            # have failed in updating Vespa to reflect them.
            promotions,
            aws_env,
        )

    try:
        await send_classifiers_profile_slack_alert(
            validation_errors=validation_errors,
            wandb_errors=wandb_errors,
            vespa_errors=vespa_errors,
            successes=successes,
            aws_env=aws_env,
            upload_to_wandb=upload_to_wandb,
            upload_to_vespa=upload_to_vespa,
            event=event,
        )
    except Exception as e:
        logger.error(f"failed to send validation alert: {e}")

    await create_classifiers_profiles_artifact(
        validation_errors=validation_errors,
        wandb_errors=wandb_errors,
        vespa_errors=vespa_errors,
        successes=successes,
        aws_env=aws_env,
        pr_number=pr_number,
    )

    if len(vespa_errors) > 0:
        raise Exception(
            f"Errors occurred while updating Vespa with classifiers profiles: {vespa_errors}"
        )
    # if classifiers specs PR errors, fail the flow
    if len(pr_errors) > 0:
        raise Exception(
            f"Errors occurred while creating classifiers specs PR: {pr_errors}"
        )
