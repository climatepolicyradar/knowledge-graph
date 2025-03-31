import os
from dataclasses import dataclass
from io import BytesIO

import boto3
from cpr_sdk.ssm import get_aws_ssm_param
from prefect import flow, get_run_logger, task
from prefect.deployments.deployments import run_deployment
from pydantic import SecretStr

from flows.deindex import deindex_labelled_passages_from_s3_to_vespa
from flows.utils import SlackNotify, file_name_from_path
from scripts.cloud import AwsEnv, function_to_flow_name, generate_deployment_name
from scripts.update_classifier_spec import ClassifierSpec
from src.concept import Concept
from src.wikibase import WikibaseSession

CDN_BUCKET_NAME_SSM_NAME = "/S3/CDNBucketName"
WIKIBASE_PASSWORD_SSM_NAME = "/Wikibase/Cloud/ServiceAccount/Password"
WIKIBASE_USERNAME_SSM_NAME = "/Wikibase/Cloud/ServiceAccount/Username"
WIKIBASE_URL_SSM_NAME = "/Wikibase/Cloud/URL"


@dataclass()
class Config:
    """Configuration used across flow runs."""

    s3_prefix: str = "concepts"
    bucket_region: str = "eu-west-1"
    aws_env: AwsEnv = AwsEnv(os.environ["AWS_ENV"])
    cdn_bucket_name: str | None = None
    wikibase_password: SecretStr | None = None
    wikibase_username: str | None = None
    wikibase_url: str | None = None
    logging_interval: int = 200
    trigger_deindexing: bool = True

    @classmethod
    def create(cls) -> "Config":
        """Create a new Config instance with initialized values."""
        config = cls()
        if not config.cdn_bucket_name:
            config.cdn_bucket_name = get_aws_ssm_param(CDN_BUCKET_NAME_SSM_NAME)
        if not config.wikibase_password:
            config.wikibase_password = SecretStr(
                get_aws_ssm_param(WIKIBASE_PASSWORD_SSM_NAME)
            )
        if not config.wikibase_username:
            config.wikibase_username = get_aws_ssm_param(WIKIBASE_USERNAME_SSM_NAME)
        if not config.wikibase_url:
            config.wikibase_url = get_aws_ssm_param(WIKIBASE_URL_SSM_NAME)

        return config

    def get_cdn_bucket_name(self) -> str:
        """Type safe way of accessing the CDN bucket name"""
        if not self.cdn_bucket_name:
            raise ValueError("`cdn_bucket_name` not set")
        else:
            return self.cdn_bucket_name

    def get_wikibase_username(self) -> str:
        """Type safe way of accessing the wikibase_username"""
        if not self.wikibase_username:
            raise ValueError("`wikibase_username` not set")
        else:
            return self.wikibase_username

    def get_wikibase_url(self) -> str:
        """Type safe way of accessing the wikibase_url"""
        if not self.wikibase_url:
            raise ValueError("`wikibase_url` not set")
        else:
            return self.wikibase_url

    def get_wikibase_password_secret_value(self) -> str:
        """Type safe way of getting the wikibase password secret value"""
        if not self.wikibase_password:
            raise ValueError("`wikibase_password` not set")
        return self.wikibase_password.get_secret_value()


def upload_to_s3(config: Config, concept: Concept) -> None:
    """Upload an individual concept to S3"""
    filename = f"{concept.wikibase_id}.json"
    key = os.path.join(config.s3_prefix, filename)
    data = concept.model_dump_json().encode()

    s3 = boto3.client("s3", region_name=config.bucket_region)
    _ = s3.put_object(
        Bucket=config.get_cdn_bucket_name(),
        Key=key,
        Body=BytesIO(data),
        ContentType="application/json",
    )


def delete_from_s3(config: Config, concept_id: str) -> None:
    """Delete an individual concept from S3"""
    filename = f"{concept_id}.json"
    key = os.path.join(config.s3_prefix, filename)

    s3 = boto3.client("s3", region_name=config.bucket_region)
    _ = s3.delete_object(
        Bucket=config.get_cdn_bucket_name(),
        Key=key,
    )


def list_s3_concepts(config: Config) -> list[str]:
    """List all concepts in S3"""
    s3 = boto3.client("s3", region_name=config.bucket_region)

    concept_paths: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(
        Bucket=config.get_cdn_bucket_name(), Prefix=config.s3_prefix
    ):
        if "Contents" in page:
            concept_paths.extend([o["Key"] for o in page["Contents"]])

    s3_concepts = [file_name_from_path(path) for path in concept_paths]
    return s3_concepts


def delete_extra_concepts_from_s3(extras_in_s3: list[str], config: Config):
    """Delete concepts from S3 that no longer exist in Wikibase."""
    logger = get_run_logger()

    logger.info(
        f"Deleting {len(extras_in_s3)} extra concepts from S3 that are no longer in Wikibase"
    )
    for i, concept_id in enumerate(extras_in_s3):
        if i % config.logging_interval == 0:
            logger.info(f"Deleting extra concept #{i}: {concept_id}")
        try:
            delete_from_s3(config, concept_id)
        except Exception as e:
            logger.error(f"Failed to delete concept #{i}: {concept_id}, error: {e}")


@task
async def trigger_deindexing(extras_in_s3: list[str], config: Config):
    logger = get_run_logger()

    # Run deployment for de-indexing
    logger.info(f"Running de-indexing deployment for {len(extras_in_s3)} concepts")
    flow_name = function_to_flow_name(deindex_labelled_passages_from_s3_to_vespa)
    deployment_name = generate_deployment_name(
        flow_name=flow_name, aws_env=config.aws_env
    )

    # Convert WikibaseIDs to ClassifierSpecs
    classifier_specs = [
        # Convert it to a dict, so Prefect can serialise it
        dict(
            ClassifierSpec(
                name=concept_id,
                # If a concept has been removed, we'll wipe all versions
                # of results, so just use the 'latest' version alias here.
                alias="latest",
            )
        )
        for concept_id in extras_in_s3
    ]

    try:
        _ = await run_deployment(
            name=f"{flow_name}/{deployment_name}",
            parameters={
                "classifier_specs": classifier_specs,
            },
            timeout=0,  # Don't wait for it to finish
        )
        logger.info("Successfully triggered de-indexing deployment")
    except Exception as e:
        logger.error(f"Failed to trigger de-indexing deployment: {e}")


@flow(
    on_failure=[SlackNotify.message],
    on_crashed=[SlackNotify.message],
)
async def wikibase_to_s3(config: Config | None = None):
    logger = get_run_logger()

    if not config:
        config = Config.create()
    logger.info(f"running with config: {config}")

    wikibase = WikibaseSession(
        username=config.get_wikibase_username(),
        password=config.get_wikibase_password_secret_value(),
        url=config.get_wikibase_url(),
    )
    wikibase_ids = wikibase.get_all_concept_ids()
    logger.info(f"Found {len(wikibase_ids)} concept IDs in Wikibase")

    for i, wikibase_id in enumerate(wikibase_ids):
        if i % config.logging_interval == 0:
            logger.info(f"Uploading concept #{i}: {wikibase_id}")
        try:
            concept = wikibase.get_concept(
                wikibase_id, include_recursive_subconcept_of=True
            )
            upload_to_s3(config, concept)
        except Exception as e:
            logger.error(f"Failed to upload concept #{i}: {wikibase_id}, error: {e}")

    # Identify leftovers from prior runs
    s3_concepts = list_s3_concepts(config)
    extras_in_s3 = [c for c in s3_concepts if c not in wikibase_ids]
    missing_from_s3 = [c for c in wikibase_ids if c not in s3_concepts]

    logger.info(
        f"{len(s3_concepts)} concepts in S3, {len(wikibase_ids)} in Wikibase."
        f"Extras: {extras_in_s3}, Missing from S3: {missing_from_s3}"
    )

    if extras_in_s3:
        await trigger_deindexing(extras_in_s3, config)
        delete_extra_concepts_from_s3(extras_in_s3, config)

    # Fail for discrepancies to trigger alerts
    if missing_from_s3:
        raise ValueError(
            f"{len(missing_from_s3)} concepts where found in Wikibase but "
            f"didnt make it to S3: {missing_from_s3}"
        )
