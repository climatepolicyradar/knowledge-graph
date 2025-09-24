import os
from dataclasses import dataclass
from io import BytesIO

import aioboto3
from cpr_sdk.ssm import get_aws_ssm_param
from prefect import flow
from pydantic import SecretStr

from flows.utils import SlackNotify, file_name_from_path, get_logger
from knowledge_graph.cloud import AwsEnv
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.wikibase import WikibaseSession

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


async def upload_to_s3(config: Config, concept: Concept) -> None:
    """Upload an individual concept to S3"""
    filename = f"{concept.wikibase_id}.json"
    key = os.path.join(config.s3_prefix, filename)
    data = concept.model_dump_json().encode()

    session = aioboto3.Session(region_name=config.bucket_region)
    async with session.client("s3") as s3:
        _ = await s3.put_object(
            Bucket=config.get_cdn_bucket_name(),
            Key=key,
            Body=BytesIO(data),
            ContentType="application/json",
        )


async def delete_from_s3(config: Config, concept_id: str) -> None:
    """Delete an individual concept from S3"""
    filename = f"{concept_id}.json"
    key = os.path.join(config.s3_prefix, filename)

    session = aioboto3.Session(region_name=config.bucket_region)
    async with session.client("s3") as s3:
        _ = await s3.delete_object(
            Bucket=config.get_cdn_bucket_name(),
            Key=key,
        )


async def list_s3_concepts(config: Config) -> list[str]:
    """List all concepts in S3"""
    session = aioboto3.Session(region_name=config.bucket_region)
    async with session.client("s3") as s3:
        concept_paths: list[str] = []
        paginator = s3.get_paginator("list_objects_v2")
        async for page in paginator.paginate(
            Bucket=config.get_cdn_bucket_name(), Prefix=config.s3_prefix
        ):
            if "Contents" in page:
                concept_paths.extend([o["Key"] for o in page["Contents"]])  # pyright: ignore[reportTypedDictNotRequiredAccess]

        s3_concepts = [file_name_from_path(path) for path in concept_paths]
        return s3_concepts


async def delete_extra_concepts_from_s3(
    extras_in_s3: list[str], config: Config
) -> list[str]:
    """Delete concepts from S3 that no longer exist in Wikibase."""
    logger = get_logger()

    logger.info(
        f"Deleting {len(extras_in_s3)} extra concepts from S3 that are no longer in Wikibase"
    )
    failures: list[str] = []
    for i, concept_id in enumerate(extras_in_s3, start=1):
        if i % config.logging_interval == 0:
            logger.info(f"Deleting extra concept #{i}: {concept_id}")
        try:
            await delete_from_s3(config, concept_id)
        except Exception as e:
            logger.error(f"Failed to delete concept #{i}: {concept_id}, error: {e}")
    return failures


@flow(
    on_failure=[SlackNotify.message],  # pyright: ignore[reportUnknownMemberType]
    on_crashed=[SlackNotify.message],  # pyright: ignore[reportUnknownMemberType]
)
async def wikibase_to_s3(config: Config | None = None):
    logger = get_logger()

    if not config:
        config = Config.create()
    logger.info(f"running with config: {config}")

    wikibase = WikibaseSession(
        username=config.get_wikibase_username(),
        password=config.get_wikibase_password_secret_value(),
        url=config.get_wikibase_url(),
    )
    wikibase_ids = await wikibase.get_all_concept_ids_async()
    logger.info(f"Found {len(wikibase_ids)} concept IDs in Wikibase")

    failed_wikibase_ids_uploads: list[WikibaseID] = []
    start = 1
    for i, wikibase_id in enumerate(wikibase_ids, start=start):
        next_interval = min(i + config.logging_interval, len(wikibase_ids))
        if i == start or i % config.logging_interval == 0:
            logger.info(f"Uploading concepts #{i}..#{next_interval}")
        try:
            concept = await wikibase.get_concept_async(
                wikibase_id, include_recursive_subconcept_of=True
            )
            await upload_to_s3(config, concept)
        except Exception as e:
            logger.error(f"Failed to upload concept #{i}: {wikibase_id}, error: {e}")
            failed_wikibase_ids_uploads.append(wikibase_id)

    # Identify leftovers from prior runs
    s3_concepts = await list_s3_concepts(config)
    extras_in_s3 = [c for c in s3_concepts if c not in wikibase_ids]
    missing_from_s3 = [c for c in wikibase_ids if c not in s3_concepts]

    logger.info(
        f"{len(s3_concepts)} concepts in S3, {len(wikibase_ids)} in Wikibase."
        f"Extras: {extras_in_s3}, Missing from S3: {missing_from_s3}"
    )

    failed_extras_in_s3_deletions: list[str] = []
    if extras_in_s3:
        failed_extras_in_s3_deletions = await delete_extra_concepts_from_s3(
            extras_in_s3, config
        )

    # There's several different things that can go wrong. Instead of
    # exiting as soon as any of them happen, the pipeline continues on
    # and do as much as possible.
    #
    # This is possible as this pipeline is/should be idempotent.
    failures: list[str] = []

    # Fail for discrepancies to trigger alerts
    if missing_from_s3:
        failures.append(
            f"{len(missing_from_s3)} concepts where found in Wikibase but didn't make it to S3: {missing_from_s3}"
        )

    if failed_extras_in_s3_deletions:
        failures.append(
            f"Some extras weren't deleted from S3: {failed_extras_in_s3_deletions}"
        )

    if failed_wikibase_ids_uploads:
        failures.append(
            f"Some concepts weren't uploaded: {failed_wikibase_ids_uploads}"
        )

    # Join all the failures together and make it so they'll be
    # reported.
    if failures:
        raise ValueError(". ".join(failures))
