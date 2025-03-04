import os
from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import boto3
from cpr_sdk.ssm import get_aws_ssm_param
from prefect import flow, get_run_logger
from pydantic import SecretStr

from flows.utils import SlackNotify, file_name_from_path
from scripts.cloud import AwsEnv
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
    cdn_bucket_name: Optional[str] = None
    wikibase_password: Optional[SecretStr] = None
    wikibase_username: Optional[str] = None
    wikibase_url: Optional[str] = None
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
        """Type safe way of accessing the cdn bucket name"""
        if not self.cdn_bucket_name:
            raise ValueError("cdn_bucket_name not set")
        else:
            return self.cdn_bucket_name

    def get_wikibase_username(self) -> str:
        """Type safe way of accessing the wikibase_username"""
        if not self.wikibase_username:
            raise ValueError("wikibase_username not set")
        else:
            return self.wikibase_username

    def get_wikibase_url(self) -> str:
        """Type safe way of accessing the wikibase_url"""
        if not self.wikibase_url:
            raise ValueError("wikibase_url not set")
        else:
            return self.wikibase_url

    def get_wikibase_password_secret_value(self) -> str:
        """Type safe way of getting the wikibase password secret value"""
        if not self.wikibase_password:
            raise ValueError("wikibase_password not set")
        return self.wikibase_password.get_secret_value()


def upload_to_s3(config: Config, concept: Concept) -> None:
    """Upload an individual concept to S3"""
    filename = f"{concept.wikibase_id}.json"
    key = os.path.join(config.s3_prefix, filename)
    data = concept.model_dump_json().encode()

    s3 = boto3.client("s3", region_name=config.bucket_region)
    s3.put_object(
        Bucket=config.get_cdn_bucket_name(),
        Key=key,
        Body=BytesIO(data),
        ContentType="application/json",
    )


def list_s3_concepts(config: Config) -> list[str]:
    """List all concepts in S3"""
    s3 = boto3.client("s3", region_name=config.bucket_region)

    concept_paths = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(
        Bucket=config.get_cdn_bucket_name(), Prefix=config.s3_prefix
    ):
        if "Contents" in page:
            concept_paths.extend([o["Key"] for o in page["Contents"]])

    s3_concepts = [file_name_from_path(path) for path in concept_paths]
    return s3_concepts


@flow(
    on_failure=[SlackNotify.message],
    on_crashed=[SlackNotify.message],
)
def wikibase_to_s3(config: Optional[Config] = None):
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
    logger.info(f"Found {len(wikibase_ids)} concept ids in Wikibase")

    for i, wikibase_id in enumerate(wikibase_ids):
        if i % config.logging_interval == 0:
            logger.info(f"Uploading concept #{i}: {wikibase_id}")
        try:
            concept = wikibase.get_concept(
                wikibase_id, include_recursive_parent_concepts=True
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

    # Fail for discrepencies to trigger alerts
    if extras_in_s3:
        raise ValueError(
            f"{len(extras_in_s3)} concepts where found in S3 but where "
            f"not part of the copy from wikibase: {extras_in_s3}"
        )
    if missing_from_s3:
        raise ValueError(
            f"{len(missing_from_s3)} concepts where found in wikibase but "
            f"didnt make it to S3: {missing_from_s3}"
        )
