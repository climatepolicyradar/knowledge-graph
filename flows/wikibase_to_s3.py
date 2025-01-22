import os
from dataclasses import dataclass
from io import BytesIO
from typing import Optional

import boto3
from cpr_sdk.ssm import get_aws_ssm_param
from prefect import flow, get_run_logger
from pydantic import SecretStr

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
    logging_interval: int = 50

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


def get_concepts_from_wikibase(config: Config) -> list[Concept]:
    """Get concepts from Wikibase"""
    wikibase = WikibaseSession(
        username=config.get_wikibase_username(),
        password=config.get_wikibase_password_secret_value(),
        url=config.get_wikibase_url(),
    )
    concepts = wikibase.get_concepts()
    return concepts


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
    response = s3.list_objects_v2(
        Bucket=config.get_cdn_bucket_name(), Prefix=config.s3_prefix
    )
    concepts = [o["Key"] for o in response["Contents"]]
    return concepts


@flow
def wikibase_to_s3(config: Optional[Config] = None):
    logger = get_run_logger()
    if not config:
        config = Config.create()
    logger.info(f"running with config: {config}")

    concepts = get_concepts_from_wikibase(config)
    logger.info(f"Found {len(concepts)} concepts in Wikibase, uploading to s3")
    for i, concept in enumerate(concepts):
        if i % config.logging_interval == 0:
            logger.info(f"Uploading concept #{i}: {concept.wikibase_id}")
        try:
            upload_to_s3(config, concept)
        except Exception as e:
            logger.error(f"Failed to upload concept #{i}: {concept.wikibase_id}")
            raise e

    # Identify leftovers from prior runs
    s3_concepts = list_s3_concepts(config)
    wikibase_concepts = [concept.wikibase_id for concept in concepts]
    extras_in_s3 = [c for c in s3_concepts if c not in wikibase_concepts]

    logger.info(
        f"{len(concepts)} concepts in S3, {len(wikibase_concepts)} in Wikibase."
        f"Extras: {extras_in_s3}"
    )
