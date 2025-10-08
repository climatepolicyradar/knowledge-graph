import os
from pathlib import Path
from typing import Annotated, Optional

from cpr_sdk.ssm import get_aws_ssm_param
from pydantic import AfterValidator, BaseModel, Field, SecretStr

from knowledge_graph.cloud import AwsEnv, get_prefect_job_variable

# Constant, s3 prefix for the aggregated results
INFERENCE_RESULTS_PREFIX = "inference_results/"
INFERENCE_DOCUMENT_SOURCE_PREFIX_DEFAULT: str = "embeddings_input/"
INFERENCE_DOCUMENT_TARGET_PREFIX_DEFAULT: str = "labelled_passages/"
AGGREGATE_DOCUMENT_SOURCE_PREFIX_DEFAULT: str = "labelled_passages/"

# SSM
WIKIBASE_PASSWORD_SSM_NAME = "/Wikibase/Cloud/ServiceAccount/Password"
WIKIBASE_USERNAME_SSM_NAME = "/Wikibase/Cloud/ServiceAccount/Username"
WIKIBASE_URL_SSM_NAME = "/Wikibase/Cloud/URL"


def validate_s3_prefix(value: str) -> str:
    """
    Validate S3 prefix format.

    Without a trailing slash, it is too lax in pattern matching. E.g.
    `embeddings_input` would match `embeddings_input` and
    `embeddings_input_test`. We only want the former.
    """
    if value.startswith("/"):
        raise ValueError("S3 prefix should not start with '/'")
    if not value:
        raise ValueError("S3 prefix cannot be empty")
    if not value.endswith("/"):
        raise ValueError("S3 prefix must end with '/'")

    return value


S3Prefix = Annotated[str, AfterValidator(validate_s3_prefix)]


class Config(BaseModel):
    """Shared Configuration used across flow runs."""

    cache_bucket: str | None = Field(default=None, description="S3 bucket for caching")
    aggregate_document_source_prefix: S3Prefix = Field(
        default=AGGREGATE_DOCUMENT_SOURCE_PREFIX_DEFAULT,
        description="S3 prefix for source documents are read from",
    )
    aggregate_inference_results_prefix: S3Prefix = Field(
        default=INFERENCE_RESULTS_PREFIX,
        description="S3 prefix for aggregated inference results are written to",
    )
    inference_document_source_prefix: S3Prefix = Field(
        default=INFERENCE_DOCUMENT_SOURCE_PREFIX_DEFAULT,
        description="S3 prefix of documents read as source for inference",
    )

    inference_document_target_prefix: S3Prefix = Field(
        default=INFERENCE_DOCUMENT_TARGET_PREFIX_DEFAULT,
        description="S3 prefix for where inference targets are written to",
    )

    bucket_region: str = Field(
        default="eu-west-1", description="AWS region for S3 bucket"
    )
    aws_env: AwsEnv = Field(
        default_factory=lambda: AwsEnv(os.environ["AWS_ENV"]),
        description="AWS environment",
    )

    pipeline_state_prefix: S3Prefix = Field(
        default="input/",
        description="S3 prefix for where new & updated documents from ingestion are located",
    )

    local_classifier_dir: Path = Field(
        default=Path("data") / "processed" / "classifiers",
        description="path to classifiers",
    )

    wandb_model_org: str = Field(
        default="climatepolicyradar_UZODYJSN66HCQ",
        description="Weights & Biases organisation for CPR",
    )

    wandb_model_registry: str = Field(
        default="wandb-registry-model",
        description="Weights & Biases model registry for CPR",
    )

    wandb_entity: str = Field(
        default="climatepolicyradar",
        description="Weights & Biases entity login credentials",
    )

    wandb_api_key: Optional[SecretStr] = Field(
        default=None,
        description="Weights & Biases API Key",
    )

    wikibase_username: Optional[str] = Field(
        default=None, description="User for authenticating with a WikibaseSession"
    )

    wikibase_password: Optional[SecretStr] = Field(
        default=None, description="Password for authenticating with a WikibaseSession"
    )

    wikibase_url: Optional[str] = Field(
        default=None,
        description="The wikibase instance base url. Used to authenticate with a WikibaseSession",
    )

    s3_concurrency_limit: int = Field(
        default=50,
        description="Use to limit asynchronous s3 operations for an individual batch.",
    )

    @classmethod
    async def create(cls) -> "Config":
        """Create a new Config instance with initialized values."""
        config = cls()
        if not config.cache_bucket:
            config.cache_bucket = await get_prefect_job_variable(
                "pipeline_cache_bucket_name"
            )

        if not config.wandb_api_key:
            config.wandb_api_key = SecretStr(get_aws_ssm_param("WANDB_API_KEY"))

        if not config.wikibase_password:
            config.wikibase_password = SecretStr(
                get_aws_ssm_param(WIKIBASE_PASSWORD_SSM_NAME)
            )

        if not config.wikibase_username:
            config.wikibase_username = get_aws_ssm_param(WIKIBASE_USERNAME_SSM_NAME)

        if not config.wikibase_url:
            config.wikibase_url = get_aws_ssm_param(WIKIBASE_URL_SSM_NAME)

        return config

    @property
    def cache_bucket_str(self) -> str:
        """Return the cache bucket, raising an error if not set."""
        if not self.cache_bucket:
            raise ValueError(
                "Cache bucket is not set in config, consider calling the `create` method first."
            )
        return self.cache_bucket

    def to_json(self) -> dict:
        """Convert the config to a JSON serializable dictionary."""
        return {
            "cache_bucket": self.cache_bucket if self.cache_bucket else None,
            "aggregate_document_source_prefix": self.aggregate_document_source_prefix,
            "aggregate_inference_results_prefix": self.aggregate_inference_results_prefix,
            "inference_document_source_prefix": self.inference_document_source_prefix,
            "inference_document_target_prefix": self.inference_document_target_prefix,
            "pipeline_state_prefix": self.pipeline_state_prefix,
            "bucket_region": self.bucket_region,
            "local_classifier_dir": self.local_classifier_dir,
            "wandb_model_org": self.wandb_model_org,
            "wandb_model_registry": self.wandb_model_registry,
            "wandb_entity": self.wandb_entity,
            "wandb_api_key": (
                self.wandb_api_key.get_secret_value() if self.wandb_api_key else None
            ),
            "aws_env": self.aws_env,
            "s3_concurrency_limit": self.s3_concurrency_limit,
        }
