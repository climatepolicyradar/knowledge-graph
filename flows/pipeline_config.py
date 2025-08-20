import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from cpr_sdk.ssm import get_aws_ssm_param
from pydantic import BaseModel, Field, SecretStr

from scripts.cloud import AwsEnv, get_prefect_job_variable

# Constant, S3 prefix for the aggregated results
INFERENCE_RESULTS_PREFIX = "inference_results"
INFERENCE_DOCUMENT_SOURCE_PREFIX_DEFAULT: str = "embeddings_input"
INFERENCE_DOCUMENT_TARGET_PREFIX_DEFAULT: str = "labelled_passages"
AGGREGATE_DOCUMENT_SOURCE_PREFIX_DEFAULT: str = "labelled_passages"


class Config(BaseModel):
    """Shared Configuration used across flow runs."""

    cache_bucket: str | None = Field(default=None, description="S3 bucket for caching")
    aggregate_document_source_prefix: str = Field(
        default=AGGREGATE_DOCUMENT_SOURCE_PREFIX_DEFAULT,
        description="S3 prefix for source documents",
    )
    aggregate_inference_results_prefix: str = Field(
        default=INFERENCE_RESULTS_PREFIX,
        description="S3 prefix for aggregated inference results",
    )
    bucket_region: str = Field(
        default="eu-west-1", description="AWS region for S3 bucket"
    )
    aws_env: AwsEnv = Field(
        default_factory=lambda: AwsEnv(os.environ["AWS_ENV"]),
        description="AWS environment",
    )

    @classmethod
    async def create(cls) -> "Config":
        """Create a new Config instance with initialized values."""
        config = cls()
        if not config.cache_bucket:
            config.cache_bucket = await get_prefect_job_variable(
                "pipeline_cache_bucket_name"
            )
        return config

    @property
    def cache_bucket_str(self) -> str:
        """Return the cache bucket, raising an error if not set."""
        if not self.cache_bucket:
            raise ValueError(
                "Cache bucket is not set in config, consider calling the `create` method first."
            )
        return self.cache_bucket


@dataclass()
class InferenceConfig:
    """Inference Configuration used for inference flow runs."""

    cache_bucket: Optional[str] = None
    inference_document_source_prefix: str = INFERENCE_DOCUMENT_SOURCE_PREFIX_DEFAULT
    inference_document_target_prefix: str = INFERENCE_DOCUMENT_TARGET_PREFIX_DEFAULT
    pipeline_state_prefix: str = "input"
    bucket_region: str = "eu-west-1"
    local_classifier_dir: Path = Path("data") / "processed" / "classifiers"
    wandb_model_org: str = "climatepolicyradar_UZODYJSN66HCQ"
    wandb_model_registry: str = "wandb-registry-model"
    wandb_entity: str = "climatepolicyradar"
    wandb_api_key: Optional[SecretStr] = None
    aws_env: AwsEnv = AwsEnv(os.environ["AWS_ENV"])

    @classmethod
    async def create(cls) -> "InferenceConfig":
        """Create a new Config instance with initialized values."""
        config = cls()

        if not config.cache_bucket:
            config.cache_bucket = await get_prefect_job_variable(
                "pipeline_cache_bucket_name"
            )
        if not config.wandb_api_key:
            config.wandb_api_key = SecretStr(get_aws_ssm_param("WANDB_API_KEY"))

        return config

    def to_json(self) -> dict:
        """Convert the config to a JSON serializable dictionary."""
        return {
            "cache_bucket": self.cache_bucket if self.cache_bucket else None,
            "document_source_prefix": self.inference_document_source_prefix,
            "document_target_prefix": self.inference_document_target_prefix,
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
        }
