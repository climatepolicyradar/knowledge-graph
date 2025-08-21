import os
from pathlib import Path
from typing import Optional

from cpr_sdk.ssm import get_aws_ssm_param
from pydantic import BaseModel, Field, SecretStr

from scripts.cloud import AwsEnv, get_prefect_job_variable

# Constant, s3 prefix for the aggregated results
INFERENCE_RESULTS_PREFIX = "inference_results"
INFERENCE_DOCUMENT_SOURCE_PREFIX_DEFAULT: str = "embeddings_input"
INFERENCE_DOCUMENT_TARGET_PREFIX_DEFAULT: str = "labelled_passages"
AGGREGATE_DOCUMENT_SOURCE_PREFIX_DEFAULT: str = "labelled_passages"


class Config(BaseModel):
    """Shared Configuration used across flow runs."""

    cache_bucket: str | None = Field(default=None, description="s3 bucket for caching")
    aggregate_document_source_prefix: str = Field(
        default=AGGREGATE_DOCUMENT_SOURCE_PREFIX_DEFAULT,
        description="s3 prefix for source documents are read from",
    )
    aggregate_inference_results_prefix: str = Field(
        default=INFERENCE_RESULTS_PREFIX,
        description="s3 prefix for aggregated inference results are written to",
    )
    inference_document_source_prefix: str = Field(
        default=INFERENCE_DOCUMENT_SOURCE_PREFIX_DEFAULT,
        description="s3 prefix of documents read as source for inference",
    )

    inference_document_target_prefix: str = Field(
        default=INFERENCE_DOCUMENT_TARGET_PREFIX_DEFAULT,
        description="s3 prefix for where inference targets are written to",
    )

    bucket_region: str = Field(
        default="eu-west-1", description="AWS region for s3 bucket"
    )
    aws_env: AwsEnv = Field(
        default_factory=lambda: AwsEnv(os.environ["AWS_ENV"]),
        description="AWS environment",
    )

    pipeline_state_prefix: str = Field(default="input")

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

        return config

    @property
    def cache_bucket_str(self) -> str:
        """Return the cache bucket, raising an error if not set."""
        if not self.cache_bucket:
            raise ValueError(
                "Cache bucket is not set in config, consider calling the `create` method first."
            )
        return self.cache_bucket
