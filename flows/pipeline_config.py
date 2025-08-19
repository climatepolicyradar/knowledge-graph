import os

from pydantic import BaseModel, Field

from flows.inference import DOCUMENT_TARGET_PREFIX_DEFAULT
from scripts.cloud import AwsEnv, get_prefect_job_variable

# Constant, S3 prefix for the aggregated results
INFERENCE_RESULTS_PREFIX = "inference_results"


class AggregateConfig(BaseModel):
    """Shared Configuration used across flow runs."""

    cache_bucket: str | None = Field(default=None, description="S3 bucket for caching")
    document_source_prefix: str = Field(
        default=DOCUMENT_TARGET_PREFIX_DEFAULT,
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
    async def create(cls) -> "AggregateConfig":
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
