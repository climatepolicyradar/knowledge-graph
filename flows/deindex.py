from dataclasses import dataclass
import os

from cpr_sdk.search_adaptors import VespaSearchAdapter
from prefect import flow, get_run_logger
from flows.index import CONCEPTS_COUNTS_PREFIX_DEFAULT
from flows.inference import DOCUMENT_TARGET_PREFIX_DEFAULT
from flows.utils import SlackNotify
from scripts.cloud import AwsEnv, ClassifierSpec, get_prefect_job_variable


DEFAULT_DOCUMENTS_BATCH_SIZE = 500
DEFAULT_INDEXING_TASK_BATCH_SIZE = 20

@dataclass()
class Config:
    """Configuration used across flow runs."""

    cache_bucket: str | None = None
    document_source_prefix: str = DOCUMENT_TARGET_PREFIX_DEFAULT
    concepts_counts_prefix: str = CONCEPTS_COUNTS_PREFIX_DEFAULT
    bucket_region: str = "eu-west-1"
    # An instance of VespaSearchAdapter.
    #
    # E.g.
    #
    # VespaSearchAdapter(
    #   instance_url="https://vespa-instance-url.com",
    #   cert_directory="certs/"
    # )
    vespa_search_adapter: VespaSearchAdapter | None = None
    aws_env: AwsEnv = AwsEnv(os.environ["AWS_ENV"])
    as_deployment: bool = True

    @classmethod
    async def create(cls) -> "Config":
        """Create a new Config instance with initialized values."""
        logger = get_run_logger()

        config = cls()

        if not config.cache_bucket:
            logger.info(
                "no cache bucket provided, getting it from Prefect job variable"
            )
            config.cache_bucket = await get_prefect_job_variable(
                "pipeline_cache_bucket_name"
            )

        return config

@flow(
    # TODO: Enable once confident
    # on_failure=[SlackNotify.message],
    # on_crashed=[SlackNotify.message],
)
    async def deindex_labelled_passages_from_s3_to_vespa(
    classifier_specs: list[ClassifierSpec] | None = None,
    document_ids: list[str] | None = None,
    config: Config | None = None,
    batch_size: int = DEFAULT_DOCUMENTS_BATCH_SIZE,
    indexing_task_batch_size: int = DEFAULT_INDEXING_TASK_BATCH_SIZE,
) -> None:
    # TODO: Build list of inference results from S3
    # TODO: Delete concepts from those document passages in Vespa
    # TODO: Delete inference results from S3
    # TODO: Build concepts counts results from S3
    # TODO: Delete concepts counts from family documents in Vespa
    # TODO: Delete concept counts from S3
    return None
