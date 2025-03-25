import os
from dataclasses import dataclass

from cpr_sdk.search_adaptors import VespaSearchAdapter
from prefect import flow
from prefect.logging import get_run_logger

from flows.boundary import (
    CONCEPTS_COUNTS_PREFIX_DEFAULT,
    Operation,
    s3_paths_or_s3_prefixes,
    updates_by_s3,
)
from flows.inference import DOCUMENT_TARGET_PREFIX_DEFAULT
from flows.utils import SlackNotify
from scripts.cloud import (
    AwsEnv,
    ClassifierSpec,
    get_prefect_job_variable,
)

DEFAULT_DOCUMENTS_BATCH_SIZE = 500
DEFAULT_DEINDEXING_TASK_BATCH_SIZE = 20


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
    on_failure=[SlackNotify.message],
    on_crashed=[SlackNotify.message],
)
async def deindex_labelled_passages_from_s3_to_vespa(
    classifier_specs: list[ClassifierSpec],
    document_ids: list[str] | None = None,
    config: Config | None = None,
    batch_size: int = DEFAULT_DOCUMENTS_BATCH_SIZE,
    deindexing_task_batch_size: int = DEFAULT_DEINDEXING_TASK_BATCH_SIZE,
) -> None:
    """
    Asynchronously de-index concepts from S3 into Vespa.

    This function retrieves inference results of concepts in documents
    from S3, "undoes" them in a Vespa instance, and deletes the
    appropriate objects from S3.

    The undoing is relative to the doing in the index pipeline. It's
    resilient to de-indexing per document failing, so that it can be
    retried.

    The name of each file in the specified S3 path is expected to
    represent the document's import ID.
    """
    logger = get_run_logger()

    if not config:
        logger.info("no config provided, creating one")

        config = await Config.create()
    else:
        logger.info("config provided")
    assert config.cache_bucket

    logger.info(f"running with config: {config}")

    logger.info(f"running with classifier specs: {classifier_specs}")

    s3_accessor = s3_paths_or_s3_prefixes(
        classifier_specs,
        document_ids,
        config.cache_bucket,
        config.document_source_prefix,
    )

    logger.info(f"s3_prefixes: {s3_accessor.prefixes}, s3_paths: {s3_accessor.paths}")

    await updates_by_s3(
        aws_env=config.aws_env,
        partial_update_flow=Operation.DEINDEX,
        s3_prefixes=s3_accessor.prefixes,
        s3_paths=s3_accessor.paths,
        batch_size=batch_size,
        updates_task_batch_size=deindexing_task_batch_size,
        as_deployment=config.as_deployment,
        cache_bucket=config.cache_bucket,
        concepts_counts_prefix=config.concepts_counts_prefix,
    )
