import asyncio
import os
from collections import Counter
from dataclasses import dataclass

from cpr_sdk.search_adaptors import VespaSearchAdapter
from prefect import flow, get_run_logger, task

from flows.index import (
    CONCEPTS_COUNTS_PREFIX_DEFAULT,
    ConceptModel,
    DocumentImportId,
    get_local_vespa_search_adapter,
    iterate_batch,
    s3_obj_generator,
    s3_paths_or_s3_prefixes,
)
from scripts.cloud import AwsEnv, ClassifierSpec, get_prefect_job_variable
from scripts.update_classifier_spec import parse_spec_file

DEFAULT_BATCH_SIZE = 50


@dataclass()
class Config:
    """Configuration used across flow runs."""

    cache_bucket: str | None = None
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


@task
async def load_document() -> Counter[ConceptModel]:
    # TODO
    return Counter()


@task
async def update_family_document():
    pass


@flow
async def count_family_document_concepts(
    classifier_specs: list[ClassifierSpec] | None = None,
    document_ids: list[str] | None = None,
    config: Config | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    logger = get_run_logger()

    if not config:
        logger.info("no config provided, creating one")

        config = await Config.create()
    else:
        logger.info("config provided")

    logger.info(f"running with config: {config}")

    if classifier_specs is None:
        logger.info("no classifier specs. passed in, loading from file")
        classifier_specs = parse_spec_file(config.aws_env)

    logger.info(f"running with classifier specs.: {classifier_specs}")

    s3_paths, s3_prefixes = s3_paths_or_s3_prefixes(
        classifier_specs,
        document_ids,
        config.cache_bucket,  # pyright: ignore[reportArgumentType]
        config.concepts_counts_prefix,
    )

    logger.info(f"s3_prefix: {s3_prefixes}, s3_paths: {s3_paths}")

    cm, vespa_search_adapter = get_local_vespa_search_adapter(None)

    with cm:
        logger.info("getting S3 object generator")
        documents_generator = s3_obj_generator(s3_prefixes, s3_paths)

        documents_batches = iterate_batch(documents_generator, batch_size=batch_size)

        documents_uris: dict[str, list[str]] = {}

        # TODO: Start grouping
        concepts_counts: Counter[ConceptModel] = Counter()

        for (
            documents_batch_num,
            documents_batch,
        ) in enumerate(documents_batches, start=1):
            logger.info(f"processing batch documents #{documents_batch_num}")

            loading_tasks = [load_document() for document_importer in documents_batch]

            logger.info(
                f"gathering {batch_size} indexing tasks for batch {documents_batch_num}"
            )
            results = await asyncio.gather(*loading_tasks, return_exceptions=True)
            logger.info(
                f"gathered {batch_size} indexing tasks for batch {documents_batch_num}"
            )

            for i, result in enumerate(results):
                document_import_id: DocumentImportId = documents_batch[i][0]

                if isinstance(result, Exception):
                    logger.error(
                        f"failed to process document `{document_import_id}`: {str(result)}",
                    )
                    continue

                concepts_counts.update(result)

                logger.info(f"processed batch documents #{documents_batch_num}")

        # TODO: Partial update in Vespa
        update_family_document()

    return None
