import asyncio
import os
from dataclasses import dataclass
from datetime import timedelta

import boto3
from cpr_sdk.search_adaptors import VespaSearchAdapter
from prefect import flow
from prefect.client.schemas.objects import FlowRun, StateType
from prefect.deployments import run_deployment
from prefect.logging import get_run_logger

import scripts.update_classifier_spec
from flows.boundary import (
    CONCEPTS_COUNTS_PREFIX_DEFAULT,
    DocumentImporter,
    Operation,
    s3_obj_generator,
    s3_paths_or_s3_prefixes,
    update_s3_with_all_successes,
    updates_by_s3,
)
from flows.inference import DOCUMENT_TARGET_PREFIX_DEFAULT
from flows.utils import SlackNotify, iterate_batch
from scripts.cloud import (
    AwsEnv,
    ClassifierSpec,
    function_to_flow_name,
    generate_deployment_name,
    get_prefect_job_variable,
)
from src.identifiers import WikibaseID

DEFAULT_DOCUMENTS_BATCH_SIZE = 250
DEFAULT_DEINDEXING_TASK_BATCH_SIZE = 10

# The "parent" AKA the higher level flows that do multiple things
PARENT_TIMEOUT_S: int = int(timedelta(hours=2).total_seconds())
# A singular task doing one thing
TASK_TIMEOUT_S: int = int(timedelta(minutes=30).total_seconds())


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


def find_all_classifier_specs_for_latest(
    to_deindex: list[ClassifierSpec],
    being_maintained: list[ClassifierSpec],
    cache_bucket: str,
    document_source_prefix: str,
    s3_client,
) -> tuple[list[ClassifierSpec], list[ClassifierSpec]]:
    """
    For classifier spec. with a version of latest, find concrete versions.

    If this de-indexing pipeline was run for concepts that have been
    entirely removed from the concept store (through merging or
    deleting), then we want to delete all artifacts [1] for those
    concepts.

    Example:
    `Q200:latest` is passed, and in our artifacts we have versions
    `v3`, `v4`, and `v7`. `v7` is otherwise the latest AKA primary version.

    We'd

    [1] In S3 for labelled passages and concepts counts and the
    corollaries in Vespa.
    """
    with_primary: list[ClassifierSpec] = []
    for_cleanup: list[ClassifierSpec] = []

    seen_primaries: set[str] = set()

    maintained = {spec.name: spec.alias for spec in being_maintained}

    for classifier_spec in to_deindex:
        # Is it a valid spec.?
        if classifier_spec.name not in maintained:
            raise ValueError(
                f"classifier spec. {classifier_spec} was not found in the maintained list"
            )

        # Has it already been seen before?
        if classifier_spec.name in seen_primaries:
            raise ValueError(f"already have {classifier_spec} as a primary")

        # If it's not the `latest`, then we don't need to do look-ups
        if classifier_spec.alias != "latest":
            seen_primaries.add(classifier_spec.name)

            with_primary.append(classifier_spec)

            continue

        # Find all the aliases in S3 for `classifier_spec.name`
        aliases: list[str] = search_s3_for_aliases(
            WikibaseID(classifier_spec.name),
            cache_bucket=cache_bucket,
            document_source_prefix=document_source_prefix,
            s3_client=s3_client,
        )

        for alias in aliases:
            curren_classifier_spec = ClassifierSpec(
                name=classifier_spec.name, alias=alias
            )
            # Is it the primary, or just one for clean-up?
            if alias == maintained[classifier_spec.name]:
                # Rely on the seen check above, to prevent dupes
                with_primary.append(curren_classifier_spec)
            else:
                if classifier_spec in for_cleanup:
                    raise ValueError(f"already have {classifier_spec} as a clean-up")

                for_cleanup.append(curren_classifier_spec)

    return with_primary, for_cleanup


def search_s3_for_aliases(
    concept: WikibaseID,
    cache_bucket: str,
    document_source_prefix: str,
    s3_client,
) -> list[str]:
    """Find all aliases for a concept that are in our artifacts."""
    aliases: set[str] = set()

    # Ensure trailing slash for accurate prefix matching
    prefix = str(os.path.join(document_source_prefix, concept)) + "/"

    paginator = s3_client.get_paginator("list_objects_v2")
    # Remove Delimiter="/" to list all objects under the prefix
    pages = paginator.paginate(Bucket=cache_bucket, Prefix=prefix)

    # Iterate through Contents (objects) instead of CommonPrefixes
    for page in pages:
        for obj in page.get("Contents", []):
            key = obj.get("Key")

            # Extract the alias from the key like 'prefix/Q123/v1/doc.json'
            key_parts: list[str] = key.strip("/").split("/")

            # Expecting prefix_part/concept/alias/filename.json
            # e.g., ['labelled_passages', 'Q100', 'v1', 'doc.json']
            expected_length = 4
            if len(key_parts) != expected_length:
                raise ValueError(
                    f"alias parts length was {len(key_parts)} and not {expected_length}: {key_parts}"
                )

            alias: str = key_parts[2]

            if alias == "latest":
                continue

            aliases.add(alias)

    # Surely if we've passed a concept to this function, all the way
    # down from the pipeline parameters, we'd expect some results.
    if len(aliases) == 0:
        raise ValueError(f"found 0 aliases for concept {concept}")

    return list(aliases)


@flow(timeout_seconds=TASK_TIMEOUT_S)
async def run_cleanup_objects_for_batch(
    documents_batch: list[DocumentImporter],
    documents_batch_num: int,
    cache_bucket: str,
    concepts_counts_prefix: str,
):
    """Run clean-up objects for a batch of documents."""
    logger = get_run_logger()

    logger.info(
        f"Cleaning up objects for batch of documents, documents in batch: {len(documents_batch)}."
    )

    for document_importer in documents_batch:
        update_s3_with_all_successes(
            document_object_uri=document_importer[1],
            cache_bucket=cache_bucket,
            concepts_counts_prefix=concepts_counts_prefix,
        )

    logger.info(f"processed batch documents #{documents_batch_num}")


async def cleanup_objects_for_batch_flow_or_deployment(
    documents_batch: list[DocumentImporter],
    documents_batch_num: int,
    cache_bucket: str,
    concepts_counts_prefix: str,
    aws_env: AwsEnv,
    as_deployment: bool,
) -> None | FlowRun:
    """Run clean-up objects for a batch of documents as a sub-flow or deployment."""
    logger = get_run_logger()
    logger.info(
        "Running clean-up of document for batch as sub-flow or deployment: "
        f"batch length {len(documents_batch)}, as_deployment: {as_deployment}"
    )

    if as_deployment:
        flow_name = function_to_flow_name(run_cleanup_objects_for_batch)
        deployment_name = generate_deployment_name(flow_name=flow_name, aws_env=aws_env)

        return await run_deployment(
            name=f"{flow_name}/{deployment_name}",
            parameters={
                "documents_batch": documents_batch,
                "documents_batch_num": documents_batch_num,
                "cache_bucket": cache_bucket,
                "concepts_counts_prefix": concepts_counts_prefix,
            },
            # Rely on the flow's own timeout
            timeout=None,
        )

    return await run_cleanup_objects_for_batch(
        documents_batch=documents_batch,
        documents_batch_num=documents_batch_num,
        cache_bucket=cache_bucket,
        concepts_counts_prefix=concepts_counts_prefix,
    )


@flow(timeout_seconds=PARENT_TIMEOUT_S)
async def cleanups_by_s3(
    batch_size: int,
    cleanups_task_batch_size: int,
    aws_env: AwsEnv,
    cache_bucket: str,
    concepts_counts_prefix: str,
    as_deployment: bool,
    s3_prefixes: list[str] | None = None,
    s3_paths: list[str] | None = None,
) -> None:
    logger = get_run_logger()

    logger.info("Getting S3 object generator")
    documents_generator = s3_obj_generator(s3_prefixes, s3_paths)
    documents_batches = iterate_batch(documents_generator, batch_size=batch_size)
    cleanups_task_batches = iterate_batch(
        data=documents_batches, batch_size=cleanups_task_batch_size
    )

    batches_with_failures: set[int] = set()

    for i, cleanups_task_batch in enumerate(cleanups_task_batches, start=1):
        logger.info(f"Processing clean-ups task batch #{i}")

        cleanups_tasks = [
            cleanup_objects_for_batch_flow_or_deployment(
                documents_batch=documents_batch,
                documents_batch_num=documents_batch_num,
                cache_bucket=cache_bucket,
                concepts_counts_prefix=concepts_counts_prefix,
                aws_env=aws_env,
                as_deployment=as_deployment,
            )
            for documents_batch_num, documents_batch in enumerate(
                cleanups_task_batch, start=1
            )
        ]

        logger.info(f"Gathering cleanups tasks for batch #{i}")
        results: list[None | BaseException | FlowRun] = await asyncio.gather(
            *cleanups_tasks, return_exceptions=True
        )
        logger.info(f"Gathered cleanups tasks for batch #{i}")

        for result in results:
            if isinstance(result, Exception):
                batches_with_failures.add(i)
                logger.error(
                    f"result was an exception. Message: `{result}`",
                )
                break

            if not as_deployment:
                if result is None:
                    continue
            else:
                # Explicitly narrow the type to FlowRun
                if not isinstance(result, FlowRun):
                    batches_with_failures.add(i)
                    logger.error(
                        f"result was not one of the expected types: {type(result)}"
                    )
                    continue

                # At this point, result is definitely a FlowRun
                flow_run: FlowRun = result

                if flow_run.state is None:
                    batches_with_failures.add(i)
                    logger.error(
                        f"flow run result's state was unexpectedly missing. Flow run name: `{flow_run.name}`",
                    )
                    continue
                else:
                    if flow_run.state.type != StateType.COMPLETED:
                        batches_with_failures.add(i)
                        logger.error(
                            f"flow run result's state was not completed. Flow run name: `{flow_run.name}`",
                        )

    if len(batches_with_failures) > 0:
        failed_batches_str = ", ".join(str(b) for b in batches_with_failures)
        raise ValueError(
            f"Failed to process document batch in clean-ups task batch. Batches with failures: #{failed_batches_str}"
        )


@flow(
    on_failure=[SlackNotify.message],
    on_crashed=[SlackNotify.message],
    timeout_seconds=PARENT_TIMEOUT_S,
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

    s3_client = boto3.client("s3")

    classifier_specs_primaries, classifier_specs_cleanups = (
        find_all_classifier_specs_for_latest(
            to_deindex=classifier_specs,
            being_maintained=scripts.update_classifier_spec.parse_spec_file(
                aws_env=config.aws_env
            ),
            cache_bucket=config.cache_bucket,
            document_source_prefix=config.document_source_prefix,
            s3_client=s3_client,
        )
    )

    logger.info(
        "running with classifier specs, "
        f"primaries: {classifier_specs_primaries}, "
        f"clean-ups: {classifier_specs_cleanups}"
    )

    s3_accessor_cleanups = s3_paths_or_s3_prefixes(
        classifier_specs=classifier_specs_cleanups,
        document_ids=document_ids,
        cache_bucket=config.cache_bucket,
        prefix=config.document_source_prefix,
    )

    logger.info(
        f"clean-ups s3_prefixes: {s3_accessor_cleanups.prefixes}, s3_paths: {s3_accessor_cleanups.paths}"
    )

    await cleanups_by_s3(
        aws_env=config.aws_env,
        s3_prefixes=s3_accessor_cleanups.prefixes,
        s3_paths=s3_accessor_cleanups.paths,
        cache_bucket=config.cache_bucket,
        batch_size=batch_size,
        cleanups_task_batch_size=deindexing_task_batch_size,
        concepts_counts_prefix=config.concepts_counts_prefix,
        as_deployment=config.as_deployment,
    )

    s3_accessor_primaries = s3_paths_or_s3_prefixes(
        classifier_specs=classifier_specs_primaries,
        document_ids=document_ids,
        cache_bucket=config.cache_bucket,
        prefix=config.document_source_prefix,
    )

    logger.info(
        f"updates s3_prefixes: {s3_accessor_primaries.prefixes}, s3_paths: {s3_accessor_primaries.paths}"
    )

    await updates_by_s3(
        aws_env=config.aws_env,
        partial_update_flow=Operation.DEINDEX,
        s3_prefixes=s3_accessor_primaries.prefixes,
        s3_paths=s3_accessor_primaries.paths,
        batch_size=batch_size,
        updates_task_batch_size=deindexing_task_batch_size,
        as_deployment=config.as_deployment,
        cache_bucket=config.cache_bucket,
        concepts_counts_prefix=config.concepts_counts_prefix,
    )
