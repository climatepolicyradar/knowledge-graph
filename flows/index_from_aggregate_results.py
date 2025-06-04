import asyncio
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from typing import Any, Final

import boto3
import httpx
from cpr_sdk.models.search import Passage as VespaPassage
from prefect import flow
from prefect.artifacts import create_markdown_artifact, create_table_artifact
from prefect.client.schemas.objects import FlowRun
from prefect.deployments import run_deployment
from prefect.logging import get_run_logger
from pydantic import PositiveInt
from vespa.application import VespaAsync
from vespa.io import VespaResponse

from flows.aggregate_inference_results import (
    Config,
    RunOutputIdentifier,
    SerialisedVespaConcept,
)
from flows.boundary import (
    CONCEPT_COUNT_SEPARATOR,
    DEFAULT_DOCUMENTS_BATCH_SIZE,
    VESPA_MAX_TIMEOUT_MS,
    DocumentImportId,
    DocumentStem,
    TextBlockId,
    VespaDataId,
    VespaHitId,
    function_to_flow_name,
    generate_deployment_name,
    get_data_id_from_vespa_hit_id,
    get_document_passages_from_vespa__generator,
    get_vespa_search_adapter_from_aws_secrets,
)
from flows.result import Err, Error, Ok, Result
from flows.utils import (
    AsyncProfiler,
    Profiler,
    SlackNotify,
    collect_unique_file_stems_under_prefix,
    iterate_batch,
    remove_translated_suffix,
    wait_for_semaphore,
)
from scripts.cloud import AwsEnv

# How many connections to Vespa to use for indexing.
DEFAULT_VESPA_MAX_CONNECTIONS_AGG_INDEXER: Final[PositiveInt] = 50
# How many indexer deployments to run concurrently.
DEFAULT_INDEXER_CONCURRENCY_LIMIT: Final[PositiveInt] = 10


def load_json_data_from_s3(bucket: str, key: str) -> dict[str, Any]:
    """Load JSON data from an S3 URI."""

    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket, Key=key)
    body = response["Body"].read().decode("utf-8")
    return json.loads(body)


async def _update_vespa_passage_concepts(
    vespa_data_id: VespaDataId,
    serialised_concepts: list[dict[str, Any]],
    vespa_connection_pool: VespaAsync,
) -> VespaResponse:
    """Update a passage in Vespa with the given concepts."""

    response: VespaResponse = await vespa_connection_pool.update_data(
        schema="document_passage",
        namespace="doc_search",
        data_id=vespa_data_id,
        # Don't create an empty document for non-existent documents
        create=False,
        fields={"concepts": serialised_concepts},
    )

    return response


async def create_aggregate_indexing_summary_artifact(
    config: Config,
    document_stems: list[DocumentStem],
    successes: list[FlowRun | BaseException],
    failures: list[FlowRun | BaseException],
) -> None:
    """Create an artifact with summary information about the indexing run."""

    # Prepare summary data for the artifact
    total_documents = len(document_stems)
    successful_document_batches_count = len(successes)
    failed_document_batches_count = len(failures)

    # Format the overview information as a string for the description
    indexing_report = f"""# Aggregate Indexing Summary

## Overview
- **Environment**: {config.aws_env.value}
- **Total documents processed**: {total_documents}
- **Successful Batches**: {successful_document_batches_count}
- **Failed Batches**: {failed_document_batches_count}
"""

    # Create classifier details table
    create_markdown_artifact(
        key="Aggregate Indexing Summary",
        description="Summary of the passages indexing run to update concept counts.",
        markdown=indexing_report,
    )


@dataclass(frozen=True)
class SimpleConcept:
    """
    A simple, hashable concept.

    As of 2025-06-03, the Concept from the cpr_sdk isn't hashable.
    """

    id: str
    name: str


@flow(
    log_prints=True,
    retries=2,
    retry_delay_seconds=5,
)
async def index_document_passages(
    config: Config,
    run_output_identifier: RunOutputIdentifier,
    document_stem: DocumentStem,
    vespa_connection_pool: VespaAsync,
) -> list[Result[list[SimpleConcept], Error]]:
    """Index aggregated inference results from S3 into Vespa document passages."""

    aggregated_results_key = os.path.join(
        config.aggregate_inference_results_prefix,
        run_output_identifier,
        f"{document_stem}.json",
    )

    print(f"Loading aggregated inference results from S3: {aggregated_results_key}")

    raw_data = load_json_data_from_s3(
        bucket=config.cache_bucket_str, key=aggregated_results_key
    )
    aggregated_inference_results: dict[TextBlockId, SerialisedVespaConcept] = {
        TextBlockId(k): v for k, v in raw_data.items()
    }

    document_id: DocumentImportId = remove_translated_suffix(document_stem)
    print(f"Querying Vespa for passages related to document import ID: {document_id}")

    passages_generator = get_document_passages_from_vespa__generator(
        document_import_id=document_id,
        vespa_connection_pool=vespa_connection_pool,
    )

    passages_in_vespa: dict[TextBlockId, tuple[VespaHitId, VespaPassage]] = {}
    async for passage_batch in passages_generator:
        passages_in_vespa.update(passage_batch)

    print(
        f"Updating concepts for document import ID: {document_id}, "
        f"found {len(passages_in_vespa)} passages in Vespa",
    )

    results: list[Result[list[SimpleConcept], Error]] = []

    for text_block_id, serialised_concepts in aggregated_inference_results.items():
        if TextBlockId(text_block_id) not in list(passages_in_vespa.keys()):
            error = Error(
                msg="text block not found in Vespa",
                metadata={"text_block_id": TextBlockId(text_block_id)},
            )
            results.append(Err(error))
            continue

        vespa_hit_id: VespaHitId = passages_in_vespa[TextBlockId(text_block_id)][0]
        vespa_data_id: VespaDataId = get_data_id_from_vespa_hit_id(vespa_hit_id)

        response = await _update_vespa_passage_concepts(
            vespa_data_id=vespa_data_id,
            serialised_concepts=serialised_concepts,
            vespa_connection_pool=vespa_connection_pool,
        )

        if not response.is_successful():
            error = Error(
                msg="Vespa update failed",
                metadata={
                    "text_block_id": TextBlockId(text_block_id),
                    "json": response.get_json(),
                },
            )
            results.append(Err(error))
            continue

        results.append(
            Ok(
                [
                    SimpleConcept(id=concept["id"], name=concept["name"])
                    for concept in serialised_concepts
                ]
            )
        )

    return results


@flow(
    log_prints=True,
    retries=2,
    retry_delay_seconds=5,
)
async def index_family_document(
    document_id: DocumentImportId,
    vespa_connection_pool: VespaAsync,
    simple_concepts: list[SimpleConcept],
) -> Result[None, Error]:
    """Index document concept counts in Vespa via partial update."""
    concepts_counts: Counter[SimpleConcept] = Counter(simple_concepts)

    concepts_counts_with_names = {
        f"{concept.id}{CONCEPT_COUNT_SEPARATOR}{concept.name}": count
        for concept, count in concepts_counts.items()
    }

    print(f"serialised concepts counts: {concepts_counts_with_names}")

    response: VespaResponse = await vespa_connection_pool.update_data(
        schema="family_document",
        namespace="doc_search",
        data_id=document_id,
        # Don't create an empty document for non-existent documents
        create=False,
        fields={
            "concept_counts": concepts_counts_with_names
        },  # Note the schema is misnamed in Vespa
    )

    if not response.is_successful():
        return Err(
            Error(
                msg="Vespa update failed",
                metadata={"json": response.get_json()},
            )
        )

    return Ok(None)


async def create_indexing_summary_artifact(
    config: Config,
    run_output_identifier: RunOutputIdentifier,
    documents_stems: list[DocumentStem],
    errors_per_document: dict[DocumentStem, list[Error]],
) -> None:
    """Create a markdown report artifact with summary information about the indexing run."""

    # Prepare summary data for the artifact
    total_documents = len(documents_stems)
    failed_documents = len(errors_per_document)
    successful_documents = total_documents - failed_documents
    total_errors = sum(len(errors) for errors in errors_per_document.values())

    # Format the overview information as a string for the description
    overview_description = f"""# Indexing from Aggregate Results Summary

## Overview
- **Run Output Identifier**: {run_output_identifier}
- **Environment**: {config.aws_env.value}
- **Total documents processed**: {total_documents}
- **Successful documents**: {successful_documents}
- **Failed documents**: {failed_documents}
- **Total errors**: {total_errors}
"""

    # Create document details table
    document_details = []
    for document_id in documents_stems:
        errors = errors_per_document.get(document_id, [])
        status = "✗" if errors else "✓"
        error_messages = (
            "; ".join([str(error.msg) for error in errors]) if errors else "N/A"
        )
        document_details.append(
            {
                "Family document ID": document_id,
                "Status": status,
                "Errors": error_messages,
            }
        )

    # Create a single artifact with overview in description and document details in table
    await create_table_artifact(
        key=f"indexing-aggregate-results-{config.aws_env.value}",
        table=document_details,
        description=overview_description,
    )


@flow(log_prints=True)
async def index_all(
    config: Config,
    run_output_identifier: RunOutputIdentifier,
    document_stem: DocumentStem,
    vespa_connection_pool: VespaAsync,
) -> Result[None, list[Error]]:
    """Indexes all (document passages and family documents) data."""
    results = await index_document_passages(
        config=config,
        run_output_identifier=run_output_identifier,
        document_stem=document_stem,
        vespa_connection_pool=vespa_connection_pool,
    )

    print("finished indexing document passages")

    simple_concepts: list[SimpleConcept] = []
    errors: list[Error] = []
    for result in results:
        match result:
            case Ok(val):
                simple_concepts.extend(val)
            case Err(err):
                print(err)
                errors.append(err)

    print(f"simple counts n: {len(simple_concepts)}")

    document_id: DocumentImportId = remove_translated_suffix(document_stem)
    result = await index_family_document(
        document_id=document_id,
        vespa_connection_pool=vespa_connection_pool,
        simple_concepts=simple_concepts,
    )

    match result:
        case Ok(_):
            print(f"indexing family document {document_stem} succeeded")
        case Err(err):
            errors.append(err)
            print(f"indexing family document {document_stem} failed: {err}")

    if errors:
        return Err(errors)
    else:
        return Ok(None)


@flow(log_prints=True)
async def index_aggregate_results_for_batch_of_documents(
    run_output_identifier: RunOutputIdentifier,
    document_stems: list[DocumentStem],
    config_json: dict[str, Any],
    indexer_max_vespa_connections: PositiveInt = (
        DEFAULT_VESPA_MAX_CONNECTIONS_AGG_INDEXER
    ),
) -> None:
    """Index aggregated inference results into Vespa for family documents and document passages."""

    logger = get_run_logger()

    if not document_stems:
        raise NotImplementedError(
            "No document stems provided. This flow is not designed to run without them."
        )

    # This doesn't correctly parse the values into the dataclass.
    config = Config(**config_json)
    config.aws_env = AwsEnv(config.aws_env)

    logger.info(
        f"Running indexing for batch with config: {config}, "
        f"no. of documents: {len(document_stems)}"
    )

    temp_dir = tempfile.TemporaryDirectory()
    vespa_search_adapter = get_vespa_search_adapter_from_aws_secrets(
        cert_dir=temp_dir.name,
        vespa_private_key_param_name="VESPA_PRIVATE_KEY_FULL_ACCESS",
        vespa_public_cert_param_name="VESPA_PUBLIC_CERT_FULL_ACCESS",
    )

    errors_per_document: dict[DocumentStem, list[Error]] = {}

    async with (
        vespa_search_adapter.client.asyncio(
            connections=indexer_max_vespa_connections,  # How many tasks to have running at once
            timeout=httpx.Timeout(VESPA_MAX_TIMEOUT_MS / 1_000),  # Seconds
        ) as vespa_connection_pool,
        AsyncProfiler(
            printer=print,
            name="indexing all",
        ),
    ):

        async def return_with_document_stem(document_stem, fn):
            try:
                result = await fn
                return (document_stem, result)
            except Exception as e:
                return (document_stem, e)

        semaphore = asyncio.Semaphore(indexer_max_vespa_connections)
        tasks = [
            wait_for_semaphore(
                semaphore,
                return_with_document_stem(
                    document_stem,
                    index_all(
                        config=config,
                        run_output_identifier=run_output_identifier,
                        document_stem=document_stem,
                        vespa_connection_pool=vespa_connection_pool,
                    ),
                ),
            )
            for document_stem in document_stems
        ]

        results: list[
            tuple[
                DocumentStem,
                Result[None, list[Error]] | BaseException,
            ]
        ] = await asyncio.gather(
            *tasks,
            # Normally this is True, but since there's the wrapper
            # function to ensure that the document ID is always
            # included, which captures exceptions, it can be False
            # here.
            return_exceptions=False,
        )

        for document_stem, result in results:
            # Conditionally set it, so the length of the presence of
            # keys can be used as an indicator of the overall presence
            # of ≥ 1 error.
            if isinstance(result, BaseException):
                errors_per_document[document_stem] = [
                    Error(msg=str(result), metadata={})
                ]
            else:
                match result:
                    case Ok(_):
                        pass
                    case Err(errors):
                        errors_per_document[document_stem] = errors

    await create_indexing_summary_artifact(
        config=config,
        run_output_identifier=run_output_identifier,
        documents_stems=document_stems,
        errors_per_document=errors_per_document,
    )

    if errors_per_document:
        raise ValueError(
            f"Failed to process {len(errors_per_document)}/{len(document_stems)} documents"
        )


@flow(
    log_prints=True,
    on_failure=[SlackNotify.message],
    on_crashed=[SlackNotify.message],
)
async def run_indexing_from_aggregate_results(
    run_output_identifier: RunOutputIdentifier,
    document_stems: list[DocumentStem] | None = None,
    config: Config | None = None,
    batch_size: int = DEFAULT_DOCUMENTS_BATCH_SIZE,
    indexer_concurrency_limit: PositiveInt = DEFAULT_INDEXER_CONCURRENCY_LIMIT,
    indexer_max_vespa_connections: PositiveInt = (
        DEFAULT_VESPA_MAX_CONNECTIONS_AGG_INDEXER
    ),
) -> None:
    """Index aggregated inference results from a list of S3 URIs into Vespa."""

    logger = get_run_logger()

    if config is None:
        config = await Config.create()

    logger.info(f"Running indexing with config: {config}")

    if not document_stems:
        logger.info(
            f"Running on all documents under run_output_identifier: {run_output_identifier}"
        )
        collected_document_stems: list[DocumentStem] = (
            collect_unique_file_stems_under_prefix(  #  type: ignore[call-arg]
                bucket_name=config.cache_bucket_str,
                prefix=os.path.join(
                    config.aggregate_inference_results_prefix, run_output_identifier
                ),
            )
        )
        document_stems = collected_document_stems
        logger.info(f"Found {len(document_stems)} document import ids to process.")

    flow_name = function_to_flow_name(index_aggregate_results_for_batch_of_documents)
    deployment_name = generate_deployment_name(
        flow_name=flow_name, aws_env=config.aws_env
    )

    semaphore = asyncio.Semaphore(indexer_concurrency_limit)

    with Profiler(
        printer=print,
        name="preparing tasks",
    ):
        tasks = [
            wait_for_semaphore(
                semaphore,
                run_deployment(
                    name=f"{flow_name}/{deployment_name}",
                    parameters={
                        "document_stems": batch,
                        "config_json": config.to_json(),
                        "run_output_identifier": run_output_identifier,
                        "indexer_max_vespa_connections": indexer_max_vespa_connections,
                    },
                    # Rely on the flow's own timeout, if any, to make sure it
                    # eventually ends[1].
                    #
                    # [1]:
                    # > Setting timeout to None will allow this function to
                    # > poll indefinitely.
                    timeout=None,
                ),
            )
            for batch in iterate_batch(document_stems, batch_size)
        ]

    with Profiler(
        printer=print,
        name="gathering tasks",
    ):
        results: list[FlowRun | BaseException] = await asyncio.gather(
            *tasks, return_exceptions=True
        )

    failures = []
    successes = []
    for result in results:
        if isinstance(result, BaseException):
            failures.append(result)
        else:
            successes.append(result)

    await create_aggregate_indexing_summary_artifact(
        config=config,
        document_stems=document_stems,
        successes=successes,
        failures=failures,
    )

    if failures:
        raise ValueError(
            f"Some batches of documents had failures: {len(failures)}/{len(results)} failed."
        )
