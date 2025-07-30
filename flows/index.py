import asyncio
import json
import os
import random
import tempfile
from collections import Counter
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from typing import Any, Final

import boto3
import httpx
from cpr_sdk.models.search import Passage as VespaPassage
from prefect import flow, unmapped
from prefect.artifacts import create_markdown_artifact, create_table_artifact
from prefect.assets import materialize
from prefect.client.schemas.objects import FlowRun
from prefect.futures import PrefectFuture, PrefectFutureList
from prefect.logging import get_run_logger
from prefect.task_runners import ThreadPoolTaskRunner

# generate_slug is being used, but in an implicit f-string
from prefect.utilities.names import generate_slug  # noqa: F401
from pydantic import PositiveInt
from vespa.application import Vespa, VespaAsync
from vespa.io import VespaResponse

from flows.aggregate import (
    Config,
    RunOutputIdentifier,
    SerialisedVespaConcept,
)
from flows.boundary import (
    CONCEPT_COUNT_SEPARATOR,
    DEFAULT_DOCUMENTS_BATCH_SIZE,
    VESPA_MAX_TIMEOUT_MS,
    TextBlockId,
    VespaDataId,
    VespaHitId,
    get_data_id_from_vespa_hit_id,
    get_document_passages_from_vespa__generator,
    get_vespa_search_adapter_from_aws_secrets,
)
from flows.result import Err, Error, Ok, Result
from flows.utils import (
    DocumentImportId,
    DocumentStem,
    Fault,
    S3Uri,
    SlackNotify,
    collect_unique_file_stems_under_prefix,
    iterate_batch,
    map_as_sub_flow,
    remove_translated_suffix,
    return_with,
    wait_for_semaphore,
)
from scripts.cloud import AwsEnv

# How many connections to Vespa to use for indexing.
DEFAULT_VESPA_MAX_CONNECTIONS_AGG_INDEXER: Final[PositiveInt] = 10
# How many indexer deployments to run concurrently.
DEFAULT_INDEXER_CONCURRENCY_LIMIT: Final[PositiveInt] = 5
# How many document passages to index concurrently per document
INDEXER_DOCUMENT_PASSAGES_CONCURRENCY_LIMIT: Final[PositiveInt] = 5


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

    path = vespa_connection_pool.app.get_document_v1_path(
        id=vespa_data_id,
        schema="document_passage",
        namespace="doc_search",
        group=None,
    )
    fields = {"concepts": serialised_concepts}

    response: VespaResponse = await vespa_connection_pool.update_data(
        schema="document_passage",
        namespace="doc_search",
        data_id=vespa_data_id,
        # Don't create an empty document for non-existent documents
        create=False,
        fields=fields,
    )

    # Currently, this function isn't aware of which index number it is
    # out of all the document passages for a family document. There
    # may be over 50,000 document passages. With these 2 constraints,
    # randomly sample from a pseudo-random distribution and
    # conditionally print this extra info.
    if random.random() < 0.1:
        # Example:
        #
        # update data at path
        # /document/v1/doc_search/document_passage/docid/CCLW.executive.10014.4470.1039
        # with fields {'concepts': [{'id': 'Q387', 'name':
        # 'concept_81', 'parent_concepts': [],
        # 'parent_concept_ids_flat': '', 'model':
        # 'KeywordClassifier("concept_81")', 'end': 157, 'start': 166,
        # 'timestamp': '2025-05-22T17:34:09.649548'}, {'id': 'Q299',
        # 'name': 'concept_51', 'parent_concepts': [],
        # 'parent_concept_ids_flat': '', 'model':
        # 'KeywordClassifier("concept_51")', 'end': 108, 'start': 115,
        # 'timestamp': '2025-05-22T17:34:09.649548'}]}
        print(f"update data at path {path} with fields {fields}")

    if not response.is_successful():
        # Account for when Vespa fails to include the body
        try:
            # `get_json` returns a Dict[1].
            #
            # [1]: https://github.com/vespa-engine/pyvespa/blob/1b42923b77d73666e0bcd1e53431906fc3be5d83/vespa/io.py#L44-L46
            json_s = json.dumps(response.get_json())
            print(f"Vespa update failed: {json_s}")
        except Exception as e:
            print(f"failed to get JSON from Vespa response: {e}")

    return response


async def create_aggregate_indexing_summary_artifact(
    config: Config,
    document_stems: Sequence[DocumentStem],
    successes: Sequence[None],
    failures: Sequence[FlowRun | BaseException],
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

    await create_markdown_artifact(  # pyright: ignore[reportGeneralTypeIssues]
        key=f"indexing-aggregate-results-summary-{config.aws_env.value}",
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


def generate_s3_uri_input_document_passages(
    cache_bucket: str,
    aggregate_inference_results_prefix: str,
    run_output_identifier: RunOutputIdentifier,
    document_stem: DocumentStem,
) -> S3Uri:
    return S3Uri(
        bucket=cache_bucket,
        key=os.path.join(
            aggregate_inference_results_prefix,
            run_output_identifier,
            f"{document_stem}.json",
        ),
    )


def generate_s3_uri_output_family_document(
    vespa: Vespa, document_stem: DocumentStem
) -> str:
    document_id: DocumentImportId = remove_translated_suffix(document_stem)
    # Example: /document/v1/doc_search/family_document/docid/CCLW.executive.10014.4470
    document_v1_path = vespa.get_document_v1_path(
        id=document_id,
        schema="document_passage",
        namespace="doc_search",
        group=None,
    )
    return f"vespa:/{document_v1_path}"


async def index_document_passages(
    config: Config,
    run_output_identifier: RunOutputIdentifier,
    document_stem: DocumentStem,
    vespa_connection_pool: VespaAsync,
    indexer_document_passages_concurrency_limit: PositiveInt = INDEXER_DOCUMENT_PASSAGES_CONCURRENCY_LIMIT,
) -> list[Result[list[SimpleConcept], Error]]:
    """Index aggregated inference results from S3 into Vespa document passages."""
    aggregated_results_s3_uri = generate_s3_uri_input_document_passages(
        cache_bucket=config.cache_bucket_str,
        aggregate_inference_results_prefix=config.aggregate_inference_results_prefix,
        run_output_identifier=run_output_identifier,
        document_stem=document_stem,
    )

    print(f"Loading aggregated inference results from S3: {aggregated_results_s3_uri}")

    raw_data = load_json_data_from_s3(
        bucket=aggregated_results_s3_uri.bucket, key=aggregated_results_s3_uri.key
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

    semaphore = asyncio.Semaphore(indexer_document_passages_concurrency_limit)
    tasks: list[Awaitable[tuple[TextBlockId, VespaResponse | Exception]]] = []
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

        tasks.append(
            wait_for_semaphore(
                semaphore,
                return_with(
                    text_block_id,
                    _update_vespa_passage_concepts(
                        vespa_data_id=vespa_data_id,
                        serialised_concepts=serialised_concepts,
                        vespa_connection_pool=vespa_connection_pool,
                    ),
                ),
            )
        )

    responses: list[
        tuple[
            TextBlockId,
            VespaResponse | Exception,
        ]
    ] = await asyncio.gather(
        *tasks,
        # Normally this is True, but since there's the wrapper
        # function to ensure that the ID is always included, which
        # captures exceptions, it can be False here.
        return_exceptions=False,
    )

    for text_block_id, response in responses:
        if isinstance(response, Exception):
            error = Error(
                msg="Vespa update failed",
                metadata={
                    "text_block_id": text_block_id,
                    "exception": str(response),
                },
            )
            results.append(Err(error))
        else:
            if not response.is_successful():
                # Account for when Vespa fails to include the body
                try:
                    # `get_json` returns a Dict[1].
                    #
                    # [1]: https://github.com/vespa-engine/pyvespa/blob/1b42923b77d73666e0bcd1e53431906fc3be5d83/vespa/io.py#L44-L46
                    json = response.get_json()
                except Exception as e:
                    print(f"failed to get JSON from Vespa response: {e}")
                    json = None

                error = Error(
                    msg="Vespa update failed",
                    metadata={
                        "text_block_id": text_block_id,
                        "json": json,
                    },
                )
                results.append(Err(error))
                continue

            serialised_concepts = aggregated_inference_results[text_block_id]

            results.append(
                Ok(
                    [
                        SimpleConcept(id=concept["id"], name=concept["name"])
                        for concept in serialised_concepts
                    ]
                )
            )

    return results


async def index_family_document(
    document_id: DocumentImportId,
    vespa_connection_pool: VespaAsync,
    simple_concepts: list[SimpleConcept],
) -> Result[None, Error]:
    """Index document concept counts in Vespa via partial update."""
    logger = get_run_logger()

    concepts_counts: Counter[SimpleConcept] = Counter(simple_concepts)

    concepts_counts_with_names = {
        f"{concept.id}{CONCEPT_COUNT_SEPARATOR}{concept.name}": count
        for concept, count in concepts_counts.items()
    }

    logger.debug(f"serialised concepts counts: {concepts_counts_with_names}")

    path = vespa_connection_pool.app.get_document_v1_path(
        id=document_id,
        schema="family_document",
        namespace="doc_search",
        group=None,
    )
    # NB: The schema is misnamed in Vespa
    fields = {"concept_counts": concepts_counts_with_names}
    logger.debug(f"update data at path {path} with fields {fields}")

    response: VespaResponse = await vespa_connection_pool.update_data(
        schema="family_document",
        namespace="doc_search",
        data_id=document_id,
        # Don't create an empty document for non-existent documents
        create=False,
        fields=fields,
    )

    if not response.is_successful():
        # `get_json` returns a Dict[1].
        #
        # [1]: https://github.com/vespa-engine/pyvespa/blob/1b42923b77d73666e0bcd1e53431906fc3be5d83/vespa/io.py#L44-L46
        logger.error(f"Vespa update failed: {json.dumps(response.get_json())}")

        return Err(
            Error(
                msg="Vespa update failed",
                metadata={"json": response.get_json()},
            )
        )

    return Ok(None)


async def create_indexing_batch_summary_artifact(
    config: Config,
    run_output_identifier: RunOutputIdentifier,
    documents_stems: list[DocumentStem],
    fault_per_document: dict[DocumentStem, Fault],
) -> None:
    """Create a markdown report artifact with summary information about the indexing run."""

    # Prepare summary data for the artifact
    total_documents = len(documents_stems)
    failed_documents = len(fault_per_document)
    successful_documents = total_documents - failed_documents
    # Count total errors across all documents (each fault may contain
    # multiple errors or just 1 exception).
    total_errors = sum(
        (
            len(
                fault.metadata.get(  # pyright: ignore[reportOptionalMemberAccess]
                    "errors", []
                )
            )
            if isinstance(
                fault.metadata.get("errors"),  # pyright: ignore[reportOptionalMemberAccess]
                list,
            )
            else 1
        )
        for fault in fault_per_document.values()
    )

    # Format the overview information as a string for the description
    overview_description = f"""# Indexing from Aggregate Results
    Summary

## Overview
- **Run Output Identifier**: {run_output_identifier}
- **Environment**: {config.aws_env.value}
- **Total documents processed**: {total_documents}
- **Successful documents**: {successful_documents}
- **Failed documents**: {failed_documents}
- **Total errors**: {total_errors}"""

    # Create document details table
    document_details = []
    for document_id in documents_stems:
        fault = fault_per_document.get(document_id)
        status = "✗" if fault else "✓"
        error_messages = str(fault) if fault else "N/A"
        document_details.append(
            {
                "Family document ID": document_id,
                "Status": status,
                "Errors": error_messages,
            }
        )

    # Create a single artifact with overview in description and document details in table
    await create_table_artifact(  # pyright: ignore[reportGeneralTypeIssues]
        key=f"indexing-aggregate-results-{config.aws_env.value}",
        table=document_details,
        description=overview_description,
    )


def task_run_name(parameters: dict[str, Any]) -> str:
    slug = generate_slug(2)

    # Prefer this flow actually running, even if a change in the
    # params hasn't been reflected in this function.
    match parameters.get("document_stem"):
        case document_stem if isinstance(document_stem, str):
            return f"{document_stem}-{slug}"
        case _:
            return slug


@materialize(
    "foo://bar",  # Asset key is not known yet
    log_prints=True,
    task_run_name=task_run_name,  # pyright: ignore[reportArgumentType]
)
async def index_all(
    document_stem: DocumentStem,
    config: Config,
    run_output_identifier: RunOutputIdentifier,
    indexer_document_passages_concurrency_limit: PositiveInt,
    indexer_max_vespa_connections: PositiveInt,
) -> DocumentStem:
    """Indexes all (document passages and family documents) data."""
    try:
        # Create Vespa connection inside the task to avoid serialization issues
        temp_dir = tempfile.TemporaryDirectory()
        vespa_search_adapter = get_vespa_search_adapter_from_aws_secrets(
            cert_dir=temp_dir.name,
            vespa_private_key_param_name="VESPA_PRIVATE_KEY_FULL_ACCESS",
            vespa_public_cert_param_name="VESPA_PUBLIC_CERT_FULL_ACCESS",
        )

        async with vespa_search_adapter.client.asyncio(
            connections=indexer_max_vespa_connections,
            timeout=httpx.Timeout(VESPA_MAX_TIMEOUT_MS / 1_000),
        ) as vespa_connection_pool:
            results = await index_document_passages(
                config=config,
                run_output_identifier=run_output_identifier,
                document_stem=document_stem,
                vespa_connection_pool=vespa_connection_pool,
                indexer_document_passages_concurrency_limit=indexer_document_passages_concurrency_limit,
            )

            simple_concepts: list[SimpleConcept] = []
            errors: list[Error] = []
            for result in results:
                match result:
                    case Ok(val):
                        simple_concepts.extend(val)
                    case Err(err):
                        errors.append(err)
                        print(f"indexing document passage failed: {err}")

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
                raise Fault(
                    msg="Failed to index document passages or family document",
                    metadata={"document_stem": document_stem, "errors": errors},
                )

        return document_stem
    except Exception as e:
        raise Fault(
            msg="Unexpected exception during document indexing",
            metadata={
                "document_stem": document_stem,
                "exception": e,
                "context": repr(e),
            },
        )


# Document passages aren't in here yet, since we can't get the
# Vespa data IDs here, to build a document V1 path. They're
# available inside the function that becomes a task.
def generate_assets(vespa: Vespa, document_stem: DocumentStem) -> Sequence[str]:
    return [
        generate_s3_uri_output_family_document(
            vespa=vespa,
            document_stem=document_stem,
        )
    ]


# Only document passages is mentioned here, since they're the
# same inputs that are used for updating family documents too.
def generate_asset_deps(
    cache_bucket: str,
    aggregate_inference_results_prefix: str,
    run_output_identifier: RunOutputIdentifier,
    document_stem: DocumentStem,
) -> Sequence[str]:
    return [
        str(
            generate_s3_uri_input_document_passages(
                cache_bucket=cache_bucket,
                aggregate_inference_results_prefix=aggregate_inference_results_prefix,
                run_output_identifier=run_output_identifier,
                document_stem=document_stem,
            )
        ),
    ]


@flow(  # pyright: ignore[reportCallIssue]
    log_prints=True,
    timeout_seconds=None,
    task_runner=ThreadPoolTaskRunner(
        # This is valid, as per the docs[1].
        #
        # [1]: https://github.com/PrefectHQ/prefect/blob/01f9d5e7d5204f5b8760b431d72db52dd78e6aca/docs/v3/concepts/task-runners.mdx#L49
        max_workers=10  # pyright: ignore[reportArgumentType]
    ),
)
async def index_batch_of_documents(
    run_output_identifier: RunOutputIdentifier,
    document_stems: list[DocumentStem],
    config_json: dict[str, Any],
    indexer_document_passages_concurrency_limit: PositiveInt = INDEXER_DOCUMENT_PASSAGES_CONCURRENCY_LIMIT,
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
    config = Config.model_validate(config_json)
    config.aws_env = AwsEnv(config.aws_env)

    logger.info(
        f"Running indexing for batch with config: {config}, "
        f"no. of documents: {len(document_stems)}"
    )

    # Create Vespa connection inside the task to avoid serialization issues
    temp_dir = tempfile.TemporaryDirectory()
    vespa: Vespa = get_vespa_search_adapter_from_aws_secrets(
        cert_dir=temp_dir.name,
        vespa_private_key_param_name="VESPA_PRIVATE_KEY_FULL_ACCESS",
        vespa_public_cert_param_name="VESPA_PUBLIC_CERT_FULL_ACCESS",
    ).client

    tasks: list[PrefectFuture[Any]] = []

    for document_stem in document_stems:
        tasks.append(
            index_all.with_options(  # pyright: ignore[reportUnknownMemberType]
                assets=generate_assets(
                    vespa=vespa,
                    document_stem=document_stem,
                ),
                asset_deps=generate_asset_deps(
                    cache_bucket=config.cache_bucket_str,
                    aggregate_inference_results_prefix=config.aggregate_inference_results_prefix,
                    run_output_identifier=run_output_identifier,
                    document_stem=document_stem,
                ),  # pyright: ignore[reportArgumentType]
            ).submit(
                document_stem=document_stem,
                config=config,
                run_output_identifier=run_output_identifier,
                indexer_document_passages_concurrency_limit=unmapped(
                    int(indexer_document_passages_concurrency_limit)  # pyright: ignore[reportArgumentType]
                ),
                indexer_max_vespa_connections=indexer_max_vespa_connections,
            )
        )

    # This is valid, as per the source code[1].
    #
    # [1]: https://github.com/PrefectHQ/prefect/blob/01f9d5e7d5204f5b8760b431d72db52dd78e6aca/src/prefect/task_runners.py#L183-L213
    futures: PrefectFutureList[Any] = PrefectFutureList(tasks)  # pyright: ignore[reportAbstractUsage]
    results = futures.result(raise_on_failure=False)

    fault_per_document: dict[DocumentStem, Fault] = {}
    for result in results:
        if isinstance(result, Fault):
            fault_per_document[result.metadata.get("document_stem")] = result  # pyright: ignore[reportOptionalMemberAccess, reportArgumentType]

    await create_indexing_batch_summary_artifact(
        config=config,
        run_output_identifier=run_output_identifier,
        documents_stems=document_stems,
        fault_per_document=fault_per_document,
    )

    if fault_per_document:
        raise ValueError(
            f"Failed to process {len(fault_per_document)}/{len(document_stems)} documents"
        )


@flow(
    log_prints=True,
    on_failure=[SlackNotify.message],
    on_crashed=[SlackNotify.message],
)
async def index(
    run_output_identifier: RunOutputIdentifier,
    document_stems: Sequence[DocumentStem] | None = None,
    config: Config | None = None,
    batch_size: int = DEFAULT_DOCUMENTS_BATCH_SIZE,
    indexer_concurrency_limit: PositiveInt = DEFAULT_INDEXER_CONCURRENCY_LIMIT,
    indexer_document_passages_concurrency_limit: PositiveInt = INDEXER_DOCUMENT_PASSAGES_CONCURRENCY_LIMIT,
    indexer_max_vespa_connections: PositiveInt = (
        DEFAULT_VESPA_MAX_CONNECTIONS_AGG_INDEXER
    ),
) -> None:
    """
    Index aggregated inference results from a list of S3 URIs into Vespa.

    Parameters:
    ----------
    run_output_identifier : str
        The identifier for an aggregation run. This also represents the S3 sub
        prefix that the aggregated results are saved to.

    document_stems : list[str]
        The list of document stems to index.

    config : AggregationConfig
        The configuration for the indexing.

    batch_size : int
        The size of the batch to index within each sub deployment.

    indexer_concurrency_limit : int
        The maximum number of indexing flows to run concurrently. This represents
        the number of ECS tasks that are run concurrently.

    indexer_document_passages_concurrency_limit : int
        The maximum number of document passages to index concurrently within each
        indexing flow.

    indexer_max_vespa_connections : int
        The maximum number of Vespa connections to use within each indexing flow.
    """

    logger = get_run_logger()

    if config is None:
        config = await Config.create()

    logger.info(f"Running indexing with config: {config}")

    if not document_stems:
        logger.info(
            f"Running on all documents under run_output_identifier: {run_output_identifier}"
        )
        collected_document_stems: list[DocumentStem] = (
            collect_unique_file_stems_under_prefix(
                bucket_name=config.cache_bucket_str,
                prefix=os.path.join(
                    config.aggregate_inference_results_prefix, run_output_identifier
                ),
            )
        )
        document_stems = collected_document_stems
        logger.info(f"Found {len(document_stems)} document import ids to process.")

    batches = iterate_batch(document_stems, batch_size)

    def parameters(
        config_json: dict[str, Any], batch: Sequence[DocumentStem]
    ) -> dict[str, Any]:
        return {
            "document_stems": batch,
            "config_json": config_json,
            "run_output_identifier": run_output_identifier,
            "indexer_document_passages_concurrency_limit": indexer_document_passages_concurrency_limit,
            "indexer_max_vespa_connections": indexer_max_vespa_connections,
        }

    successes, failures = await map_as_sub_flow(  # pyright: ignore[reportCallIssue]
        fn=index_batch_of_documents,  # pyright: ignore[reportArgumentType]
        aws_env=config.aws_env,
        counter=indexer_concurrency_limit,
        batches=batches,
        parameters=parameters,
        config_json=config.model_dump(),
        unwrap_result=False,
    )

    await create_aggregate_indexing_summary_artifact(
        config=config,
        document_stems=document_stems,
        successes=successes,
        failures=failures,
    )

    if failures:
        raise ValueError(
            f"Some batches of documents had failures: {len(failures)}/{len(successes) + len(failures)} failed."
        )
