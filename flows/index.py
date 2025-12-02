import asyncio
import json
import os
import random
import tempfile
from collections import Counter
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from typing import Any, Final, NamedTuple, TypeAlias

import httpx
from cpr_sdk.models.search import Document as VespaDocument
from cpr_sdk.models.search import Passage as VespaPassage
from mypy_boto3_s3.type_defs import (
    PutObjectOutputTypeDef,
)
from prefect import flow, task
from prefect.artifacts import (
    create_table_artifact,
)
from prefect.client.schemas import FlowRun
from prefect.context import FlowRunContext, get_run_context
from prefect.futures import PrefectFuture, PrefectFutureList
from prefect.task_runners import ThreadPoolTaskRunner
from prefect.utilities.names import generate_slug
from pydantic import (
    BaseModel,
    PositiveInt,
)

# generate_slug is being used, but in an implicit f-string
from vespa.application import VespaAsync
from vespa.io import VespaResponse

from flows.aggregate import (
    METADATA_FILE_NAME as AGGREGATE_METADATA_FILE_NAME,
)
from flows.aggregate import (
    Metadata as AggregateMetadata,
)
from flows.aggregate import (
    MiniClassifierSpec,
    SerialisedVespaConcept,
    parse_model_field,
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
from flows.config import Config
from flows.result import Err, Error, Ok, Result, is_err, unwrap_err
from flows.utils import (
    DocumentImportId,
    DocumentStem,
    Fault,
    JsonDict,
    ParameterisedFlow,
    RunOutputIdentifier,
    S3Uri,
    SlackNotify,
    collect_unique_file_stems_under_prefix,
    get_logger,
    iterate_batch,
    map_as_sub_flow,
    remove_translated_suffix,
    return_with,
    wait_for_semaphore,
)
from knowledge_graph.cloud import AwsEnv, get_async_session
from knowledge_graph.identifiers import WikibaseID

# A serialised Vespa span, see cpr_sdk.models.search.Passage.Span
SerialisedVespaSpan: TypeAlias = list[dict[str, str]]

# How many connections to Vespa to use for indexing.
DEFAULT_VESPA_MAX_CONNECTIONS_AGG_INDEXER: Final[PositiveInt] = 10
# How many indexer deployments to run concurrently.
DEFAULT_INDEXER_CONCURRENCY_LIMIT: Final[PositiveInt] = 5
# How many document passages to index concurrently per document
INDEXER_DOCUMENT_PASSAGES_CONCURRENCY_LIMIT: Final[PositiveInt] = 5

METADATA_FILE_NAME = "metadata.json"


class Metadata(BaseModel):
    """Lineage information for this index run."""

    flow_run: FlowRun
    run_output_identifier: RunOutputIdentifier
    config: Config


async def store_metadata(
    config: Config,
    run_output_identifier: RunOutputIdentifier,
) -> None:
    """Store metadata for the index run."""
    logger = get_logger()

    run_context = get_run_context()
    if isinstance(run_context, FlowRunContext):
        if run_context.flow_run is None:
            raise ValueError("run context is missing flow run")

        metadata = Metadata(
            flow_run=run_context.flow_run,
            run_output_identifier=run_output_identifier,
            config=config,
        )

        metadata_json = metadata.model_dump_json()

        logger.debug(f"writing index metadata: {metadata_json}")

        s3_uri = S3Uri(
            bucket=config.cache_bucket_str,
            key=os.path.join(
                config.index_results_prefix,
                run_output_identifier,
                METADATA_FILE_NAME,
            ),
        )

        session = get_async_session(config.aws_env, config.bucket_region)
        async with session.client("s3") as s3_client:
            response: PutObjectOutputTypeDef = await s3_client.put_object(
                Bucket=s3_uri.bucket,
                Key=s3_uri.key,
                Body=metadata_json,
                ContentType="application/json",
            )

            status_code = response["ResponseMetadata"]["HTTPStatusCode"]
            if status_code != 200:
                raise ValueError(
                    f"Failed to store index metadata to S3. Status code: {status_code}"
                )

        logger.debug(f"wrote index metadata to {s3_uri}")


async def load_aggregate_metadata(
    config: Config,
    run_output_identifier: RunOutputIdentifier,
) -> Result[AggregateMetadata, Error]:
    """Load metadata from aggregate run output."""
    logger = get_logger()

    metadata_key = os.path.join(
        config.aggregate_inference_results_prefix,
        run_output_identifier,
        "metadata.json",
    )

    try:
        metadata_dict = await load_async_json_data_from_s3(
            bucket=config.cache_bucket_str,
            key=metadata_key,
            config=config,
        )
        metadata = AggregateMetadata.model_validate(metadata_dict)
        return Ok(metadata)
    except Exception as e:
        logger.warning(f"Failed to load aggregate metadata from {metadata_key}: {e}")
        return Err(
            Error(
                msg="Failed to load aggregate metadata",
                metadata={
                    "metadata_key": metadata_key,
                    "exception": str(e),
                },
            )
        )


async def load_async_json_data_from_s3(
    bucket: str, key: str, config: Config
) -> dict[str, Any]:
    """Load JSON data from an S3 URI asynchronously"""

    session = get_async_session(config.aws_env, config.bucket_region)
    async with session.client("s3") as s3client:
        response = await s3client.get_object(Bucket=bucket, Key=key)
        body = await response["Body"].read()
        decoded_body = body.decode("utf-8")
        return json.loads(decoded_body)


async def _update_vespa_passage_concepts(
    vespa_data_id: VespaDataId,
    serialised_concepts: Sequence[SerialisedVespaConcept],
    serialised_spans: Sequence[SerialisedVespaSpan],
    vespa_connection_pool: VespaAsync,
    enable_v2_concepts: bool = False,
) -> VespaResponse:
    """Update a passage in Vespa with the given concepts."""

    path = vespa_connection_pool.app.get_document_v1_path(
        id=vespa_data_id,
        schema="document_passage",
        namespace="doc_search",
        group=None,
    )
    fields = {
        "concepts": serialised_concepts,
    }
    if enable_v2_concepts:
        fields["spans"] = serialised_spans

    response: VespaResponse = await vespa_connection_pool.update_data(
        schema="document_passage",
        namespace="doc_search",
        data_id=vespa_data_id,
        # Don't create an empty document for non-existent documents
        create=False,
        fields=fields,
    )

    logger = get_logger()

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
        logger.info(f"update data at path {path} with fields {fields}")

    if not response.is_successful():
        # Account for when Vespa fails to include the body
        try:
            # `get_json` returns a Dict[1].
            #
            # [1]: https://github.com/vespa-engine/pyvespa/blob/1b42923b77d73666e0bcd1e53431906fc3be5d83/vespa/io.py#L44-L46
            json_s = json.dumps(response.get_json())
            logger.error(f"Vespa update failed: {json_s}")
        except Exception as e:
            logger.error(f"failed to get JSON from Vespa response: {e}")

    return response


async def create_indexing_summary_artifact(
    config: Config,
    document_stems: Sequence[DocumentStem],
    successes: Sequence[FlowRun],
    failures: Sequence[FlowRun | BaseException],
) -> None:
    """Create an artifact with summary information about the indexing run."""

    # Prepare summary data for the artifact
    total_documents = len(document_stems)
    successful_document_batches_count = len(successes)
    failed_document_batches_count = len(failures)

    # Format the overview information as a string for the description
    overview_description = f"""# Aggregate Indexing Summary

## Overview
- **Environment**: {config.aws_env.value}
- **Total documents processed**: {total_documents}
- **Successful Batches**: {successful_document_batches_count}
- **Failed Batches**: {failed_document_batches_count}
"""

    details = []
    for failure in failures:
        if isinstance(failure, BaseException):
            details.append(
                {
                    "Failure": f"Exception: {type(failure).__name__}: {str(failure)}",
                    "Flow Run": "N/A",
                }
            )
        elif isinstance(failure, FlowRun):
            state_info = (
                f"State: {failure.state.type.value if failure.state else 'Unknown'}"
            )
            if failure.state and failure.state.message:
                state_info += f", Message: {failure.state.message}"
            details.append(
                {
                    "Failure": state_info,
                    "Flow Run": failure.name,
                }
            )

    await create_table_artifact(  # pyright: ignore[reportGeneralTypeIssues]
        key=f"indexing-aggregate-results-summary-{config.aws_env.value}",
        table=details,
        description=overview_description,
    )


@dataclass(frozen=True)
class SimpleConcept:
    """
    A simple, hashable concept.

    As of 2025-06-03, the Concept from the cpr_sdk isn't hashable.

    The parsed_model field contains the parsed classifier spec info
    from the model field. It is None for old format concepts (before
    the new model format was introduced).
    """

    id: str
    name: str
    model: str
    parsed_model: MiniClassifierSpec | None


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


async def index_document_passages(
    config: Config,
    run_output_identifier: RunOutputIdentifier,
    document_stem: DocumentStem,
    vespa_connection_pool: VespaAsync,
    indexer_document_passages_concurrency_limit: PositiveInt = INDEXER_DOCUMENT_PASSAGES_CONCURRENCY_LIMIT,
    enable_v2_concepts: bool = False,
) -> list[Result[list[SimpleConcept], Error]]:
    """Index aggregated inference results from S3 into Vespa document passages."""
    aggregated_results_s3_uri = generate_s3_uri_input_document_passages(
        cache_bucket=config.cache_bucket_str,
        aggregate_inference_results_prefix=config.aggregate_inference_results_prefix,
        run_output_identifier=run_output_identifier,
        document_stem=document_stem,
    )

    logger = get_logger()

    logger.info(
        f"Loading aggregated inference results from S3: {aggregated_results_s3_uri}"
    )

    raw_data = await load_async_json_data_from_s3(
        bucket=aggregated_results_s3_uri.bucket,
        key=aggregated_results_s3_uri.key,
        config=config,
    )
    aggregated_inference_results: dict[
        TextBlockId,
        Sequence[SerialisedVespaConcept],
    ] = {TextBlockId(k): v for k, v in raw_data.items()}

    document_id: DocumentImportId = remove_translated_suffix(document_stem)
    logger.info(
        f"Querying Vespa for passages related to document import ID: {document_id}"
    )

    passages_generator = get_document_passages_from_vespa__generator(
        document_import_id=document_id,
        vespa_connection_pool=vespa_connection_pool,
    )

    passages_in_vespa: dict[TextBlockId, tuple[VespaHitId, VespaPassage]] = {}
    async for passage_batch in passages_generator:
        passages_in_vespa.update(passage_batch)

    logger.info(
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

        if enable_v2_concepts:
            serialised_spans = build_v2_passage_spans(
                text_block_id=text_block_id,
                serialised_concepts=serialised_concepts,
            )
        else:
            serialised_spans = []

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
                        serialised_spans=serialised_spans,
                        vespa_connection_pool=vespa_connection_pool,
                        enable_v2_concepts=enable_v2_concepts,
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
                except Exception:
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

            simple_concepts = []
            for concept in serialised_concepts:
                if not isinstance(concept, dict):
                    logger.warning(f"Expected dict for concept, got {type(concept)}")
                    continue

                concept_id = concept.get("id")
                concept_name = concept.get("name")
                concept_model = concept.get("model")

                if not all(
                    isinstance(v, str)
                    for v in [concept_id, concept_name, concept_model]
                ):
                    logger.warning(
                        f"Invalid concept fields in {text_block_id}: "
                        f"id={concept_id}, name={concept_name}, model={concept_model}"
                    )
                    continue

                simple_concepts.append(
                    SimpleConcept(
                        id=concept_id,  # type: ignore[arg-type]  # validated above
                        name=concept_name,  # type: ignore[arg-type]  # validated above
                        model=concept_model,  # type: ignore[arg-type]  # validated above
                        parsed_model=parse_model_field(concept_model),  # type: ignore[arg-type]  # validated above
                    )
                )

            results.append(Ok(simple_concepts))

    return results


class Index(NamedTuple):
    """An index for a span into a passage."""

    start: int
    end: int


def build_v2_passage_spans(
    text_block_id: TextBlockId,
    serialised_concepts: Sequence[SerialisedVespaConcept],
) -> Sequence[SerialisedVespaSpan]:
    """Group and enrich v1 spans into v2 spans."""
    logger = get_logger()

    spans: dict[Index, VespaPassage.Span] = {}
    for serialised_concept_json in serialised_concepts:
        # Make sure it's valid and easier to access
        vespa_concept: VespaPassage.Concept = VespaPassage.Concept.model_validate(
            serialised_concept_json
        )

        parsed_model = parse_model_field(vespa_concept.model)
        if parsed_model is None:
            continue

        try:
            # This is mostly done to ensure that it's a valid Wikibase ID
            wikibase_id = WikibaseID(vespa_concept.id)
        except ValueError as e:
            logger.warning(
                f"invalid Wikibase ID {vespa_concept.id} for concept in "
                f"text block {text_block_id}: {e}"
            )
            continue

        # Verify the Wikibase ID in the model field matches the
        # concept ID. The concept ID in the v1 concept in the SDK
        # should be Wikibase ID. It's not a canonical ID for a concept.
        if parsed_model.wikibase_id != wikibase_id:
            logger.warning(
                f"Wikibase ID mismatch for concept in text block {text_block_id}: "
                f"concept.id={wikibase_id}, model={vespa_concept.model}"
            )
            continue

        index = Index(
            start=vespa_concept.start,
            end=vespa_concept.end,
        )

        concept: VespaPassage.Span.ConceptV2 = VespaPassage.Span.ConceptV2(
            concept_id=parsed_model.concept_id,
            concept_wikibase_id=parsed_model.wikibase_id,
            classifier_id=parsed_model.classifier_id,
        )

        if span := spans.get(index):
            span.concepts_v2 = list(span.concepts_v2) + [concept]
        else:
            spans[index] = VespaPassage.Span(
                start=index.start,
                end=index.end,
                concepts_v2=[concept],
            )

    return [v.model_dump(mode="json") for _k, v in spans.items()]  # type: ignore[return-value]


def build_v2_document_concepts(
    simple_concepts: list[SimpleConcept],
) -> list[dict[str, Any]]:
    """Group and enrich v1 concepts into v2 concepts."""
    logger = get_logger()

    # Count occurrences of each concept
    concepts_counter: Counter[SimpleConcept] = Counter(simple_concepts)

    document_concepts: list[VespaDocument.ConceptV2] = []
    for concept, count in concepts_counter.items():
        # Check if we have parsed model info (new format)
        if concept.parsed_model is None:
            # Old format - skip v2 enrichment
            logger.debug(
                f"skipping v2 enrichment for concept {concept.id}: old model format"
            )
            continue

        # This is mostly done to ensure that it's a valid Wikibase ID
        try:
            wikibase_id = WikibaseID(concept.id)
        except ValueError as e:
            logger.warning(f"invalid Wikibase ID {concept.id} for concept: {e}")
            continue

        # Verify the Wikibase ID in the model field matches the
        # concept ID. The concept ID in the v1 concept in the SDK
        # should be Wikibase ID. It's not a canonical ID for a concept.
        if concept.parsed_model.wikibase_id != wikibase_id:
            logger.warning(
                f"Wikibase ID mismatch for concept: "
                f"concept.id={wikibase_id}, model={concept.model}"
            )
            continue

        document_concept = VespaDocument.ConceptV2(
            concept_id=concept.parsed_model.concept_id,
            concept_wikibase_id=concept.parsed_model.wikibase_id,
            classifier_id=concept.parsed_model.classifier_id,
            count=count,
        )
        document_concepts.append(document_concept)

    return [concept.model_dump(mode="json") for concept in document_concepts]


async def index_family_document(
    document_id: DocumentImportId,
    vespa_connection_pool: VespaAsync,
    simple_concepts: list[SimpleConcept],
    enable_v2_concepts: bool = False,
) -> Result[None, Error]:
    """Index document concept counts in Vespa via partial update."""
    logger = get_logger()

    concepts_counts: Counter[SimpleConcept] = Counter(simple_concepts)

    concepts_counts_with_names = {
        f"{concept.id}{CONCEPT_COUNT_SEPARATOR}{concept.name}": count
        for concept, count in concepts_counts.items()
    }

    logger.debug(f"serialised concepts counts: {concepts_counts_with_names}")

    concepts_v2 = build_v2_document_concepts(
        simple_concepts=simple_concepts,
    )

    path = vespa_connection_pool.app.get_document_v1_path(
        id=document_id,
        schema="family_document",
        namespace="doc_search",
        group=None,
    )
    fields: JsonDict = JsonDict(
        {
            # NB: The schema is misnamed in Vespa
            "concept_counts": concepts_counts_with_names,
        }
    )

    if enable_v2_concepts:
        fields["concepts_v2"] = concepts_v2

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


@task(  # pyright: ignore[reportCallIssue]
    task_run_name=task_run_name,  # pyright: ignore[reportArgumentType]
)
async def index_all(
    document_stem: DocumentStem,
    config: Config,
    run_output_identifier: RunOutputIdentifier,
    indexer_document_passages_concurrency_limit: PositiveInt,
    indexer_max_vespa_connections: PositiveInt,
    enable_v2_concepts: bool = False,
) -> DocumentStem:
    """Indexes all (document passages and family documents) data."""
    try:
        # Create Vespa connection inside the task to avoid serialization issues
        temp_dir = tempfile.TemporaryDirectory()
        vespa_search_adapter = get_vespa_search_adapter_from_aws_secrets(
            cert_dir=temp_dir.name,
            vespa_private_key_param_name="VESPA_PRIVATE_KEY_FULL_ACCESS",
            vespa_public_cert_param_name="VESPA_PUBLIC_CERT_FULL_ACCESS",
            aws_env=config.aws_env,
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
                enable_v2_concepts=enable_v2_concepts,
            )

            simple_concepts: list[SimpleConcept] = []
            errors: list[Error] = []
            for result in results:
                match result:
                    case Ok(val):
                        simple_concepts.extend(val)
                    case Err(err):
                        errors.append(err)

            document_id: DocumentImportId = remove_translated_suffix(document_stem)

            result = await index_family_document(
                document_id=document_id,
                vespa_connection_pool=vespa_connection_pool,
                simple_concepts=simple_concepts,
                enable_v2_concepts=enable_v2_concepts,
            )

            if is_err(result):
                errors.append(unwrap_err(result))

            if errors:
                raise Fault(
                    msg="Failed to index document passages or family document",
                    metadata={
                        "document_stem": document_stem,
                        "errors": errors,
                    },
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


@flow(  # pyright: ignore[reportCallIssue]
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
    enable_v2_concepts: bool = False,
) -> None:
    """Index aggregated inference results into Vespa for family documents and document passages."""

    logger = get_logger()

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

    tasks: list[PrefectFuture[Any]] = []

    for document_stem in document_stems:
        tasks.append(
            index_all.submit(  # pyright: ignore[reportFunctionMemberAccess]
                document_stem=document_stem,
                config=config,
                run_output_identifier=run_output_identifier,
                indexer_document_passages_concurrency_limit=int(
                    indexer_document_passages_concurrency_limit
                ),
                indexer_max_vespa_connections=indexer_max_vespa_connections,
                enable_v2_concepts=enable_v2_concepts,
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
    enable_v2_concepts: bool | None = None,
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

    config : Config
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

    enable_v2_concepts : bool
        Whether to index them into Vespa or not. If set to boolean
        value will be inferred.
    """

    logger = get_logger()

    if config is None:
        config = await Config.create()

    logger.info(f"Running indexing with config: {config}")

    # `None` indicates that there wasn't a human set value, and thus
    # we can programmatically set a value.
    #
    # Labs is treated as different to the other 3, as per usual.
    if enable_v2_concepts is None:
        enable_v2_concepts = config.aws_env in [
            AwsEnv.sandbox,
            AwsEnv.staging,
            AwsEnv.production,
        ]

    logger.info(f"v2 concepts enabled: {enable_v2_concepts}")

    try:
        if config.cache_bucket:
            await store_metadata(
                config=config,
                run_output_identifier=run_output_identifier,
            )
    except Exception as e:
        logger.error(f"Failed to store index metadata: {e}")

    # Load metadata from aggregate run
    _ = await load_aggregate_metadata(
        config=config,
        run_output_identifier=run_output_identifier,
    )

    if not document_stems:
        logger.info(
            f"Running on all documents under run_output_identifier: {run_output_identifier}"
        )
        collected_document_stems: list[
            DocumentStem
        ] = await collect_unique_file_stems_under_prefix(
            bucket_name=config.cache_bucket_str,
            prefix=os.path.join(
                config.aggregate_inference_results_prefix,
                run_output_identifier,
            ),
            bucket_region=config.bucket_region,
            disallow={AGGREGATE_METADATA_FILE_NAME},
        )
        document_stems = collected_document_stems
        logger.info(f"Found {len(document_stems)} document import ids to process.")

    batches = iterate_batch(document_stems, batch_size)

    def parameters(batch: Sequence[DocumentStem]) -> dict[str, Any]:
        return {
            "document_stems": batch,
            "config_json": config.model_dump(),
            "run_output_identifier": run_output_identifier,
            "indexer_document_passages_concurrency_limit": indexer_document_passages_concurrency_limit,
            "indexer_max_vespa_connections": indexer_max_vespa_connections,
            "enable_v2_concepts": enable_v2_concepts,
        }

    parameterised_batches: Sequence[ParameterisedFlow] = []
    for batch in batches:
        parameterised_batches.append(
            ParameterisedFlow(
                # The typing doesn't pick up the Flow decorator
                fn=index_batch_of_documents,  # pyright: ignore[reportArgumentType]
                params=parameters(batch),
            )
        )

    successes, failures = await map_as_sub_flow(  # pyright: ignore[reportCallIssue]
        aws_env=config.aws_env,
        counter=indexer_concurrency_limit,
        parameterised_batches=parameterised_batches,
        unwrap_result=False,
    )

    await create_indexing_summary_artifact(
        config=config,
        document_stems=document_stems,
        successes=successes,
        failures=failures,
    )

    if failures:
        raise ValueError(
            f"Some batches of documents had failures: {len(failures)}/{len(successes) + len(failures)} failed."
        )
