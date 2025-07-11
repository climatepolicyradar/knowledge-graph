"""Boundary between Prefect, Vespa, and AWS."""

import asyncio
import base64
import gc
import json
import logging
import math
import os
import re
import sys
import tempfile
import time
from collections import Counter
from collections.abc import AsyncGenerator, Generator, Sequence
from datetime import timedelta
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import (
    Any,
    Callable,
    Final,
    NewType,
    Protocol,
    TypeVar,
    Union,
)

import boto3
import httpx
import tenacity
import vespa.querybuilder as qb
from cpr_sdk.models.search import Concept as VespaConcept
from cpr_sdk.models.search import Document as VespaDocument
from cpr_sdk.models.search import Passage as VespaPassage
from cpr_sdk.s3 import _get_s3_keys_with_prefix, _s3_object_read_text
from cpr_sdk.search_adaptors import VespaSearchAdapter
from cpr_sdk.ssm import get_aws_ssm_param
from cpr_sdk.utils import dig
from prefect import flow, get_run_logger
from prefect.client.schemas.objects import FlowRun, StateType
from prefect.deployments import run_deployment
from prefect.logging import get_logger
from pydantic import BaseModel, NonNegativeInt, PositiveInt
from types_aiobotocore_s3.client import S3Client
from vespa.application import VespaAsync
from vespa.exceptions import VespaError
from vespa.io import VespaQueryResponse, VespaResponse
from vespa.package import Document, Schema
from vespa.querybuilder import Grouping as G

from flows.utils import (
    AsyncProfiler,
    DocumentImporter,
    DocumentImportId,
    DocumentObjectUri,
    DocumentStem,
    Profiler,
    S3Uri,
    SlackNotify,
    get_labelled_passage_paths,
    iterate_batch,
    remove_translated_suffix,
    s3_file_exists,
)
from scripts.cloud import (
    AwsEnv,
    ClassifierSpec,
    function_to_flow_name,
    generate_deployment_name,
)
from src.concept import Concept
from src.exceptions import QueryError
from src.identifiers import FamilyDocumentID, WikibaseID
from src.labelled_passage import LabelledPassage
from src.span import Span

# Provide a generic type to use instead of `Any` for types hints
T = TypeVar("T")

CONCEPT_COUNT_SEPARATOR: Final[str] = ":"
CONCEPTS_COUNTS_PREFIX_DEFAULT: Final[str] = "concepts_counts"
DEFAULT_DOCUMENTS_BATCH_SIZE: Final[PositiveInt] = 50
DEFAULT_TEXT_BLOCKS_BATCH_SIZE: Final[PositiveInt] = 20
DEFAULT_UPDATES_TASK_BATCH_SIZE: Final[PositiveInt] = 5

# Get more logs
logging.basicConfig(level=logging.DEBUG)

# Set the garbage collection debugging flags. Debugging information will be written to sys.stderr. See below for a list of debugging flags which can be combined using bit operations to control debugging.
gc.set_debug(gc.DEBUG_UNCOLLECTABLE | gc.DEBUG_SAVEALL)


def total_milliseconds(td: timedelta) -> int:
    return int(td.total_seconds() * 1_000)


VESPA_MAX_LIMIT: int = 50_000
# Vespa's own default is 500ms [1]
#
# [1] https://vespa-engine.github.io/pyvespa/query.html#error-handling
VESPA_DEFAULT_TIMEOUT_MS: int = total_milliseconds(timedelta(milliseconds=500))
VESPA_MAX_TIMEOUT_MS: int = total_milliseconds(timedelta(minutes=5))
# The maximum number of elements to use in equivalent operator of a vespa yql query.
VESPA_MAX_EQUIV_ELEMENTS_IN_QUERY: int = 1_000

# The "parent" AKA the higher level flows that do multiple things.
PARENT_TIMEOUT_S: int = int(timedelta(hours=4).total_seconds())

# A continuation token used by vespa to enable pagination over query results
ContinuationToken = NewType("ContinuationToken", str)


class S3Accessor(BaseModel):
    """Representing S3 paths and prefixes for accessing documents."""

    paths: Sequence[str] | None = None
    prefixes: Sequence[str] | None = None

    def __str__(self) -> str:
        """String representation of the S3Accessor for logging"""
        prefix_count = len(self.prefixes) if self.prefixes else 0
        path_count = len(self.paths) if self.paths else 0
        return f"(prefixes={prefix_count}, paths={path_count})"

    def __repr__(self) -> str:
        """String representation of the S3Accessor for logging"""
        return self.__str__()


# AKA LabelledPassage
# Example: 18593
TextBlockId = NewType("TextBlockId", str)
SpanId = NewType("SpanId", str)
# The ID used in Vespa, that we don't keep in our models in the CPR
# SDK, that is in a Hit.
# Example: id:doc_search:document_passage::UNFCCC.party.1062.0.18593
VespaHitId = NewType("VespaHitId", str)
# The same as above, but without the schema
# Example: UNFCCC.party.1062.0.18593
VespaDataId = NewType("VespaDataId", str)


class Operation(Enum):
    """The kind of operation to take as far as creates, removes, and updates."""

    DEINDEX = "deindex"


def vespa_retry(
    max_attempts: int = 3,
    wait_seconds: int = 2,
    exception_types: tuple[type[Exception], ...] = (QueryError, VespaError),
) -> Callable:
    """Template for retries, use as a decorator."""

    return tenacity.retry(
        retry=tenacity.retry_if_exception_type(exception_types),
        stop=tenacity.stop_after_attempt(max_attempts),
        wait=tenacity.wait_fixed(wait_seconds),
        before_sleep=lambda retry_state: print(
            f"Retrying after error. Attempt {retry_state.attempt_number} of {max_attempts}"
        ),
        after=tenacity.after_log(
            logger=logging.getLogger(f"{__name__}.vespa_retry"), log_level=logging.INFO
        ),
        reraise=True,
    )


def get_vespa_search_adapter_from_aws_secrets(
    cert_dir: str,
    vespa_instance_url_param_name: str = "VESPA_INSTANCE_URL",
    vespa_public_cert_param_name: str = "VESPA_PUBLIC_CERT_READ",
    vespa_private_key_param_name: str = "VESPA_PRIVATE_KEY_READ",
) -> VespaSearchAdapter:
    """
    Get a VespaSearchAdapter instance by retrieving secrets from AWS Secrets Manager.

    We then save the secrets to local files in the cert_dir directory and instantiate
    the VespaSearchAdapter.
    """
    cert_dir_path = Path(cert_dir)
    if not cert_dir_path.exists():
        raise FileNotFoundError(f"Certificate directory does not exist: {cert_dir}")

    vespa_instance_url = get_aws_ssm_param(vespa_instance_url_param_name)
    vespa_public_cert_encoded = get_aws_ssm_param(vespa_public_cert_param_name)
    vespa_private_key_encoded = get_aws_ssm_param(vespa_private_key_param_name)

    vespa_public_cert = base64.b64decode(vespa_public_cert_encoded).decode("utf-8")
    vespa_private_key = base64.b64decode(vespa_private_key_encoded).decode("utf-8")

    cert_path = cert_dir_path / "cert.pem"
    key_path = cert_dir_path / "key.pem"

    with open(cert_path, "w", encoding="utf-8") as f:
        f.write(vespa_public_cert)

    with open(key_path, "w", encoding="utf-8") as f:
        f.write(vespa_private_key)

    return VespaSearchAdapter(
        instance_url=vespa_instance_url,
        cert_directory=str(cert_dir_path),
    )


def s3_object_write_text(s3_uri: str, text: str) -> None:
    """Write text content to an S3 object."""
    # Parse the S3 URI
    s3_path: Path = Path(s3_uri)
    if len(s3_path.parts) < 3:
        raise ValueError(f"Invalid S3 path: {s3_path}")

    bucket: str = s3_path.parts[1]
    key = str(Path(*s3_path.parts[2:]))

    # Create BytesIO buffer with the text content
    body = BytesIO(text.encode("utf-8"))

    # Upload to S3
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")


async def s3_object_write_text_async(s3: S3Client, s3_uri: S3Uri, text: str) -> None:
    """Put an object in S3, async."""
    body = BytesIO(text.encode("utf-8"))
    await s3.put_object(
        Bucket=s3_uri.bucket,
        Key=s3_uri.key,
        Body=body,
        ContentType="application/json",
    )


async def s3_copy_file(s3: S3Client, source: S3Uri, target: S3Uri) -> None:
    """Copy a file from one S3 location to another."""
    await s3.copy_object(
        Bucket=source.bucket,
        CopySource=source.uri,
        Key=target.key,
    )


def s3_obj_generator_from_s3_prefixes(
    s3_prefixes: Sequence[str],
) -> Generator[
    DocumentImporter,
    None,
    None,
]:
    """Return a generator that yields keys from a list of S3 prefixes."""
    for s3_prefix in s3_prefixes:
        try:
            # E.g. Path("s3://bucket/prefix/file.json").parts[1] == "bucket"
            bucket = Path(s3_prefix).parts[1]
            object_keys = _get_s3_keys_with_prefix(s3_prefix=s3_prefix)
            for key in object_keys:
                stem = DocumentStem(Path(key).stem)
                key = DocumentObjectUri(os.path.join("s3://", bucket, key))

                yield DocumentImporter((stem, key))
        except Exception as e:
            print(
                f"failed to yield from S3 prefix. Error: {str(e)}",
            )
            continue


def s3_obj_generator_from_s3_paths(
    s3_paths: Sequence[str],
) -> Generator[
    DocumentImporter,
    None,
    None,
]:
    """
    Return a generator that yields keys from a Sequence of S3 paths.

    We extract the key from the S3 path by removing the first two
    elements in the path.

    E.g. "s3://bucket/prefix/file.json" -> "prefix/file.json"
    """
    for s3_path in s3_paths:
        try:
            stem = DocumentStem(Path(s3_path).stem)
            uri = DocumentObjectUri(s3_path)
            yield DocumentImporter((stem, uri))
        except Exception as e:
            print(
                f"failed to yield from S3 path. Error: {str(e)}",
            )
            continue


def s3_obj_generator(
    s3_prefixes: Sequence[str] | None,
    s3_paths: Sequence[str] | None,
) -> Generator[
    DocumentImporter,
    None,
    None,
]:
    """
    Return a generator that returns S3 objects for each path or prefix.

    These will be for each output from the inference stage, of
    labelled passages.
    """
    match (s3_prefixes, s3_paths):
        case (list(), list()):
            raise ValueError(
                "Either s3_prefixes or s3_paths must be provided, not both."
            )
        case (list(), None):
            print("S3 object generator: prefixes")
            return s3_obj_generator_from_s3_prefixes(s3_prefixes=s3_prefixes)
        case (None, list()):
            print("S3 object generator: paths")
            return s3_obj_generator_from_s3_paths(s3_paths=s3_paths)
        case (None, None):
            raise ValueError("Either s3_prefix or s3_paths must be provided.")
        case _:
            raise ValueError("Invalid combination of s3_prefixes and s3_paths.")


def s3_paths_or_s3_prefixes(
    classifier_specs: Sequence[ClassifierSpec] | None,
    document_ids: Sequence[DocumentImportId] | None,
    cache_bucket: str,
    prefix: str,
) -> S3Accessor:
    """
    Return the paths or prefix for the documents and classifiers.

    - s3_prefix: The S3 prefix (directory) to yield objects from. Example:
      "s3://cpr-sandbox-data-pipeline-cache/labelled_passages"
    - s3_paths: A list of S3 object keys to yield objects from. Example:
      [
        "s3://cpr-sandbox-data-pipeline-cache/labelled_passages/Q787/v4/CCLW.executive.1813.2418.json",
        "s3://cpr-sandbox-data-pipeline-cache/labelled_passages/Q787/v4/CCLW.legislative.10695.6015.json",
      ]
    """
    match (classifier_specs, document_ids):
        case (None, None):
            # Run on all documents, regardless of classifier
            print("run on all documents, regardless of classifier")
            s3_prefix: str = "s3://" + os.path.join(
                cache_bucket,
                prefix,
            )
            return S3Accessor(paths=None, prefixes=[s3_prefix])

        case (list(), None):
            # Run on all documents, for the specified classifier
            print("run on all documents, for the specified classifier")
            s3_prefixes = [
                "s3://"
                + os.path.join(
                    cache_bucket,
                    prefix,
                    classifier_spec.name,
                    classifier_spec.alias,
                )
                for classifier_spec in classifier_specs
            ]
            return S3Accessor(paths=None, prefixes=s3_prefixes)

        case (list(), list()):
            # Run on specified documents, for the specified classifier
            print("run on specified documents, for the specified classifier")

            document_paths = get_labelled_passage_paths(
                document_ids=document_ids,
                classifier_specs=classifier_specs,
                cache_bucket=cache_bucket,
                labelled_passages_prefix=prefix,
            )

            print(
                f"Identified {len(document_paths)} documents to process from {len(document_ids)} document IDs"
            )
            return S3Accessor(paths=document_paths, prefixes=None)

        case (None, list()):
            raise ValueError(
                "if document IDs are specified, a classifier "
                "specifcation must also be specified, since they're "
                "namespaced by classifiers (e.g. "
                "`s3://cpr-sandbox-data-pipeline-cache/labelled_passages/Q787/"
                "v4/CCLW.legislative.10695.6015.json`)"
            )
        case _:
            raise ValueError(
                "Invalid combination of classifier_specs and document_ids."
            )


def load_labelled_passages_by_uri(
    document_object_uri: DocumentObjectUri,
) -> list[LabelledPassage]:
    """Load and transforms the S3 object's body into LabelledPassages objects."""
    object_json = json.loads(_s3_object_read_text(s3_path=document_object_uri))
    if len(object_json) == 0:
        return []

    # We had a window where we hadn't serialised the labelled
    # passages correctly, and needed this special handling.
    #
    # This has now been fixed[1], and in the near future this can be removed.
    #
    # [1] https://linear.app/climate-policy-radar/issue/PLA-505/labelled-passage-serialisation-varies-in-format-and-should-be-the-same
    if isinstance(object_json[0], str):
        object_json = [json.loads(labelled_passage) for labelled_passage in object_json]

    return [LabelledPassage(**labelled_passage) for labelled_passage in object_json]


def get_model_from_span(span: Span) -> str:
    """
    Get the model used to label the span.

    Labellers are stored in a list, these can contain many labellers as seen in the
    example below, referring to human and machine annotators.

    [
        "alice",
        "bob",
        "68edec6f-fe74-413d-9cf1-39b1c3dad2c0",
        'KeywordClassifier("extreme weather")',
    ]

    In the context of inference the labellers array should only hold the model used to
    label the span as seen in the example below.

    [
        'KeywordClassifier("extreme weather")',
    ]
    """
    if len(span.labellers) != 1:
        raise ValueError(
            f"Span has more than one labeller. Expected 1, got {len(span.labellers)}."
        )
    return span.labellers[0]


def get_parent_concepts_from_concept(
    concept: Concept,
) -> tuple[list[dict], str]:
    """
    Extract parent concepts from a Concept object.

    Currently we pull the name from the Classifier used to label the passage, this
    doesn't hold the concept id. This is a temporary solution that is not desirable as
    the relationship between concepts can change frequently and thus shouldn't be
    coupled with inference.
    """
    parent_concepts = [
        {"id": subconcept, "name": ""} for subconcept in concept.subconcept_of
    ]
    parent_concept_ids_flat = (
        ",".join([parent_concept["id"] for parent_concept in parent_concepts]) + ","
    )

    return parent_concepts, parent_concept_ids_flat


def convert_labelled_passage_to_concepts(
    labelled_passage: LabelledPassage,
) -> list[VespaConcept]:
    """
    Convert a labelled passage to a list of VespaConcept objects and their text block ID.

    The labelled passage contains a list of spans relating to concepts
    that we must convert to VespaConcept objects.
    """
    concepts: list[VespaConcept] = []
    concept_json: Union[dict, None] = labelled_passage.metadata.get("concept")

    if not concept_json and not labelled_passage.spans:
        return concepts

    if not concept_json and labelled_passage.spans:
        print(
            "We have spans but no concept metadata for "
            f"labelled passage {labelled_passage.id}"
        )
        raise ValueError(
            "We have spans but no concept metadata.",
        )

    # The concept used to label the passage holds some information on the parent
    # concepts and thus this is being used as a temporary solution for providing
    # the relationship between concepts. This has the downside that it ties a
    # labelled passage to a particular concept when in fact the Spans that a
    # labelled passage has can be labelled by multiple concepts.
    concept = Concept.model_validate(concept_json)
    parent_concepts, parent_concept_ids_flat = get_parent_concepts_from_concept(
        concept=concept
    )

    # This expands the list from `n` for `LabelledPassages` to `n` for `Spans`
    for span_idx, span in enumerate(labelled_passage.spans):
        if span.concept_id is None:
            # Include the Span index since Span's don't have IDs
            print(
                f"span concept ID is missing: LabelledPassage.id={labelled_passage.id}, Span index={span_idx}"
            )
            continue

        if not span.timestamps:
            print(
                f"span timestamps are missing: LabelledPassage.id={labelled_passage.id}, Span index={span_idx}"
            )
            continue

        concepts.append(
            VespaConcept(
                id=span.concept_id,
                name=concept.preferred_label,
                parent_concepts=parent_concepts,
                parent_concept_ids_flat=parent_concept_ids_flat,
                model=get_model_from_span(span),
                end=span.end_index,
                start=span.start_index,
                # These timestamps _should_ all be the same,
                # but just in case, take the latest.
                timestamp=max(span.timestamps),
            )
        )

    return concepts


@vespa_retry()
def get_document_from_vespa(
    document_import_id: DocumentImportId,
    vespa_search_adapter: VespaSearchAdapter,
) -> tuple[VespaHitId, VespaDocument]:
    """Retrieve a passage for a document in Vespa."""
    logger = get_logger()

    logger.info(f"Getting document from Vespa: `{document_import_id}`")

    condition = qb.QueryField("document_import_id").contains(document_import_id)

    yql = (
        qb.select("*")  # pyright: ignore[reportAttributeAccessIssue]
        .from_("family_document")
        .where(condition)
    )

    vespa_query_response: VespaQueryResponse = vespa_search_adapter.client.query(
        yql=yql
    )

    if not vespa_query_response.is_successful():
        raise QueryError(vespa_query_response.get_status_code())
    if len(vespa_query_response.hits) != 1:
        raise ValueError(
            f"Expected 1 document `{document_import_id}`, got {len(vespa_query_response.hits)}"
        )

    logger.info(
        (
            f"Vespa search response for document: {document_import_id} "
            f"with {len(vespa_query_response.hits)} hits"
        )
    )

    hit = vespa_query_response.hits[0]
    document_id = hit["id"]
    document = VespaDocument.model_validate(hit["fields"])

    return document_id, document


@vespa_retry()
def get_document_passage_from_vespa(
    text_block_id: TextBlockId,
    document_import_id: DocumentImportId,
    vespa_search_adapter: VespaSearchAdapter,
) -> tuple[VespaHitId, VespaPassage]:
    """Retrieve a passage for a document in Vespa."""
    logger = get_logger()

    logger.info(
        f"Getting document passage from Vespa: {document_import_id}, text block: {text_block_id}"
    )

    condition = qb.QueryField("family_document_ref").contains(
        f"id:doc_search:family_document::{document_import_id}"
    ) & qb.QueryField("text_block_id").contains(text_block_id)

    yql = qb.select("*").from_("document_passage").where(condition)

    vespa_query_response: VespaQueryResponse = vespa_search_adapter.client.query(
        yql=yql
    )

    if not vespa_query_response.is_successful():
        raise QueryError(vespa_query_response.get_status_code())
    if len(vespa_query_response.hits) != 1:
        raise ValueError(
            f"Expected 1 document passage for text block `{text_block_id}`, got {len(vespa_query_response.hits)}"
        )

    logger.info(
        (
            f"Vespa search response for document: {document_import_id} "
            f"with {len(vespa_query_response.hits)} hits"
        )
    )

    hit = vespa_query_response.hits[0]
    passage_id = hit["id"]
    passage = VespaPassage.model_validate(hit["fields"])

    return passage_id, passage


def get_continuation_tokens_from_query_response(
    vespa_query_response: VespaQueryResponse,
) -> list[ContinuationToken] | None:
    """
    Retrieve continuation tokens from the query response if it exists.

    Continuation tokens can occur at the top level, e.g. under `"children"`, or deeper
    within the nested structure. We take the continuation tokens deeper in the nested
    structure that exist for each of the hits in the group.
    """

    continuation_tokens = []

    vespa_query_response_root = vespa_query_response.json["root"]
    group_hits = dig(vespa_query_response_root, "children", 0, "children", default=[])
    for hit in group_hits:
        hit_continuation_token = dig(hit, "continuation", "next", default=None)
        if hit_continuation_token:
            continuation_tokens.append(hit_continuation_token)
    return continuation_tokens or None


def get_vespa_passages_from_query_response(
    vespa_query_response: VespaQueryResponse,
) -> dict[TextBlockId, tuple[VespaHitId, VespaPassage]]:
    """Retrieve the passages from the query response."""

    vespa_query_response_root = vespa_query_response.json["root"]
    passages_root = dig(
        vespa_query_response_root, "children", 0, "children", 0, "children", default=[]
    )
    passage_hits = [
        dig(passage_root, "children", 0, "children", 0)
        for passage_root in passages_root
    ]

    vespa_passages: dict[TextBlockId, tuple[VespaHitId, VespaPassage]] = {}
    for passage in passage_hits:
        fields = passage.get("fields")
        if not fields:
            raise ValueError(
                f"Vespa passage with no 'fields': {passage}, "
                f"sample of passage hits: {passage_hits[:5]}"
            )

        text_block_id = fields.get("text_block_id")
        if not text_block_id:
            raise ValueError(
                f"Vespa passage with no 'text_block_id' in passage: {passage}, "
                f"sample of passage hits: {passage_hits[:5]}"
            )

        text_block_id = TextBlockId(text_block_id)
        passage_id = VespaHitId(passage.get("id"))
        vespa_pasage = VespaPassage.model_validate(fields)

        vespa_passages[text_block_id] = (
            passage_id,
            vespa_pasage,
        )

    return vespa_passages


@vespa_retry(
    max_attempts=2,
    exception_types=(
        ValueError,
        QueryError,
    ),
)
async def make_query_and_extract_passages(
    vespa_connection_pool: VespaAsync,
    query: qb.Query,
    query_profile: str,
) -> tuple[VespaQueryResponse, dict[TextBlockId, tuple[VespaHitId, VespaPassage]]]:
    """Make the query and extract the passages."""

    vespa_query_response: VespaQueryResponse = await vespa_connection_pool.query(
        yql=query,
        queryProfile=query_profile,
    )

    if not vespa_query_response.is_successful():
        raise QueryError(vespa_query_response.get_status_code())

    return vespa_query_response, get_vespa_passages_from_query_response(
        vespa_query_response
    )


async def get_document_passages_from_vespa__generator(
    document_import_id: DocumentImportId,
    vespa_connection_pool: VespaAsync,
    continuation_tokens: list[ContinuationToken] | None = [],
    grouping_max: int = 5_000,
    grouping_precision: int = 100_000,
    query_profile: str = "default",
) -> AsyncGenerator[dict[TextBlockId, tuple[VespaHitId, VespaPassage]], None]:
    """
    An async generator of vespa passages using continuation tokens to paginate.

    Continuation tokens are opaque objects that are used to move through the grouping
    step of a query to facilitate pagination over results.
    - https://docs.vespa.ai/en/reference/grouping-syntax.html?mode=cloud#continuations

    params:
    - document_import_id: The import id to filter on passages in vespa with.
    - vespa_connection_pool: The vespa connection pool to use as the query client.
    - continuation_tokens: The tokens used to paginate over the vespa hits.
    - grouping_max: The maximum amount of grouping subquery hits to return at once.
    - grouping_precision: How much do we want to value accuracy over bandiwdth.
    - query_profile: The query profile to use for the query. This is defined in the
        search/query-profiles/ subdirectory of the application package for vespa.
    """

    conditions = qb.QueryField("document_import_id").contains(document_import_id)

    # Group the results of the select query by text_block_id in to groups of
    # grouping_max size. For each of the results in the group we output the summary of
    # the passage in vespa.
    # - https://docs.vespa.ai/en/grouping.html?mode=cloud
    grouping = G.all(
        G.group("text_block_id"),
        G.max(grouping_max),
        G.precision(grouping_precision),
        G.each(G.each(G.output(G.summary()))),
    )

    tokens = continuation_tokens or []

    while tokens is not None:
        query: qb.Query = (
            qb.select("*")  # type: ignore
            .from_(
                Schema(name="document_passage", document=Document()),
            )
            .where(conditions)
            .set_limit(0)
            .groupby(grouping, continuations=tokens)
        )

        vespa_query_response, vespa_passages = await make_query_and_extract_passages(
            vespa_connection_pool=vespa_connection_pool,
            query=query,
            query_profile=query_profile,
        )

        if vespa_passages:
            yield vespa_passages

        tokens: list[ContinuationToken] | None = (
            get_continuation_tokens_from_query_response(vespa_query_response)
        )


@vespa_retry()
async def get_document_passages_from_vespa(
    document_import_id: DocumentImportId,
    text_blocks_ids: list[TextBlockId] | None,
    vespa_connection_pool: VespaAsync,
) -> list[tuple[VespaHitId, VespaPassage]]:
    """Retrieve some or all passages for a document in Vespa."""
    print(f"Getting document passages from Vespa: {document_import_id}")

    id = FamilyDocumentID(id=document_import_id)

    family_document_ref: qb.QueryField = qb.QueryField("family_document_ref")

    conditions = family_document_ref.contains(str(id))

    # Possibly don't bother even going to Vespa
    if text_blocks_ids is not None and len(text_blocks_ids) == 0:
        return []

    if text_blocks_ids is not None:
        text_blocks_ids_n: PositiveInt = len(text_blocks_ids)

        print(f"{text_blocks_ids_n} text blocks' IDs passed in")

        if text_blocks_ids_n > VESPA_MAX_LIMIT:
            raise ValueError(
                f"{text_blocks_ids_n} text block IDs exceeds {VESPA_MAX_LIMIT}"
            )

        text_block_id: qb.QueryField = qb.QueryField("text_block_id")

        # equiv expects ≥ 2
        if text_blocks_ids_n == 1:
            text_block_id_contains = text_blocks_ids[0]
        else:
            text_block_id_contains = qb.equiv(*text_blocks_ids)

        conditions &= text_block_id.contains(text_block_id_contains)

    text_blocks_ids_n: PositiveInt = (
        VESPA_MAX_LIMIT if text_blocks_ids is None else len(text_blocks_ids)
    )

    timeout_ms: int = max(
        # Consider the overall max timeout
        VESPA_MAX_TIMEOUT_MS,
        # Per every n text block IDs, allow the default timeout
        math.ceil((text_blocks_ids_n / 5_000) * VESPA_DEFAULT_TIMEOUT_MS),
    )

    print(
        f"using timeout of {timeout_ms} milliseconds for {text_blocks_ids_n} text blocks' IDs"
    )

    query: qb.Query = (
        qb.select("*")
        .from_(
            Schema(name="document_passage", document=Document()),
        )
        .where(conditions)
        .set_limit(VESPA_MAX_LIMIT)
        .set_timeout(timeout_ms)
    )

    vespa_query_response: VespaQueryResponse = await vespa_connection_pool.query(
        yql=query
    )

    if not vespa_query_response.is_successful():
        raise QueryError(vespa_query_response.get_status_code())

    # From `.root.fields.totalCount`
    total_count: NonNegativeInt = vespa_query_response.number_documents_retrieved

    print(
        (
            f"Vespa search response for document: {document_import_id} "
            f"with {len(vespa_query_response.hits)} hits, "
            f"limit {VESPA_MAX_LIMIT}, and total count {total_count}"
        )
    )

    return [
        (passage["id"], VespaPassage.model_validate(passage["fields"]))
        for passage in vespa_query_response.hits
    ]


def get_data_id_from_vespa_hit_id(hit_id: VespaHitId) -> VespaDataId:
    """
    Extract non-schema namespaced ID (last element after "::")

    Example:
    "CCLW.executive.10014.4470.623" from document passage ID like
    "id:doc_search:document_passage::CCLW.executive.10014.4470.623".
    """
    splits = hit_id.split("::")
    if len(splits) != 2:
        raise ValueError(f"received {len(splits)} splits, when expecting 2: {splits}")
    return VespaDataId(splits[1])


def get_text_block_id_from_vespa_data_id(data_id: VespaDataId) -> TextBlockId:
    """
    Extract just the text block ID from a fully qualified Vespa ID for it.

    Example:
    "1273" from Vespa data ID like "CCLW.executive.10014.4470.1273".
    """
    splits = data_id.split(".")
    expected_splits = 5
    if len(splits) != expected_splits:
        raise ValueError(
            f"received {len(splits)} splits, when expecting {expected_splits}: {splits}"
        )
    # Get the last of the splits
    return TextBlockId(splits[-1])


def get_document_passage_from_all_document_passages(
    text_block_id: TextBlockId,
    document_passages: list[tuple[VespaHitId, VespaPassage]],
) -> tuple[VespaDataId, VespaPassage]:
    """
    Get the document passage, if it exists.

    Earlier, we get all of the family document's document passages. We do this once, so there's
    1 big network request to Vespa. We still need to confirm that the document passage exists.
    """
    hit_id_and_passage = next(
        (
            passage
            for passage in document_passages
            if passage[1].text_block_id == text_block_id
        ),
        None,
    )

    if not hit_id_and_passage:
        raise ValueError(
            f"could not found document passage `{text_block_id}` for family document"
        )

    data_id = get_data_id_from_vespa_hit_id(hit_id_and_passage[0])

    return data_id, hit_id_and_passage[1]


class ConceptModel(BaseModel):
    """
    A concept and the model used to identify it.

    This class represents a pairing of a Wikibase concept ID with the name of the model
    used to identify that concept in text. It is hashable to allow use in sets and as
    dictionary keys.

    Attributes:
        wikibase_id: The Wikibase ID of the concept
        model_name: The name of the model used to identify the concept

    Example:
        >>> model = ConceptModel(wikibase_id=WikibaseID("Q123"),
        ...                     model_name='KeywordClassifier("professional services sector")')
        >>> str(model)
        'Q123:KeywordClassifier("professional services sector")'
    """

    wikibase_id: WikibaseID
    model_name: str

    def __str__(self) -> str:
        """
        Convert the ConceptModel to a string in the format 'WIKIBASE_ID:MODEL_NAME'.

        Returns:
            str: String representation in the format 'WIKIBASE_ID:MODEL_NAME'
        """
        return f"{self.wikibase_id}{CONCEPT_COUNT_SEPARATOR}{self.concept_name}"

    def __hash__(self) -> int:
        """
        Generate a hash value for the ConceptModel.

        The hash is based on both the wikibase_id and model_name to ensure
        uniqueness when used in sets or as dictionary keys.

        Returns:
            int: Hash value of the ConceptModel
        """
        return hash((self.wikibase_id, self.model_name))

    def __eq__(self, other: object) -> bool:
        """
        Compare this ConceptModel with another for equality.

        Two ConceptModels are equal if they have the same wikibase_id and model_name.

        Args:
            other: The object to compare with

        Returns:
            bool: True if the objects are equal, False otherwise
        """
        if not isinstance(other, ConceptModel):
            return NotImplemented
        return (
            self.wikibase_id == other.wikibase_id
            and self.model_name == other.model_name
        )

    @property
    def concept_name(self) -> str:
        """
        Extract the concept name from the model name.

        Examples:
            >>> model = ConceptModel(wikibase_id=WikibaseID("Q123"),
            ...                     model_name='KeywordClassifier("professional services sector")')
            >>> model.concept_name
            'professional services sector'
            >>> model2 = ConceptModel(wikibase_id=WikibaseID("Q456"),
            ...                      model_name='LLMClassifier("agriculture")')
            >>> model2.concept_name
            'agriculture'
            >>> model3 = ConceptModel(wikibase_id=WikibaseID("Q789"),
            ...                      model_name='InvalidModelName')
            >>> model3.concept_name  # doctest: +IGNORE_EXCEPTION_DETAIL
            Traceback (most recent call last):
                ...
            ValueError: Could not extract concept name from model name 'InvalidModelName'
        """
        match = re.search(r'"([^"]+)"', self.model_name)
        if match:
            return match.group(1)

        raise ValueError(
            f"Could not extract concept name from model name '{self.model_name}'"
        )


class _ConceptsCombiner(Protocol):
    """Protocol defining interface for combining concepts changes with existing concepts from Vespa."""

    def __call__(
        self,
        passage: VespaPassage,
        concepts: list[VespaConcept],
    ) -> list[dict[str, Any]]: ...


class _UpdateResultCallback(Protocol):
    """Protocol defining interface for handling update operation results from Vespa."""

    def __call__(
        self,
        concepts_counts: Counter[ConceptModel],
        grouped_concepts: dict[TextBlockId, list[VespaConcept]],
        response: VespaResponse,
        text_block_id: TextBlockId,
    ) -> None: ...


class _ConceptsCountsCombiner(Protocol):
    """Protocol defining interface for combined concepts' counts after attempting updates to Vespa."""

    async def __call__(
        self,
        document_importer: DocumentImporter,
        concepts_counts: Counter[ConceptModel],
        cache_bucket: str,
        concepts_counts_prefix: str,
        document_labelled_passages: list[LabelledPassage],
    ) -> None: ...


def op_to_fn(
    operation: Operation,
) -> tuple[
    _ConceptsCombiner,
    _UpdateResultCallback,
    _ConceptsCountsCombiner,
]:
    """Get the appropriate functions to implement an operation"""
    match operation:
        case Operation.DEINDEX:
            return (
                remove_concepts_from_existing_vespa_concepts,
                remove_result_callback,
                update_s3_with_latest_concepts_counts,
            )


@flow(
    # This is the next place, after the top-level (de-)index pipeline
    # where we want to give a timeout, that's smaller than that
    # top-level.
    timeout_seconds=PARENT_TIMEOUT_S,
    log_prints=True,
)
async def run_partial_updates_of_concepts_for_batch(
    documents_batch: Sequence[DocumentImporter],
    documents_batch_num: int,
    cache_bucket: str,
    concepts_counts_prefix: str,
    partial_update_flow: Operation,
    vespa_search_adapter: VespaSearchAdapter | None = None,
    text_update_task_batch_size: int = DEFAULT_TEXT_BLOCKS_BATCH_SIZE,
) -> None:
    """Run partial updates for concepts in a batch of documents."""
    logger = get_run_logger()
    logger.info(
        f"Updating concepts for batch of documents, documents in batch: {len(documents_batch)}. Operation is `{partial_update_flow}`"
    )

    (
        merge_serialise_concepts_cb,
        vespa_response_handler_cb,
        concepts_counts_updater_cb,
    ) = op_to_fn(partial_update_flow)

    if vespa_search_adapter is None:
        temp_dir = tempfile.TemporaryDirectory()

        vespa_search_adapter = get_vespa_search_adapter_from_aws_secrets(
            cert_dir=temp_dir.name,
            vespa_private_key_param_name="VESPA_PRIVATE_KEY_FULL_ACCESS",
            vespa_public_cert_param_name="VESPA_PUBLIC_CERT_FULL_ACCESS",
        )

    failures = 0

    for i, document_importer in enumerate(documents_batch):
        try:
            await run_partial_updates_of_concepts_for_document_passages(
                document_importer=document_importer,
                cache_bucket=cache_bucket,
                concepts_counts_prefix=concepts_counts_prefix,
                merge_serialise_concepts_cb=merge_serialise_concepts_cb,
                vespa_response_handler_cb=vespa_response_handler_cb,
                concepts_counts_updater_cb=concepts_counts_updater_cb,
                vespa_search_adapter=vespa_search_adapter,
                text_update_task_batch_size=text_update_task_batch_size,
            )

            logger.info(f"processed batch documents #{documents_batch_num}")

        except Exception as e:
            document_stem: DocumentStem = documents_batch[i][0]

            logger.error(
                f"failed to process document `{document_stem}`: {e.__str__()}",
            )
            failures += 1

    if failures:
        raise ValueError(f"{failures}/{len(documents_batch)} partial updates failed")

    return None


async def run_partial_updates_of_concepts_for_batch_flow_or_deployment(
    documents_batch: Sequence[DocumentImporter],
    documents_batch_num: int,
    cache_bucket: str,
    concepts_counts_prefix: str,
    aws_env: AwsEnv,
    as_deployment: bool,
    partial_update_flow: Operation,
    semaphore: asyncio.Semaphore,
    vespa_search_adapter: VespaSearchAdapter | None = None,
    text_update_task_batch_size: int = DEFAULT_TEXT_BLOCKS_BATCH_SIZE,
) -> FlowRun | None:
    """Run partial updates for a batch of documents as a sub-flow or deployment."""
    async with semaphore:
        if as_deployment:
            flow_name = function_to_flow_name(run_partial_updates_of_concepts_for_batch)
            deployment_name = generate_deployment_name(
                flow_name=flow_name, aws_env=aws_env
            )

            return await run_deployment(
                name=f"{flow_name}/{deployment_name}",
                parameters={
                    "documents_batch": documents_batch,
                    "documents_batch_num": documents_batch_num,
                    "cache_bucket": cache_bucket,
                    "concepts_counts_prefix": concepts_counts_prefix,
                    "partial_update_flow": partial_update_flow,
                    "text_update_task_batch_size": text_update_task_batch_size,
                },
                # Rely on the flow's own timeout
                timeout=None,
            )

        return await run_partial_updates_of_concepts_for_batch(
            documents_batch=documents_batch,
            documents_batch_num=documents_batch_num,
            cache_bucket=cache_bucket,
            concepts_counts_prefix=concepts_counts_prefix,
            partial_update_flow=partial_update_flow,
            vespa_search_adapter=vespa_search_adapter,
            text_update_task_batch_size=text_update_task_batch_size,
        )


@AsyncProfiler(printer=print)
async def updates_by_s3(
    aws_env: AwsEnv,
    cache_bucket: str,
    concepts_counts_prefix: str,
    partial_update_flow: Operation,
    s3_prefixes: list[str] | None = None,
    s3_paths: list[str] | None = None,
    batch_size: int = DEFAULT_DOCUMENTS_BATCH_SIZE,
    updates_task_batch_size: int = DEFAULT_UPDATES_TASK_BATCH_SIZE,
    text_update_task_batch_size: int = DEFAULT_TEXT_BLOCKS_BATCH_SIZE,
    as_deployment: bool = True,
    vespa_search_adapter: VespaSearchAdapter | None = None,
) -> None:
    """
    Asynchronously update de-index concepts from S3 files into Vespa.

    This function retrieves concept documents from files stored in an S3 path and
    de-indexes them in a Vespa instance. The name of each file in the specified S3 path is
    expected to represent the document's import ID.

    When `s3_prefix` is provided, the function will use all files within that S3
    prefix (directory). When `s3_paths` is provided, the function will use only the
    files specified in the list of S3 object keys. If both are provided `s3_paths` will
    be used.

    Assumptions:
    - The S3 file names represent document import IDs.

    params:
    - s3_prefix: The S3 prefix (directory) to yield objects from.
        E.g. "s3://bucket/prefix/"
    - s3_paths: A list of S3 object keys to yield objects from.
        E.g. {"s3://bucket/prefix/file1.json", "s3://bucket/prefix/file2.json"}
    """
    logger = get_run_logger()

    failures = 0

    logger.info("Getting S3 object generator")
    documents_generator = s3_obj_generator(s3_prefixes, s3_paths)
    documents_batches = iterate_batch(documents_generator, batch_size=batch_size)

    semaphore = asyncio.Semaphore(text_update_task_batch_size)

    updates_tasks = []

    for documents_batch_num, documents_batch in enumerate(documents_batches, start=1):
        updates_tasks.append(
            run_partial_updates_of_concepts_for_batch_flow_or_deployment(
                documents_batch=documents_batch,
                documents_batch_num=documents_batch_num,
                cache_bucket=cache_bucket,
                concepts_counts_prefix=concepts_counts_prefix,
                aws_env=aws_env,
                as_deployment=as_deployment,
                partial_update_flow=partial_update_flow,
                vespa_search_adapter=vespa_search_adapter,
                text_update_task_batch_size=text_update_task_batch_size,
                semaphore=semaphore,
            )
        )

    logger.info("Gathering updates tasks")
    updates_results: list["FlowRun | BaseException | None"] = await asyncio.gather(
        *updates_tasks, return_exceptions=True
    )
    logger.info("Gathered updates tasks")

    for result in updates_results:
        if result is None:
            pass
        elif isinstance(result, BaseException):
            failures += 1
            logger.error(
                f"failed to process document batch in updates task: {str(result)}",
            )
        elif isinstance(result, FlowRun):
            flow_run: FlowRun = result
            if not flow_run.state:
                failures += 1
                logger.error(
                    f"flow run's state was unknown. Flow run name: `{flow_run.name}`",
                )
            elif flow_run.state.type != StateType.COMPLETED:
                failures += 1
                logger.error(
                    f"flow run's state was not completed. Flow run name: `{flow_run.name}`",
                )
        else:
            failures += 1
            logger.error(
                f"unexpected result type: {type(result)}",
            )

    if failures:
        raise ValueError("there was at least 1 task that failed")


@AsyncProfiler(printer=print)
async def _update_text_block(
    semaphore: asyncio.Semaphore,
    text_block_id: TextBlockId,
    document_passage_id: VespaHitId,
    document_passage: VespaPassage,
    merge_serialise_concepts_cb: _ConceptsCombiner,
    concepts: list[VespaConcept],
    vespa_connection_pool: VespaAsync,
) -> tuple[VespaResponse, TextBlockId, VespaHitId]:
    async with semaphore:
        # Prepare the concepts' counts for Vespa
        serialised_concepts = merge_serialise_concepts_cb(
            document_passage,
            concepts,
        )

        data_id = get_data_id_from_vespa_hit_id(document_passage_id)
        response: VespaResponse = await vespa_connection_pool.update_data(
            schema="document_passage",
            namespace="doc_search",
            data_id=data_id,
            fields={"concepts": serialised_concepts},
        )

        return (
            response,
            text_block_id,
            document_passage_id,
        )


# No timeout set since the caller of this has one.
@flow(
    log_prints=True,
    retries=2,
    retry_delay_seconds=5,
    on_failure=[SlackNotify.message],
    on_crashed=[SlackNotify.message],
)
async def run_partial_updates_of_concepts_for_document_passages(
    document_importer: DocumentImporter,
    cache_bucket: str,
    concepts_counts_prefix: str,
    # The different callbacks don't currently have type hints since when they were added,
    # there were issues with the Prefect `@flow` decorator
    # How to merge concepts for the document passage pre- and post-fetching
    merge_serialise_concepts_cb,
    # What to do with the response, in particular with our failures and concepts counts tracking
    vespa_response_handler_cb,
    # The final effect of recording the change in concepts counts to an artifact
    concepts_counts_updater_cb,
    vespa_search_adapter: VespaSearchAdapter | None = None,
    text_update_task_batch_size: int = DEFAULT_TEXT_BLOCKS_BATCH_SIZE,
) -> Counter[ConceptModel]:
    """
    Run partial update for VespaConcepts on text blocks for a document.

    This is done in the document_passage index.
    """
    logger = get_run_logger()

    # --------------------------------------------------------------------------
    # Load and transform the inference results from S3
    # --------------------------------------------------------------------------

    logger.info("loading S3 labelled passages")
    document_labelled_passages = load_labelled_passages_by_uri(document_importer[1])

    logger.info("converting labelled passages to Vespa concepts")
    grouped_concepts: dict[TextBlockId, list[VespaConcept]] = {
        TextBlockId(labelled_passage.id): convert_labelled_passage_to_concepts(
            labelled_passage
        )
        for labelled_passage in document_labelled_passages
    }

    grouped_concepts_n = len(grouped_concepts)
    logger.info(f"starting partial updates for {grouped_concepts_n} grouped concepts")

    document_import_id = remove_translated_suffix(document_importer[0])

    if vespa_search_adapter is None:
        temp_dir = tempfile.TemporaryDirectory()

        vespa_search_adapter = get_vespa_search_adapter_from_aws_secrets(
            cert_dir=temp_dir.name,
            vespa_private_key_param_name="VESPA_PRIVATE_KEY_FULL_ACCESS",
            vespa_public_cert_param_name="VESPA_PUBLIC_CERT_FULL_ACCESS",
        )

    logger.info("creating Vespa connection pool")
    async with (
        vespa_search_adapter.client.asyncio(  # pyright: ignore[reportOptionalMemberAccess]
            connections=DEFAULT_DOCUMENTS_BATCH_SIZE,  # How many tasks to have running at once
            timeout=httpx.Timeout(VESPA_MAX_TIMEOUT_MS / 1_000),  # Seconds
        ) as vespa_connection_pool
    ):
        # -----------------------------------------------------------------------
        # Get the document passages from Vespa, so we know the latest state
        # -----------------------------------------------------------------------

        logger.info("Starting getting document passages from Vespa")
        passages_generator = get_document_passages_from_vespa__generator(
            document_import_id=DocumentImportId(document_import_id),
            vespa_connection_pool=vespa_connection_pool,
        )
        logger.info("got document passage from Vespa generator")

        text_blocks: dict[TextBlockId, tuple[VespaHitId, VespaPassage]] = {}
        text_blocks_n = 0

        with Profiler(
            printer=print,
            name="getting document passages from Vespa",
        ):
            async for passage_batch in passages_generator:
                text_blocks.update(passage_batch)
                text_blocks_n = len(text_blocks)
                logger.info(
                    f"No. of text blocks in text blocks dict so far: {text_blocks_n}"
                )

        text_blocks_size_in_bytes = sys.getsizeof(text_blocks)
        text_blocks_size_in_mb = text_blocks_size_in_bytes / (1024 * 1024)

        logger.info(
            "Finished getting document passages from Vespa. Total "
            f"number of text blocks: {text_blocks_n}. "
            f"Memory size (Mb): {text_blocks_size_in_mb}. "
        )

        # It's normal for there to be less text blocks from inference than vespa
        # because inference filters out more types. However it should never be
        # The other way around
        if grouped_concepts_n > text_blocks_n:
            raise ValueError(
                f"There were {grouped_concepts_n} text block ids from the labelled "
                f"passages but {text_blocks_n} document passages were read from "
                f"Vespa for {document_import_id}"
            )

        # -----------------------------------------------------------------------
        # Update each document passage in Vespa
        # -----------------------------------------------------------------------

        # We'll also collate failures, so we make a best attempt to
        # get as much as possible in, but don't pretend everything
        # went okay.
        #
        # Keep them separate, so it's easier to report on them later.
        missing_text_block_ids_from_vespa: list[TextBlockId] = []
        missing_text_block_ids_from_labelled_passages: list[TextBlockId] = []
        updates_exceptions: list[BaseException] = []
        responses_failures: dict[TextBlockId, VespaResponse] = {}
        responses_successes: dict[TextBlockId, VespaResponse] = {}

        # This is the main process of updating each document passage

        logger.info("starting partial updates")

        responses: list[
            tuple[VespaResponse, TextBlockId, VespaHitId] | BaseException
        ] = []

        with Profiler(
            printer=print,
            name="partial updates",
        ):
            semaphore = asyncio.Semaphore(text_update_task_batch_size)

            tasks = []
            for text_block_id, concepts in grouped_concepts.items():
                if text_block_id not in text_blocks:
                    logger.error(
                        f"document passage `{text_block_id}` not found in Vespa for document `{document_import_id}`"
                    )
                    missing_text_block_ids_from_vespa.append(text_block_id)
                    continue

                document_passage_id = text_blocks[text_block_id][0]
                document_passage = text_blocks[text_block_id][1]

                tasks.append(
                    _update_text_block(
                        semaphore=semaphore,
                        text_block_id=text_block_id,
                        document_passage_id=document_passage_id,
                        document_passage=document_passage,
                        merge_serialise_concepts_cb=merge_serialise_concepts_cb,
                        concepts=concepts,
                        vespa_connection_pool=vespa_connection_pool,
                    )
                )

            start = time.perf_counter()
            logger.info("starting gathering updates tasks")
            tasks_results = await asyncio.gather(*tasks, return_exceptions=True)
            responses.extend(tasks_results)
            logger.info(
                f"finished gathering updates tasks in {time.perf_counter() - start:.2f}s"
            )

            for response in responses:
                if isinstance(response, BaseException):
                    logger.error(f"text block update for failed: {str(response)}")
                    updates_exceptions.append(response)
                else:
                    response, text_block_id, document_passage_id = response

                    if not response.is_successful():
                        response_json = json.dumps(response.get_json())
                        logger.error(
                            f"document passage update failed. {document_passage_id=}, {response_json=}"
                        )
                        responses_failures[text_block_id] = response
                    else:
                        responses_successes[text_block_id] = response

        if missing_text_block_ids_from_vespa:
            logger.error(
                f"the following text blocks (IDs) were missing from Vespa: {','.join(missing_text_block_ids_from_vespa)}"
            )

        # Map over the results and combine/calculate the
        # resultant concepts' counts.
        concepts_counts: Counter[ConceptModel] = Counter()

        logger.info("starting combining responses from Vespa")

        with Profiler(
            printer=print,
            name="handling Vespa responses",
        ):
            # Bring them together, since the callbacks vary in their
            # behaviour on if it was a success or not.
            for text_block_id, response in (
                responses_failures | responses_successes
            ).items():
                try:
                    vespa_response_handler_cb(
                        concepts_counts,
                        grouped_concepts,
                        response,
                        text_block_id,
                    )
                except ValueError as e:
                    missing_text_block_ids_from_labelled_passages.append(
                        TextBlockId(str(e))
                    )

        if missing_text_block_ids_from_labelled_passages:
            logger.error(
                f"the following text blocks (IDs) were missing from labelled passages: {','.join(missing_text_block_ids_from_labelled_passages)}"
            )

        # Write the concepts' counts for S3
        logger.info("starting concepts counting and writing to store")
        start = time.perf_counter()

        try:
            await concepts_counts_updater_cb(
                document_importer=document_importer,
                concepts_counts=concepts_counts,
                cache_bucket=cache_bucket,
                concepts_counts_prefix=concepts_counts_prefix,
                document_labelled_passages=document_labelled_passages,
            )
        except Exception as e:
            logger.error("failed to write concepts counts to S3")
            raise e

        logger.info(
            f"finished concepts counting and writing to store in {time.perf_counter() - start:.2f}s"
        )

        if (
            missing_text_block_ids_from_vespa
            or missing_text_block_ids_from_labelled_passages
            or updates_exceptions
            or responses_failures
        ):
            missing_text_block_ids_from_vespa_n = len(missing_text_block_ids_from_vespa)
            missing_text_block_ids_from_labelled_passages_n = len(
                missing_text_block_ids_from_labelled_passages
            )
            updates_exceptions_n = len(updates_exceptions)
            responses_failures_n = len(responses_failures)

            combined_failures = (
                missing_text_block_ids_from_vespa_n
                + missing_text_block_ids_from_labelled_passages_n
                + updates_exceptions_n
                + responses_failures_n
            )
            raise ValueError(
                f"there were {combined_failures} combined failures. Please "
                "review the logs for specifics. "
                f"{missing_text_block_ids_from_vespa_n} missing text blocks from Vespa, "
                f"{missing_text_block_ids_from_labelled_passages_n} missing text blocks from labelled passages, "
                f"{updates_exceptions_n} updates raised exceptions, "
                f"{responses_failures_n} updates responses were failures"
            )

        # All Vespa updating has finished, return the final concepts' counts
        return concepts_counts


def remove_result_callback(
    concepts_counts: Counter[ConceptModel],
    grouped_concepts: dict[TextBlockId, list[VespaConcept]],
    response: VespaResponse,
    text_block_id: TextBlockId,
) -> None:
    # Update concepts counts
    concepts = grouped_concepts[text_block_id]

    # Example:
    #
    # ..
    # "labellers": [
    #   "KeywordClassifier(\"professional services sector\")"
    # ],
    # ...
    concepts_models = [
        ConceptModel(wikibase_id=WikibaseID(concept.id), model_name=concept.model)
        for concept in concepts
    ]

    # Set 0s in the counter for all seen concepts. This ensures
    # all concepts are represented in the counter even if they're
    # not updated.
    for concept_model in concepts_models:
        if concept_model not in concepts_counts:
            concepts_counts[concept_model] = 0

    if not response.is_successful():
        # Since we failed to remove them from the spans, make sure
        # they're accounted for as remaining.
        concepts_counts.update(concepts_models)


def remove_concepts_from_existing_vespa_concepts(
    passage: VespaPassage,
    concepts: list[VespaConcept],
) -> list[dict[str, Any]]:
    """
    Update a passage's concepts with the updated/removed concepts.

    During the update we remove all the old concepts related to a model. This is as it
    was decided that holding out dated concepts/spans on the passage in Vespa for a
    model is not useful.

    It is also, not possible to duplicate a Concept object in the concepts array as we
    are removing all instances where the model is the same.
    """
    # Get the models to remove
    concepts_to_remove__models = [concept.model for concept in concepts]

    # It's an optional sequence at the moment, so massage it
    concepts_in_vespa: list[VespaConcept] = (
        list(passage.concepts) if passage.concepts is not None else []
    )

    # We'll be removing all of the listed concepts, so filter them out
    concepts_in_vespa_to_keep = [
        concept
        for concept in concepts_in_vespa
        if concept.model not in concepts_to_remove__models
    ]

    return [concept_.model_dump(mode="json") for concept_ in concepts_in_vespa_to_keep]


async def update_s3_with_latest_concepts_counts(
    document_importer: DocumentImporter,
    concepts_counts: Counter[ConceptModel],
    cache_bucket: str,
    concepts_counts_prefix: str,
    document_labelled_passages: list[LabelledPassage],
) -> None:
    # Ideally, we'd remove the concepts count file entirely, but, we may fail above in updating
    # 1 or more document passages in Vespa, which means that they'd still have the concept present.
    #
    # To avoid a mismatch of the family documents' concepts counts, and what's _still_ reflected on
    # document passages due to failed partial updates, still write an updated concepts counts to
    # S3.
    #
    # However, if we successfully removed all of the concepts from the document passages, then we can
    # delete it. Then, also update the family document's concepts counts to remove it from there.

    # Remove entries with a value of 0 from the counter
    concepts_counts_filtered = Counter(
        {k: v for k, v in concepts_counts.items() if v != 0}
    )

    # If after filtering out, there's no concepts, that means we
    # succeeded in all the partial updates to the document
    # passages.
    if len(concepts_counts_filtered) == 0:
        print("successfully updated all concepts")
        update_s3_with_all_successes(
            document_object_uri=document_importer[1],
            cache_bucket=cache_bucket,
            concepts_counts_prefix=concepts_counts_prefix,
        )
    # We didn't succeed with all, so write the concepts counts still
    else:
        print("only updated some concepts")
        update_s3_with_some_successes(
            document_object_uri=document_importer[1],
            concepts_counts_filtered=concepts_counts_filtered,
            document_labelled_passages=document_labelled_passages,
            cache_bucket=cache_bucket,
            concepts_counts_prefix=concepts_counts_prefix,
        )

    return None


def update_s3_with_all_successes(
    document_object_uri: DocumentObjectUri,
    cache_bucket: str,
    concepts_counts_prefix: str,
) -> None:
    """Clean-up S3 objects for a document's labelled passages and concepts counts."""
    print("updating S3 with all successes")

    s3 = boto3.client("s3")

    s3_uri = Path(document_object_uri)

    # First, delete the concepts counts object
    # Get all parts after the prefix (e.g. "CCLW.executive.1813.2418.json")
    key_parts = "/".join(s3_uri.parts[3:])  # Skip s3://bucket/labelled_passages/

    concepts_counts_key = f"{concepts_counts_prefix}/{key_parts}"

    print(
        f"deleting concepts counts from bucket `{cache_bucket}`, key: `{concepts_counts_key}`"
    )
    if not s3_file_exists(bucket_name=cache_bucket, file_key=concepts_counts_key):
        print(
            "planned on deleting concepts counts from bucket: "
            f"`{cache_bucket}`, key: `{concepts_counts_key}`, "
            "but the object doesn't exist"
        )
    else:
        s3.delete_object(Bucket=cache_bucket, Key=concepts_counts_key)

    print("updated S3 with deleted concepts counts")

    # Second, delete the labelled passages
    # Get all parts except for the bucket (e.g. "labelled_passages/Q787/v4/CCLW.executive.1813.2418.json")
    labelled_passages_key = "/".join(s3_uri.parts[2:])  # Skip s3://bucket/

    print(
        f"deleting labelled passages from bucket `{cache_bucket}`, key: `{labelled_passages_key}`"
    )
    if not s3_file_exists(bucket_name=cache_bucket, file_key=labelled_passages_key):
        print(
            "planned on deleting labelled passages from bucket: "
            f"`{cache_bucket}`, key: `{labelled_passages_key}`, "
            "but the object doesn't exist"
        )
    else:
        s3.delete_object(Bucket=cache_bucket, Key=labelled_passages_key)

    print("updated S3 with deleted labelled passages")

    print("updated S3 with all successes")

    return None


def update_s3_with_some_successes(
    document_object_uri: DocumentObjectUri,
    concepts_counts_filtered: Counter[ConceptModel],
    document_labelled_passages: list[LabelledPassage],
    cache_bucket: str,
    concepts_counts_prefix: str,
) -> None:
    print("updating S3 with partial successes")

    # First, update the concepts counts object
    serialised_concepts_counts = serialise_concepts_counts(concepts_counts_filtered)

    s3_uri = Path(document_object_uri)

    # Get all parts after the prefix (e.g. "Q787/v4/CCLW.executive.1813.2418.json")
    key_parts = "/".join(s3_uri.parts[3:])  # Skip s3://bucket/labelled_passages/

    concepts_counts_uri = f"s3://{cache_bucket}/{concepts_counts_prefix}/{key_parts}"

    s3_object_write_text(
        s3_uri=concepts_counts_uri,
        text=serialised_concepts_counts,
    )

    print("updated S3 with updated concepts counts")

    # Second, update the labelled passages
    concept_ids_to_keep: list[WikibaseID] = [
        concept_model.wikibase_id for concept_model in concepts_counts_filtered
    ]

    filtered_labelled_passages: list[LabelledPassage] = []

    for labelled_passage in document_labelled_passages:
        # It doesn't matter if this list is empty, as it
        # emulates an empty result from the inference
        # pipeline.
        updated_spans: list[Span] = [
            span
            for span in labelled_passage.spans
            if span.concept_id in concept_ids_to_keep
        ]

        labelled_passage.spans = updated_spans

        filtered_labelled_passages.append(labelled_passage)

    save_labelled_passages_by_uri(
        document_object_uri=document_object_uri,
        labelled_passages=filtered_labelled_passages,
    )

    print("updated S3 with updated labelled passages")

    print("updated S3 with partial successes")

    return None


def save_labelled_passages_by_uri(
    document_object_uri: DocumentObjectUri,
    labelled_passages: list[LabelledPassage],
) -> None:
    """Save LabelledPassages objects to S3."""
    object_json = json.dumps(
        [labelled_passage.model_dump_json() for labelled_passage in labelled_passages]
    )

    s3_object_write_text(
        s3_uri=document_object_uri,
        text=object_json,
    )


def serialise_concepts_counts(concepts_counts: Counter[ConceptModel]) -> str:
    return json.dumps({str(k): v for k, v in concepts_counts.items()})
