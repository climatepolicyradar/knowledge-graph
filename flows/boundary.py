"""Boundary between Prefect, Vespa, and AWS."""

import base64
import gc
import json
import logging
import math
import re
from collections.abc import AsyncGenerator, Sequence
from datetime import timedelta
from io import BytesIO
from pathlib import Path
from typing import (
    Callable,
    Final,
    NewType,
    TypeVar,
    Union,
)

import tenacity
import vespa.querybuilder as qb
from cpr_sdk.models.search import Concept as VespaConcept
from cpr_sdk.models.search import Document as VespaDocument
from cpr_sdk.models.search import Passage as VespaPassage
from cpr_sdk.s3 import _s3_object_read_text
from cpr_sdk.search_adaptors import VespaSearchAdapter
from cpr_sdk.ssm import get_aws_ssm_param
from cpr_sdk.utils import dig
from prefect.logging import get_logger
from pydantic import BaseModel, NonNegativeInt, PositiveInt
from types_aiobotocore_s3.client import S3Client
from vespa.application import VespaAsync
from vespa.exceptions import VespaError
from vespa.io import VespaQueryResponse
from vespa.package import Document, Schema
from vespa.querybuilder import Grouping as G

from flows.utils import (
    DocumentImportId,
    DocumentObjectUri,
    S3Uri,
)
from src.concept import Concept
from src.exceptions import QueryError
from src.identifiers import FamilyDocumentID, WikibaseID
from src.labelled_passage import LabelledPassage
from src.span import Span

# Provide a generic type to use instead of `Any` for types hints
T = TypeVar("T")

CONCEPT_COUNT_SEPARATOR: Final[str] = ":"
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

    yql = qb.select("*").from_("document_passage").where(condition)  # type: ignore[attr-defined]

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
        if hit_continuation_token := dig(hit, "continuation", "next", default=None):
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
    - grouping_precision: How much do we want to value accuracy over bandwidth.
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
            text_block_id_contains = qb.equiv(*text_blocks_ids)  # type: ignore[attr-defined]

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
        qb.select("*")  # type: ignore[attr-defined]
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

        if match := re.search(r'"([^"]+)"', self.model_name):
            return match.group(1)

        raise ValueError(
            f"Could not extract concept name from model name '{self.model_name}'"
        )
