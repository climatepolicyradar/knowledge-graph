"""Boundary between Prefect, Vespa, and AWS."""

import asyncio
import base64
import contextlib
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Callable, Generator
from enum import Enum
from io import BytesIO
from pathlib import Path
from typing import Any, ContextManager, TypeAlias, Union

import boto3
import vespa.querybuilder as qb
from cpr_sdk.models.search import Concept as VespaConcept
from cpr_sdk.models.search import Document as VespaDocument
from cpr_sdk.models.search import Passage as VespaPassage
from cpr_sdk.s3 import _get_s3_keys_with_prefix, _s3_object_read_text
from cpr_sdk.search_adaptors import VespaSearchAdapter
from cpr_sdk.ssm import get_aws_ssm_param
from prefect import flow, get_run_logger
from prefect.client.schemas.objects import FlowRun, StateType
from prefect.deployments.deployments import run_deployment
from prefect.logging import get_logger
from pydantic import BaseModel
from vespa.io import VespaQueryResponse, VespaResponse

from flows.utils import (
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
from src.exceptions import PartialUpdateError, QueryError
from src.identifiers import WikibaseID
from src.labelled_passage import LabelledPassage
from src.span import Span

HTTP_OK = 200
CONCEPT_COUNT_SEPARATOR: str = ":"
CONCEPTS_COUNTS_PREFIX_DEFAULT: str = "concepts_counts"
DEFAULT_DOCUMENTS_BATCH_SIZE = 500
DEFAULT_UPDATES_TASK_BATCH_SIZE = 20

# Needed to get document passages from Vespa
# Example: CCLW.executive.1813.2418
DocumentImportId: TypeAlias = str
# Needed to load the inference results
# Example: s3://cpr-sandbox-data-pipeline-cache/labelled_passages/Q787/v4/CCLW.executive.1813.2418.json
DocumentObjectUri: TypeAlias = str
DocumentStem: TypeAlias = str
# Passed to a self-sufficient flow run
DocumentImporter: TypeAlias = tuple[DocumentStem, DocumentObjectUri]


class S3Accessor(BaseModel):
    """Representing S3 paths and prefixes for accessing documents."""

    paths: list[str] | None = None
    prefixes: list[str] | None = None


# AKA LabelledPassage
TextBlockId: TypeAlias = str
SpanId: TypeAlias = str
# The ID used in Vespa, that we don't keep in our models in the CPR
# SDK, that is in a Hit.
VespaHitId: TypeAlias = str
# The same as above, but without the schema
VespaDataId: TypeAlias = str


class Operation(Enum):
    """The kind of operation to take as far as creates, removes, and updates."""

    INDEX = "index"
    DEINDEX = "deindex"


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


def s3_obj_generator_from_s3_prefixes(
    s3_prefixes: list[str],
) -> Generator[
    DocumentImporter,
    None,
    None,
]:
    """Return a generator that yields keys from a list of S3 prefixes."""
    logger = get_logger()

    for s3_prefix in s3_prefixes:
        try:
            # E.g. Path("s3://bucket/prefix/file.json").parts[1] == "bucket"
            bucket = Path(s3_prefix).parts[1]
            object_keys = _get_s3_keys_with_prefix(s3_prefix=s3_prefix)
            for key in object_keys:
                stem: DocumentStem = Path(key).stem
                key: DocumentObjectUri = os.path.join("s3://", bucket, key)

                yield stem, key
        except Exception as e:
            logger.error(
                f"failed to yield from S3 prefix. Error: {str(e)}",
            )
            continue


def s3_obj_generator_from_s3_paths(
    s3_paths: list[str],
) -> Generator[
    DocumentImporter,
    None,
    None,
]:
    """
    Return a generator that yields keys from a list of S3 paths.

    We extract the key from the S3 path by removing the first two
    elements in the path.

    E.g. "s3://bucket/prefix/file.json" -> "prefix/file.json"
    """
    logger = get_logger()
    for s3_path in s3_paths:
        try:
            stem: DocumentStem = Path(s3_path).stem
            uri: DocumentObjectUri = s3_path
            yield stem, uri
        except Exception as e:
            logger.error(
                f"failed to yield from S3 path. Error: {str(e)}",
            )
            continue


def s3_obj_generator(
    s3_prefixes: list[str] | None,
    s3_paths: list[str] | None,
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
    logger = get_run_logger()

    match (s3_prefixes, s3_paths):
        case (list(), list()):
            raise ValueError(
                "Either s3_prefixes or s3_paths must be provided, not both."
            )
        case (list(), None):
            logger.info("S3 object generator: prefixes")
            return s3_obj_generator_from_s3_prefixes(s3_prefixes=s3_prefixes)
        case (None, list()):
            logger.info("S3 object generator: paths")
            return s3_obj_generator_from_s3_paths(s3_paths=s3_paths)
        case (None, None):
            raise ValueError("Either s3_prefix or s3_paths must be provided.")


def s3_paths_or_s3_prefixes(
    classifier_specs: list[ClassifierSpec] | None,
    document_ids: list[str] | None,
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
    logger = get_run_logger()

    match (classifier_specs, document_ids):
        case (None, None):
            # Run on all documents, regardless of classifier
            logger.info("run on all documents, regardless of classifier")
            s3_prefix: str = "s3://" + os.path.join(
                cache_bucket,
                prefix,
            )
            return S3Accessor(paths=None, prefixes=[s3_prefix])

        case (list(), None):
            # Run on all documents, for the specified classifier
            logger.info("run on all documents, for the specified classifier")
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
            logger.info("run on specified documents, for the specified classifier")

            document_paths = get_labelled_passage_paths(
                document_ids=document_ids,
                classifier_specs=classifier_specs,
                cache_bucket=cache_bucket,
                labelled_passages_prefix=prefix,
            )

            logger.info(
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


def load_labelled_passages_by_uri(
    document_object_uri: DocumentObjectUri,
) -> list[LabelledPassage]:
    """Load and transforms the S3 object's body into LabelledPassages objects."""
    object_json = json.loads(_s3_object_read_text(s3_path=document_object_uri))
    if len(object_json) == 0:
        return []

    # Currently we _sometimes_ serialise the labelled passages as a
    # JSON list, but the items of the list are a funny serialisation
    # of them, so they're still raw strings, when loaded in.
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
    logger = get_run_logger()

    concepts: list[VespaConcept] = []
    concept_json: Union[dict, None] = labelled_passage.metadata.get("concept")

    if not concept_json and not labelled_passage.spans:
        return concepts

    if not concept_json and labelled_passage.spans:
        logger.error(
            "We have spans but no concept metadata.",
            extra={"labelled_passage_id": labelled_passage.id},
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
            logger.error(
                f"span concept ID is missing: LabelledPassage.id={labelled_passage.id}, Span index={span_idx}"
            )
            continue

        if not span.timestamps:
            logger.error(
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


def get_vespa_search_adapter(
    vespa_search_adapter: VespaSearchAdapter | None,
) -> tuple[
    ContextManager[str] | ContextManager[None],
    VespaSearchAdapter,
]:
    """
    Get a Vespa search adapter, if none provided.

    It uses certs fetched, and then, saved to disk, if none was provided.
    """
    logger = get_run_logger()

    # We want the directory used for the `VespaSearchAdapter` to be
    # automatically cleaned up.
    #
    # To do this, we rely on the `tempfile.TemporaryDirectory`'s behaviour,
    # or, a `contextlib.nullcontext` no-op, if a temporary directory
    # wasn't needed.
    if vespa_search_adapter is None:
        logger.info("no Vespa search adapter, getting it from AWS secrets")
        cm = tempfile.TemporaryDirectory()

        vespa_search_adapter = get_vespa_search_adapter_from_aws_secrets(
            cert_dir=cm.name,  # type: ignore
            vespa_private_key_param_name="VESPA_PRIVATE_KEY_FULL_ACCESS",
            vespa_public_cert_param_name="VESPA_PUBLIC_CERT_FULL_ACCESS",
        )
    else:
        logger.info("Vespa search adapter provided")
        cm = contextlib.nullcontext()

    return cm, vespa_search_adapter


def get_document_from_vespa(
    document_import_id: DocumentImportId,
    vespa_search_adapter: VespaSearchAdapter,
) -> tuple[VespaHitId, VespaDocument]:
    """Retrieve a passage for a document in Vespa."""
    logger = get_logger()

    logger.info(f"Getting document from Vespa: `{document_import_id}`")

    condition = qb.QueryField("document_import_id").contains(document_import_id)

    yql = (
        qb.select("*")  # pyright:ignore[reportAttributeAccessIssue]
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


def get_document_passage_from_vespa(
    text_block_id: str,
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


def get_document_passages_from_vespa(
    document_import_id: DocumentImportId,
    vespa_search_adapter: VespaSearchAdapter,
) -> list[tuple[VespaHitId, VespaPassage]]:
    """
    Retrieve all the passages for a document in Vespa.

    params:
    - document_import_id: The document import id for a unique family document.
    """
    logger = get_logger()

    logger.info(f"Getting document passages from Vespa: {document_import_id}")

    vespa_query_response: VespaQueryResponse = vespa_search_adapter.client.query(
        yql=(
            # trunk-ignore(bandit/B608)
            "select * from document_passage where family_document_ref contains "
            f'"id:doc_search:family_document::{document_import_id}"'
        )
    )

    if (status_code := vespa_query_response.get_status_code()) != HTTP_OK:
        raise QueryError(status_code)

    logger.info(
        (
            f"Vespa search response for document: {document_import_id} "
            f"with {len(vespa_query_response.hits)} hits"
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
    return splits[1]


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


async def partial_update_text_block(
    text_block_id: TextBlockId,
    concepts: list[VespaConcept],  # A possibly empty list
    document_import_id: DocumentImportId,
    vespa_search_adapter: VespaSearchAdapter,
    update_function: Callable[[VespaPassage, list[VespaConcept]], list[dict[str, Any]]],
):
    """Partial update a singular text block and its concepts."""
    document_passage_id, document_passage = get_document_passage_from_vespa(
        text_block_id, document_import_id, vespa_search_adapter
    )

    data_id = get_data_id_from_vespa_hit_id(document_passage_id)

    serialised_concepts = update_function(document_passage, concepts)

    response: VespaResponse = vespa_search_adapter.client.update_data(  # pyright: ignore[reportOptionalMemberAccess]
        schema="document_passage",
        namespace="doc_search",
        data_id=data_id,
        fields={"concepts": serialised_concepts},
    )

    if (status_code := response.get_status_code()) != HTTP_OK:
        raise PartialUpdateError(data_id, status_code)

    return None


def op_to_fn(operation: Operation):
    """Get the appropriate function to implement an operation"""
    match operation:
        case Operation.INDEX:
            return run_partial_updates_of_concepts_for_document_passages__update
        case Operation.DEINDEX:
            return run_partial_updates_of_concepts_for_document_passages__remove


@flow
async def run_partial_updates_of_concepts_for_batch(
    documents_batch: list[DocumentImporter],
    documents_batch_num: int,
    cache_bucket: str,
    concepts_counts_prefix: str,
    partial_update_flow: Operation,
) -> None:
    """Run partial updates for concepts in a batch of documents."""
    logger = get_run_logger()
    logger.info(
        f"Updating concepts for batch of documents, documents in batch: {len(documents_batch)}."
    )

    fn = op_to_fn(partial_update_flow)

    failures = 0

    for i, document_importer in enumerate(documents_batch):
        try:
            await fn(
                document_importer=document_importer,
                cache_bucket=cache_bucket,
                concepts_counts_prefix=concepts_counts_prefix,
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
    documents_batch: list[DocumentImporter],
    documents_batch_num: int,
    cache_bucket: str,
    concepts_counts_prefix: str,
    aws_env: AwsEnv,
    as_deployment: bool,
    partial_update_flow: Operation,
) -> FlowRun | None:
    """Run partial updates for a batch of documents as a sub-flow or deployment."""
    logger = get_run_logger()
    logger.info(
        "Running partial updates of concepts for batch as sub-flow or deployment: "
        f"batch length {len(documents_batch)}, as_deployment: {as_deployment}"
    )

    if as_deployment:
        flow_name = function_to_flow_name(run_partial_updates_of_concepts_for_batch)
        deployment_name = generate_deployment_name(flow_name=flow_name, aws_env=aws_env)

        return await run_deployment(
            name=f"{flow_name}/{deployment_name}",
            parameters={
                "documents_batch": documents_batch,
                "documents_batch_num": documents_batch_num,
                "cache_bucket": cache_bucket,
                "concepts_counts_prefix": concepts_counts_prefix,
                "partial_update_flow": partial_update_flow,
            },
            timeout=3600,
        )

    return await run_partial_updates_of_concepts_for_batch(
        documents_batch=documents_batch,
        documents_batch_num=documents_batch_num,
        cache_bucket=cache_bucket,
        concepts_counts_prefix=concepts_counts_prefix,
        partial_update_flow=partial_update_flow,
    )


@flow
async def updates_by_s3(
    aws_env: AwsEnv,
    cache_bucket: str,
    concepts_counts_prefix: str,
    partial_update_flow: Operation,
    s3_prefixes: list[str] | None = None,
    s3_paths: list[str] | None = None,
    batch_size: int = DEFAULT_DOCUMENTS_BATCH_SIZE,
    updates_task_batch_size: int = DEFAULT_UPDATES_TASK_BATCH_SIZE,
    as_deployment: bool = True,
) -> None:
    """
    Asynchronously update (de-)index concepts from S3 files into Vespa.

    This function retrieves concept documents from files stored in an S3 path and
    (de-)indexes them in a Vespa instance. The name of each file in the specified S3 path is
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
    updates_task_batches = iterate_batch(
        data=documents_batches, batch_size=updates_task_batch_size
    )

    for i, updates_task_batch in enumerate(updates_task_batches, start=1):
        logger.info(f"Processing updates task batch #{i}")

        updates_tasks = [
            run_partial_updates_of_concepts_for_batch_flow_or_deployment(
                documents_batch=documents_batch,
                documents_batch_num=documents_batch_num,
                cache_bucket=cache_bucket,
                concepts_counts_prefix=concepts_counts_prefix,
                aws_env=aws_env,
                as_deployment=as_deployment,
                partial_update_flow=partial_update_flow,
            )
            for documents_batch_num, documents_batch in enumerate(
                updates_task_batch, start=1
            )
        ]

        logger.info(f"Gathering updates tasks for batch #{i}")
        batch_results: list[Any] = await asyncio.gather(
            *updates_tasks, return_exceptions=True
        )
        logger.info(f"Gathered updates tasks for batch #{i}")

        for result in batch_results:
            if isinstance(result, Exception):
                failures += 1
                logger.error(
                    f"failed to process document batch in updates task batch #{i}: {str(result)}",
                )
                continue

            if as_deployment:
                if result is None:
                    continue
            else:
                if isinstance(result, list):
                    for task_result in result:
                        if task_result.type != StateType.COMPLETED:
                            failures += 1
                            logger.error(
                                f"flow run task_result's state was not completed. Flow run name: `{task_result.name}`",
                            )

    if failures:
        raise ValueError("there was at least 1 task that failed")


# Index -------------------------------------------------------------------------


@flow
async def run_partial_updates_of_concepts_for_document_passages__update(
    document_importer: DocumentImporter,
    cache_bucket: str,
    concepts_counts_prefix: str,
    vespa_search_adapter: VespaSearchAdapter | None = None,
) -> Counter[ConceptModel]:
    """
    Run partial update for VespaConcepts on text blocks for a document.

    This is done in the document_passage index.

    Assumptions:

    - The ID field of the VespaConcept object holds the
    context of the text block that it relates to. E.g. the concept ID
    1.10 would relate to the text block ID 10.
    """
    logger = get_run_logger()

    cm, vespa_search_adapter = get_vespa_search_adapter(vespa_search_adapter)

    logger.info("getting S3 labelled passages generator")
    document_labelled_passages = load_labelled_passages_by_uri(document_importer[1])

    with cm:
        logger.info("converting labelled passages to Vespa concepts")
        grouped_concepts: dict[TextBlockId, list[VespaConcept]] = {
            labelled_passage.id: convert_labelled_passage_to_concepts(labelled_passage)
            for labelled_passage in document_labelled_passages
        }

        logger.info(
            f"starting partial updates for {len(grouped_concepts)} grouped concepts"
        )

        batches = iterate_batch(
            list(grouped_concepts.items()),
            batch_size=DEFAULT_DOCUMENTS_BATCH_SIZE,
        )

        concepts_counts: Counter[ConceptModel] = Counter()

        for batch_num, batch in enumerate(batches, start=1):
            logger.info(f"processing partial updates batch {batch_num}")

            document_import_id = remove_translated_suffix(document_importer[0])

            partial_update_tasks = [
                partial_update_text_block(
                    text_block_id=text_block_id,
                    concepts=concepts,
                    document_import_id=document_import_id,
                    vespa_search_adapter=vespa_search_adapter,
                    update_function=update_concepts_on_existing_vespa_concepts,
                )
                for text_block_id, concepts in batch
            ]

            logger.info(f"gathering partial updates tasks for batch {batch_num}")
            results = await asyncio.gather(
                *partial_update_tasks, return_exceptions=True
            )
            logger.info(
                f"gathered partial {len(results)} updates tasks for batch {batch_num}"
            )

            for i, result in enumerate(results):
                text_block_id, concepts = batch[i]

                if isinstance(result, Exception):
                    logger.error(
                        f"failed to do partial update for text block `{text_block_id}`: {str(result)}",
                    )

                    continue

                # Example:
                #
                # ..
                # "labellers": [
                #   "KeywordClassifier(\"professional services sector\")"
                # ],
                # ...
                concepts_models = [
                    ConceptModel(
                        wikibase_id=WikibaseID(concept.id), model_name=concept.model
                    )
                    for concept in concepts
                ]

                concepts_counts.update(concepts_models)

        # Write concepts counts to S3
        try:
            s3_uri = Path(document_importer[1])
            # Get all parts after the prefix (e.g. "Q787/v4/CCLW.executive.1813.2418.json")
            key_parts = "/".join(s3_uri.parts[3:])  # Skip s3:/bucket/labelled_passages/

            # Create new path with concepts_counts_prefix
            concepts_counts_uri = (
                f"s3://{cache_bucket}/{concepts_counts_prefix}/{key_parts}"
            )

            serialised_concepts_counts = json.dumps(
                {str(k): v for k, v in concepts_counts.items()}
            )

            # Write to S3
            s3_object_write_text(
                s3_uri=concepts_counts_uri,
                text=serialised_concepts_counts,
            )
        except Exception as e:
            logger.error(f"Failed to write concepts counts to S3: {str(e)}")

        return concepts_counts


def update_concepts_on_existing_vespa_concepts(
    passage: VespaPassage,
    concepts: list[VespaConcept],
) -> list[dict[str, Any]]:
    """
    Update a passage's concepts with the new concepts.

    During the update we remove all the old concepts related to a model. This is as it
    was decided that holding out dated concepts/spans on the passage in Vespa for a
    model is not useful.

    It is also, not possible to duplicate a Concept object in the concepts array as we
    are removing all instances where the model is the same.
    """
    if not passage.concepts:
        return [concept.model_dump(mode="json") for concept in concepts]

    new_concept_models = {concept.model for concept in concepts}

    existing_concepts_to_keep = [
        concept
        for concept in passage.concepts
        if concept.model not in new_concept_models
    ]

    updated_concepts = existing_concepts_to_keep + concepts

    return [concept_.model_dump(mode="json") for concept_ in updated_concepts]


# De-index ----------------------------------------------------------------------


@flow
async def run_partial_updates_of_concepts_for_document_passages__remove(
    document_importer: DocumentImporter,
    cache_bucket: str,
    concepts_counts_prefix: str,
    vespa_search_adapter: VespaSearchAdapter | None = None,
) -> None:
    """
    Run partial update for VespaConcepts on text blocks for a document.

    This is done in the document_passage index.

    Assumptions:

    - The ID field of the VespaConcept object holds the
    context of the text block that it relates to. E.g. the concept ID
    1.10 would relate to the text block ID 10.
    """
    logger = get_run_logger()

    cm, vespa_search_adapter = get_vespa_search_adapter(vespa_search_adapter)

    logger.info("loading S3 labelled passages")
    document_labelled_passages = load_labelled_passages_by_uri(document_importer[1])

    logger.info("converting labelled passages to Vespa concepts")
    grouped_concepts: dict[TextBlockId, list[VespaConcept]] = {
        labelled_passage.id: convert_labelled_passage_to_concepts(labelled_passage)
        for labelled_passage in document_labelled_passages
    }

    logger.info(
        f"starting partial updates for {len(grouped_concepts)} grouped concepts"
    )

    batches = iterate_batch(
        list(grouped_concepts.items()),
        batch_size=DEFAULT_DOCUMENTS_BATCH_SIZE,
    )

    has_failures = False

    with cm:
        for batch_num, batch in enumerate(batches, start=1):
            logger.info(f"processing partial updates batch {batch_num}")

            partial_update_tasks = [
                partial_update_text_block(
                    text_block_id=text_block_id,
                    document_import_id=remove_translated_suffix(document_importer[0]),
                    concepts=concepts,
                    vespa_search_adapter=vespa_search_adapter,
                    update_function=remove_concepts_from_existing_vespa_concepts,
                )
                for text_block_id, concepts in batch
            ]

            logger.info(f"gathering partial updates tasks for batch {batch_num}")
            results = await asyncio.gather(
                *partial_update_tasks, return_exceptions=True
            )
            logger.info(
                f"gathered partial {len(results)} updates tasks for batch {batch_num}"
            )

            failures = list(
                filter(
                    lambda result: isinstance(result, Exception),
                    results,
                )
            )

            # It seems odd to not worry about the failures. When we
            # calculate the concepts' counts though, we account for
            # failures explicitly.
            #
            # That accounting is carried over implicitly into updating
            # S3 with the final concepts' counts.

            concepts_counts = calculate_concepts_counts_from_results(results, batch)

            await update_s3_with_latest_concepts_counts(
                document_importer=document_importer,
                concepts_counts=concepts_counts,
                cache_bucket=cache_bucket,
                concepts_counts_prefix=concepts_counts_prefix,
                document_labelled_passages=document_labelled_passages,
            )

            # Now, we finally do a little bit of worrying about
            # failures, so they aren't invisible.

            if failures:
                has_failures = True

    if has_failures:
        raise ValueError("there was at least 1 failure")

    return None


def remove_concepts_from_existing_vespa_concepts(
    passage: VespaPassage,
    concepts_to_remove: list[VespaConcept],
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
    concepts_to_remove__models = [concept.model for concept in concepts_to_remove]

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


def calculate_concepts_counts_from_results(
    results: list[BaseException | None],
    batch: list[tuple[TextBlockId, list[VespaConcept]]],
) -> Counter[ConceptModel]:
    logger = get_run_logger()

    # This can handle multiple concepts, but, in practice at the
    # moment, this function is operating on a DocumentImporter,
    # which represents a labelled passages object, which is per
    # concept.
    concepts_counts: Counter[ConceptModel] = Counter()

    for i, result in enumerate(results):
        _text_block_id, concepts = batch[i]

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

        if isinstance(result, Exception):
            # Since we failed to remove them from the spans, make sure
            # they're accounted for as remaining.
            logger.info(f"partial update failed: {str(result)}")
            concepts_counts.update(concepts_models)

    return concepts_counts


async def update_s3_with_latest_concepts_counts(
    document_importer: DocumentImporter,
    concepts_counts: Counter[ConceptModel],
    cache_bucket: str,
    concepts_counts_prefix: str,
    document_labelled_passages: list[LabelledPassage],
) -> None:
    logger = get_run_logger()

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
        logger.info("successfully updated all concepts")
        update_s3_with_all_successes(
            document_object_uri=document_importer[1],
            cache_bucket=cache_bucket,
            concepts_counts_prefix=concepts_counts_prefix,
        )
    # We didn't succeed with all, so write the concepts counts still
    else:
        logger.info("only updated some concepts")
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
    logger = get_run_logger()

    logger.info("updating S3 with all successes")

    s3 = boto3.client("s3")

    s3_uri = Path(document_object_uri)

    # First, delete the concepts counts object
    # Get all parts after the prefix (e.g. "CCLW.executive.1813.2418.json")
    key_parts = "/".join(s3_uri.parts[3:])  # Skip s3://bucket/labelled_passages/

    concepts_counts_key = f"{concepts_counts_prefix}/{key_parts}"

    logger.info(
        f"deleting concepts counts from bucket `{cache_bucket}`, key: `{concepts_counts_key}`"
    )
    if not s3_file_exists(bucket_name=cache_bucket, file_key=concepts_counts_key):
        logger.warning(
            "planned on deleting concepts counts from bucket: "
            f"`{cache_bucket}`, key: `{concepts_counts_key}`, "
            "but the object doesn't exist"
        )
    else:
        s3.delete_object(Bucket=cache_bucket, Key=concepts_counts_key)

    logger.info("updated S3 with deleted concepts counts")

    # Second, delete the labelled passages
    # Get all parts except for the bucket (e.g. "labelled_passages/Q787/v4/CCLW.executive.1813.2418.json")
    labelled_passages_key = "/".join(s3_uri.parts[2:])  # Skip s3://bucket/

    logger.info(
        f"deleting labelled passages from bucket `{cache_bucket}`, key: `{labelled_passages_key}`"
    )
    if not s3_file_exists(bucket_name=cache_bucket, file_key=labelled_passages_key):
        logger.warning(
            "planned on deleting labelled passages from bucket: "
            f"`{cache_bucket}`, key: `{labelled_passages_key}`, "
            "but the object doesn't exist"
        )
    else:
        s3.delete_object(Bucket=cache_bucket, Key=labelled_passages_key)

    logger.info("updated S3 with deleted labelled passages")

    logger.info("updated S3 with all successes")

    return None


def update_s3_with_some_successes(
    document_object_uri: DocumentObjectUri,
    concepts_counts_filtered: Counter[ConceptModel],
    document_labelled_passages: list[LabelledPassage],
    cache_bucket: str,
    concepts_counts_prefix: str,
) -> None:
    logger = get_run_logger()

    logger.info("updating S3 with partial successes")

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

    logger.info("updated S3 with updated concepts counts")

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

    logger.info("updated S3 with updated labelled passages")

    logger.info("updated S3 with partial successes")

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
