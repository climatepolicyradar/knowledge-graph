"""Boundary between Prefect, Vespa, and AWS."""

import asyncio
import base64
import contextlib
import json
import os
import re
import tempfile
from collections.abc import Generator
from io import BytesIO
from pathlib import Path
from typing import Callable, ContextManager, TypeAlias

import boto3
import vespa.querybuilder as qb
from cpr_sdk.models.search import Concept as VespaConcept
from cpr_sdk.models.search import Document as VespaDocument
from cpr_sdk.models.search import Passage as VespaPassage
from cpr_sdk.s3 import _get_s3_keys_with_prefix, _s3_object_read_text
from cpr_sdk.search_adaptors import VespaSearchAdapter
from cpr_sdk.ssm import get_aws_ssm_param
from prefect import flow, get_run_logger
from prefect.deployments.deployments import run_deployment
from prefect.logging import get_logger
from pydantic import BaseModel
from vespa.io import VespaQueryResponse, VespaResponse

from flows.utils import iterate_batch
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
DEFAULT_DOCUMENTS_BATCH_SIZE = 500
DEFAULT_INDEXING_TASK_BATCH_SIZE = 20

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
        _ = f.write(vespa_public_cert)

    with open(key_path, "w", encoding="utf-8") as f:
        _ = f.write(vespa_private_key)

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
    _ = s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")


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

            # FIXME: Add translated documents here!
            #   This is handled in a later stacked PR.
            document_paths = [
                "s3://"
                + os.path.join(
                    cache_bucket,
                    prefix,
                    classifier_spec.name,
                    classifier_spec.alias,
                    f"{doc_id}.json",
                )
                for classifier_spec in classifier_specs
                for doc_id in document_ids
            ]
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

    # The concept used to label the passage holds some information on the parent
    # concepts and thus this is being used as a temporary solution for providing
    # the relationship between concepts. This has the downside that it ties a
    # labelled passage to a particular concept when in fact the Spans that a
    # labelled passage has can be labelled by multiple concepts.
    concept = Concept.model_validate(labelled_passage.metadata["concept"])
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
    update_function: Callable,
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


@flow
async def run_partial_updates_of_concepts_for_batch(
    documents_batch: list[DocumentImporter],
    documents_batch_num: int,
    cache_bucket: str,
    concepts_counts_prefix: str,
    partial_update_flow: Callable,
) -> None:
    """Run partial updates for concepts in a batch of documents."""

    logger = get_run_logger()
    logger.info(
        f"Updating concepts for batch of documents, documents in batch: {len(documents_batch)}."
    )
    for i, document_importer in enumerate(documents_batch):
        try:
            _ = await partial_update_flow(
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
            continue


async def run_partial_updates_of_concepts_for_batch_flow_or_deployment(
    documents_batch: list[DocumentImporter],
    documents_batch_num: int,
    cache_bucket: str,
    concepts_counts_prefix: str,
    aws_env: AwsEnv,
    as_deployment: bool,
    partial_update_flow: Callable,
) -> None:
    """Run partial updates for a batch of documents as a sub-flow or deployment."""
    logger = get_run_logger()
    logger.info(
        "Running partial updates of concepts for batch as sub-flow or deployment: "
        f"batch length {len(documents_batch)}, as_deployment: {as_deployment}"
    )

    if as_deployment:
        flow_name = function_to_flow_name(partial_update_flow)
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
async def index_by_s3(
    partial_update_flow: Callable,
    aws_env: AwsEnv,
    cache_bucket: str,
    concepts_counts_prefix: str,
    vespa_search_adapter: VespaSearchAdapter | None,
    s3_prefixes: list[str] | None = None,
    s3_paths: list[str] | None = None,
    batch_size: int = DEFAULT_DOCUMENTS_BATCH_SIZE,
    indexing_task_batch_size: int = DEFAULT_INDEXING_TASK_BATCH_SIZE,
    as_deployment: bool = True,
) -> None:
    """
    Asynchronously index concepts from S3 files into Vespa.

    This function retrieves concept documents from files stored in an S3 path and
    indexes them in a Vespa instance. The name of each file in the specified S3 path is
    expected to represent the document's import ID.

    When `s3_prefix` is provided, the function will index all files within that S3
    prefix (directory). When `s3_paths` is provided, the function will index only the
    files specified in the list of S3 object keys. If both are provided `s3_paths` will
    be used.

    Assumptions:
    - The S3 file names represent document import IDs.

    params:
    - s3_prefix: The S3 prefix (directory) to yield objects from.
        E.g. "s3://bucket/prefix/"
    - s3_paths: A list of S3 object keys to yield objects from.
        E.g. {"s3://bucket/prefix/file1.json", "s3://bucket/prefix/file2.json"}
    - vespa_search_adapter: An instance of VespaSearchAdapter.
        E.g. VespaSearchAdapter(
            instance_url="https://vespa-instance-url.com",
            cert_directory="certs/"
        )
    """
    logger = get_run_logger()

    cm, vespa_search_adapter = get_vespa_search_adapter(vespa_search_adapter)

    with cm:
        logger.info("Getting S3 object generator")
        documents_generator = s3_obj_generator(s3_prefixes, s3_paths)
        documents_batches = iterate_batch(documents_generator, batch_size=batch_size)
        indexing_task_batches = iterate_batch(
            data=documents_batches, batch_size=indexing_task_batch_size
        )

        for i, indexing_task_batch in enumerate(indexing_task_batches, start=1):
            logger.info(f"Processing indexing task batch #{i}")

            indexing_tasks = [
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
                    indexing_task_batch, start=1
                )
            ]

            logger.info(f"Gathering indexing tasks for batch #{i}")
            results = await asyncio.gather(*indexing_tasks, return_exceptions=True)
            logger.info(f"Gathered indexing tasks for batch #{i}")

            for result in results:
                if isinstance(result, Exception):
                    logger.error(
                        f"failed to process document batch in indexing task batch #{i}: {str(result)}",
                    )
                    continue
