import asyncio
import base64
import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
from cpr_sdk.models.search import Concept as VespaConcept
from cpr_sdk.models.search import Passage as VespaPassage
from cpr_sdk.search_adaptors import VespaSearchAdapter
from cpr_sdk.ssm import get_aws_ssm_param
from prefect import flow
from prefect.deployments.deployments import run_deployment
from prefect.logging import get_run_logger
from pydantic import BaseModel
from vespa.io import VespaResponse

from flows.boundary import (
    DocumentImporter,
    DocumentImportId,
    DocumentObjectUri,
    TextBlockId,
    convert_labelled_passage_to_concepts,
    get_data_id_from_vespa_hit_id,
    get_document_passage_from_vespa,
    get_vespa_search_adapter,
    load_labelled_passages_by_uri,
    s3_obj_generator,
    s3_object_write_text,
    s3_paths_or_s3_prefixes,
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
from src.concept import Concept
from src.exceptions import PartialUpdateError
from src.identifiers import WikibaseID
from src.labelled_passage import LabelledPassage
from src.span import Span

DEFAULT_DOCUMENTS_BATCH_SIZE = 500
DEFAULT_INDEXING_TASK_BATCH_SIZE = 20
HTTP_OK = 200
CONCEPTS_COUNTS_PREFIX_DEFAULT: str = "concepts_counts"
CONCEPT_COUNT_SEPARATOR: str = ":"


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


def save_labelled_passages_by_uri(
    document_object_uri: DocumentObjectUri,
    labelled_passages: list[LabelledPassage],
) -> None:
    """Save LabelledPassages objects to S3."""
    object_json = json.dumps(
        [labelled_passage.model_dump_json() for labelled_passage in labelled_passages]
    )

    _ = s3_object_write_text(
        s3_uri=document_object_uri,
        text=object_json,
    )


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


def get_updated_passage_concepts(
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


@flow
async def run_partial_updates_of_concepts_for_document_passages(
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

    logger.info("getting S3 labelled passages generator")
    document_labelled_passages = load_labelled_passages_by_uri(document_importer[1])

    with cm:
        logger.info(
            (
                "getting document passages from Vespa for document "
                f"import ID {document_importer[0]}"
            )
        )

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

        for batch_num, batch in enumerate(batches, start=1):
            logger.info(f"processing partial updates batch {batch_num}")

            partial_update_tasks = [
                partial_update_text_block(
                    text_block_id=text_block_id,
                    document_import_id=document_importer[0],
                    concepts=concepts,
                    vespa_search_adapter=vespa_search_adapter,
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

            concepts_counts = calculate_concepts_counts_from_results(results, batch)

            await update_s3_with_latest_concepts_counts(
                document_importer=document_importer,
                concepts_counts=concepts_counts,
                cache_bucket=cache_bucket,
                concepts_counts_prefix=concepts_counts_prefix,
                document_labelled_passages=document_labelled_passages,
            )


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
    logger = get_run_logger()

    logger.info("updating S3 with all successes")

    s3 = boto3.client("s3")

    s3_uri = Path(document_object_uri)

    # First, delete the concepts counts object
    # Get all parts after the prefix (e.g. "Q787/v4/CCLW.executive.1813.2418.json")
    key_parts = "/".join(s3_uri.parts[3:])  # Skip s3://bucket/labelled_passages/

    concepts_counts_key = f"{concepts_counts_prefix}/{key_parts}"

    _ = s3.delete_object(Bucket=cache_bucket, Key=concepts_counts_key)

    logger.info("updated S3 with deleted concepts counts")

    # Second, delete the labelled passages
    # Get all parts except for the bucket (e.g. "labelled_passages/Q787/v4/CCLW.executive.1813.2418.json")
    labelled_passages_key = "/".join(s3_uri.parts[2:])  # Skip s3://bucket/

    _ = s3.delete_object(Bucket=cache_bucket, Key=labelled_passages_key)

    logger.info("updated S3 with deleted labelled passages")

    logger.info("updated S3 with all successes")

    return None


def serialise_concepts_counts(concepts_counts: Counter[ConceptModel]) -> str:
    return json.dumps({str(k): v for k, v in concepts_counts.items()})


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

    _ = s3_object_write_text(
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

    _ = save_labelled_passages_by_uri(
        document_object_uri=document_object_uri,
        labelled_passages=filtered_labelled_passages,
    )

    logger.info("updated S3 with updated labelled passages")

    logger.info("updated S3 with partial successes")

    return None


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


async def partial_update_text_block(
    text_block_id: TextBlockId,
    document_import_id: DocumentImportId,
    concepts: list[VespaConcept],  # A possibly empty list
    vespa_search_adapter: VespaSearchAdapter,
) -> None:
    """Partial update a singular text block and its concepts, if any."""
    document_passage_id, document_passage = get_document_passage_from_vespa(
        text_block_id, document_import_id, vespa_search_adapter
    )

    data_id = get_data_id_from_vespa_hit_id(document_passage_id)

    serialised_concepts = get_updated_passage_concepts(
        passage=document_passage,
        concepts_to_remove=concepts,
    )

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
) -> None:
    """Run partial updates for concepts in a batch of documents."""

    logger = get_run_logger()
    logger.info(
        f"Updating concepts for batch of documents, documents in batch: {len(documents_batch)}."
    )
    for i, document_importer in enumerate(documents_batch):
        try:
            _ = await run_partial_updates_of_concepts_for_document_passages(
                document_importer=document_importer,
                cache_bucket=cache_bucket,
                concepts_counts_prefix=concepts_counts_prefix,
            )

            logger.info(f"processed batch documents #{documents_batch_num}")

        except Exception as e:
            document_import_id: DocumentImportId = documents_batch[i][0]
            logger.error(
                f"failed to process document `{document_import_id}`: {e.__str__()}",
            )
            continue


async def run_partial_updates_of_concepts_for_batch_flow_or_deployment(
    documents_batch: list[DocumentImporter],
    documents_batch_num: int,
    cache_bucket: str,
    concepts_counts_prefix: str,
    aws_env: AwsEnv,
    as_deployment: bool,
) -> None:
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
            },
            timeout=3600,
        )

    return await run_partial_updates_of_concepts_for_batch(
        documents_batch=documents_batch,
        documents_batch_num=documents_batch_num,
        cache_bucket=cache_bucket,
        concepts_counts_prefix=concepts_counts_prefix,
    )


@flow
async def deindex_by_s3(
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


@flow(
    on_failure=[SlackNotify.message],
    on_crashed=[SlackNotify.message],
)
async def deindex_labelled_passages_from_s3_to_vespa(
    classifier_specs: list[ClassifierSpec],
    document_ids: list[str] | None = None,
    config: Config | None = None,
    batch_size: int = DEFAULT_DOCUMENTS_BATCH_SIZE,
    indexing_task_batch_size: int = DEFAULT_INDEXING_TASK_BATCH_SIZE,
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

    await deindex_by_s3(
        aws_env=config.aws_env,
        vespa_search_adapter=config.vespa_search_adapter,
        s3_prefixes=s3_accessor.prefixes,
        s3_paths=s3_accessor.paths,
        batch_size=batch_size,
        indexing_task_batch_size=indexing_task_batch_size,
        as_deployment=config.as_deployment,
        cache_bucket=config.cache_bucket,
        concepts_counts_prefix=config.concepts_counts_prefix,
    )
