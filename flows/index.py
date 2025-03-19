import asyncio
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import vespa.querybuilder as qb
from cpr_sdk.models.search import Concept as VespaConcept
from cpr_sdk.models.search import Passage as VespaPassage
from cpr_sdk.search_adaptors import VespaSearchAdapter
from prefect import flow
from prefect.deployments.deployments import run_deployment
from prefect.logging import get_logger, get_run_logger
from vespa.io import VespaQueryResponse, VespaResponse

from flows.boundary import (
    HTTP_OK,
    ConceptModel,
    DocumentImporter,
    DocumentImportId,
    DocumentStem,
    TextBlockId,
    VespaHitId,
    convert_labelled_passage_to_concepts,
    get_data_id_from_vespa_hit_id,
    get_vespa_search_adapter,
    load_labelled_passages_by_uri,
    s3_obj_generator,
    s3_object_write_text,
    s3_paths_or_s3_prefixes,
)
from flows.inference import DOCUMENT_TARGET_PREFIX_DEFAULT
from flows.utils import (
    SlackNotify,
    iterate_batch,
    remove_translated_suffix,
)
from scripts.cloud import (
    AwsEnv,
    ClassifierSpec,
    function_to_flow_name,
    generate_deployment_name,
    get_prefect_job_variable,
)
from scripts.update_classifier_spec import parse_spec_file
from src.exceptions import PartialUpdateError, QueryError
from src.identifiers import WikibaseID

DEFAULT_DOCUMENTS_BATCH_SIZE = 500
DEFAULT_INDEXING_TASK_BATCH_SIZE = 20
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


async def partial_update_text_block(
    text_block_id: TextBlockId,
    concepts: list[VespaConcept],  # A possibly empty list
    document_import_id: DocumentImportId,
    vespa_search_adapter: VespaSearchAdapter,
):
    """Partial update a singular text block and its concepts."""
    document_passage_id, document_passage = get_document_passage_from_vespa(
        text_block_id, document_import_id, vespa_search_adapter
    )

    data_id = get_data_id_from_vespa_hit_id(document_passage_id)

    serialised_concepts = update_concepts_on_existing_vespa_concepts(
        document_passage,
        concepts,
    )

    response: VespaResponse = vespa_search_adapter.client.update_data(  # pyright: ignore[reportOptionalMemberAccess]
        schema="document_passage",
        namespace="doc_search",
        data_id=data_id,
        fields={"concepts": serialised_concepts},
    )

    if (status_code := response.get_status_code()) != HTTP_OK:
        raise PartialUpdateError(data_id, status_code)


# FIXME: From what I can tell this function should be identical to the related function
# deindex.py. In deindex.py we need to remove_translated_suffix from the
# document_import_id and in this function I'm assuming we should add the updates to how
# we calculate concepts counts from results.
@flow
async def run_partial_updates_of_concepts_for_document_passages(
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

            # We query vespa for document passages that contain a matching import id.
            # The document imported contains the file stem which could contain a
            # translated suffix. We remove this suffix to get the document import id.
            # E.g. CCLW.executive.1.1_translated_en -> CCLW.executive.1.1
            document_import_id = remove_translated_suffix(document_importer[0])

            partial_update_tasks = [
                partial_update_text_block(
                    text_block_id=text_block_id,
                    concepts=concepts,
                    document_import_id=document_import_id,
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
            _ = s3_object_write_text(
                s3_uri=concepts_counts_uri,
                text=serialised_concepts_counts,
            )
        except Exception as e:
            logger.error(f"Failed to write concepts counts to S3: {str(e)}")

        return concepts_counts


# FIXME: We can just pass in the callable and deduplicate against deindex.py
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
            document_stem: DocumentStem = documents_batch[i][0]
            logger.error(
                f"failed to process document `{document_stem}`: {e.__str__()}",
            )
            continue


# FIXME: Identical function to that in index.py
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


# FIXME: Identical to deindex_by_s3
@flow
async def index_by_s3(
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
async def index_labelled_passages_from_s3_to_vespa(
    classifier_specs: list[ClassifierSpec] | None = None,
    document_ids: list[str] | None = None,
    config: Config | None = None,
    batch_size: int = DEFAULT_DOCUMENTS_BATCH_SIZE,
    indexing_task_batch_size: int = DEFAULT_INDEXING_TASK_BATCH_SIZE,
) -> None:
    """
    Asynchronously index concepts from S3 into Vespa.

    This function retrieves concept documents from files stored in an
    S3 path and indexes them in a Vespa instance. The name of each
    file in the specified S3 path is expected to represent the
    document's import ID.
    """
    logger = get_run_logger()

    if not config:
        logger.info("no config provided, creating one")

        config = await Config.create()
    else:
        logger.info("config provided")
    assert config.cache_bucket

    logger.info(f"running with config: {config}")

    if classifier_specs is None:
        logger.info("no classifier specs. passed in, loading from file")
        classifier_specs = parse_spec_file(config.aws_env)

    logger.info(f"running with classifier specs: {classifier_specs}")

    s3_accessor = s3_paths_or_s3_prefixes(
        classifier_specs,
        document_ids,
        config.cache_bucket,
        config.document_source_prefix,
    )

    logger.info(f"s3_prefixes: {s3_accessor.prefixes}, s3_paths: {s3_accessor.paths}")

    await index_by_s3(
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
