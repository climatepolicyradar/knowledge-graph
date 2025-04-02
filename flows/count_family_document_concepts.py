import asyncio
import json
import os
from collections import Counter, defaultdict
from collections.abc import Awaitable
from dataclasses import dataclass

from cpr_sdk.s3 import _s3_object_read_text
from cpr_sdk.search_adaptors import VespaSearchAdapter
from prefect import flow, get_run_logger
from prefect.client.schemas.objects import FlowRun, StateType
from prefect.deployments.deployments import run_deployment
from vespa.io import VespaResponse

from flows.boundary import (
    CONCEPT_COUNT_SEPARATOR,
    CONCEPTS_COUNTS_PREFIX_DEFAULT,
    DocumentImportId,
    DocumentObjectUri,
    S3Accessor,
    get_vespa_search_adapter,
    s3_obj_generator,
    s3_paths_or_s3_prefixes,
)
from flows.utils import SlackNotify, iterate_batch
from scripts.cloud import (
    AwsEnv,
    ClassifierSpec,
    function_to_flow_name,
    generate_deployment_name,
    get_prefect_job_variable,
)
from scripts.update_classifier_spec import parse_spec_file
from src.concept import Concept
from src.exceptions import PartialUpdateError
from src.identifiers import WikibaseID

DEFAULT_BATCH_SIZE = 50


@dataclass()
class Config:
    """Configuration used across flow runs."""

    cache_bucket: str | None = None
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


async def partial_update_family_document_concepts_counts(
    document_import_id: DocumentImportId,
    concepts_counts_with_names: dict[str, int],
    vespa_search_adapter: VespaSearchAdapter,
) -> None:
    """
    Update document concept counts in Vespa via partial updates.

    Similar to index.update_concepts_on_existing_vespa_concepts, during the update
    we remove all the old concepts related to a model. This is as it
    was decided that holding out dated concepts counts on the document
    in Vespa for a model is not useful.
    """

    response: VespaResponse = vespa_search_adapter.client.update_data(
        schema="family_document",
        namespace="doc_search",
        data_id=document_import_id,
        fields={
            "concept_counts": concepts_counts_with_names
        },  # Note the schema is misnamed in Vespa
    )

    if not response.is_successful():
        raise PartialUpdateError(document_import_id, response.get_status_code())

    return None


async def load_parse_concepts_counts(
    document_object_uri: DocumentObjectUri,
) -> Counter[Concept]:
    """
    Load and parse concept counts from a JSON file into a Counter of Concepts.

    At the moment, these are only expected to have 1 concept-count
    pair in them, from the indexing pipeline.
    """
    # Load object from S3
    object_json: dict[str, int] = json.loads(
        _s3_object_read_text(s3_path=document_object_uri)
    )

    # Parse the count(s)
    counter: Counter[Concept] = Counter()

    for concept_key, concept_count in object_json.items():
        wikibase_id, preferred_label = concept_key.split(
            CONCEPT_COUNT_SEPARATOR,
            maxsplit=1,
        )

        concept = Concept(
            preferred_label=preferred_label,
            wikibase_id=WikibaseID(wikibase_id),
        )

        # Add to counter
        counter[concept] = concept_count

    return counter


@flow(
    on_failure=[SlackNotify.message],
    on_crashed=[SlackNotify.message],
)
async def load_update_document_concepts_counts(
    document_import_id: DocumentImportId,
    document_object_uris: list[DocumentObjectUri],
    batch_size: int,
    vespa_search_adapter: VespaSearchAdapter | None = None,
) -> dict[str, int]:
    """Load and aggregate concept counts from document URIs."""
    logger = get_run_logger()

    concepts_counts: Counter[Concept] = Counter()

    document_object_uris_batches = iterate_batch(
        document_object_uris, batch_size=batch_size
    )

    has_failures = False

    for (
        batch_num,
        batch,
    ) in enumerate(document_object_uris_batches, start=1):
        logger.info(f"processing batch document object URIs #{batch_num}")

        tasks = [
            load_parse_concepts_counts(document_object_uri)
            for document_object_uri in batch
        ]

        logger.info(
            f"gathering {batch_size} load and parse concepts counts tasks for batch {batch_num}"
        )
        results = await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )
        logger.info(
            f"gathered {batch_size} load and parse concepts counts tasks for batch {batch_num}"
        )

        for i, result in enumerate(results):
            current_document_import_id: DocumentImportId = batch[i]

            if isinstance(result, Exception):
                logger.error(
                    f"failed to process document `{current_document_import_id}`: {str(result)}",
                )
                has_failures = True
                continue

            if isinstance(result, Counter):
                concepts_counts = concepts_counts + result

            logger.info(f"processed batch document object URIs #{batch_num}")

    cm, vespa_search_adapter = get_vespa_search_adapter(vespa_search_adapter)

    # Continue on, even if there were failures, as that's accounted for when
    # calculating the concepts' counts.

    # Serialise them
    concepts_counts_with_names = {
        f"{concept.wikibase_id}{CONCEPT_COUNT_SEPARATOR}{concept.preferred_label}": count
        for concept, count in concepts_counts.items()
    }

    with cm:
        await partial_update_family_document_concepts_counts(
            document_import_id,
            concepts_counts_with_names,
            vespa_search_adapter,
        )

    # Now, we finally do a little bit of worrying about
    # failures, so they aren't invisible.

    if has_failures:
        raise ValueError("there was at least 1 failure")

    return concepts_counts_with_names


def load_update_document_concepts_counts_as(
    document_import_id: DocumentImportId,
    document_object_uris: list[DocumentObjectUri],
    batch_size: int,
    vespa_search_adapter: VespaSearchAdapter | None,
    aws_env: AwsEnv,
    as_deployment: bool,
) -> Awaitable[dict[str, int]]:
    """Run load document concepts either as a subflow or directly."""
    if as_deployment:
        flow_name = function_to_flow_name(load_update_document_concepts_counts)
        deployment_name = generate_deployment_name(flow_name=flow_name, aws_env=aws_env)

        return run_deployment(
            name=f"{flow_name}/{deployment_name}",
            parameters={
                "document_import_id": document_import_id,
                "document_object_uris": document_object_uris,
                "batch_size": batch_size,
            },
            timeout=1200,
        )
    else:
        return load_update_document_concepts_counts(
            document_import_id,
            document_object_uris,
            batch_size,
            vespa_search_adapter,
        )


def group_documents_uris(
    s3_accessor: S3Accessor,
) -> dict[DocumentImportId, list[DocumentObjectUri]]:
    """
    Group document URIs by their document import ID.

    We do this so we can do a document's concepts all "together", and then move onto other documents.
    As opposed to going through an un-grouped list, where possibly a document will have concepts at the
    "start" and "end" of the S3 prefix.

    URIs that don't end with the expected suffix, will have it appended.
    """
    documents_generator = s3_obj_generator(
        s3_prefixes=s3_accessor.prefixes,
        s3_paths=s3_accessor.paths,
    )

    documents_by_id: dict[DocumentImportId, list[DocumentObjectUri]] = defaultdict(list)

    # Group documents by their ID
    for document_import_id, document_object_uri in documents_generator:
        # Check if URI ends with {document_import_id}.json
        expected_suffix = f"{document_import_id}.json"

        if not document_object_uri.endswith(expected_suffix):
            # If not, append it
            document_object_uri = (
                document_object_uri.rstrip("/") + "/" + expected_suffix
            )

        documents_by_id[document_import_id].append(document_object_uri)

    return dict(documents_by_id)  # Convert defaultdict to regular dict


@flow(
    on_failure=[SlackNotify.message],
    on_crashed=[SlackNotify.message],
)
async def count_family_document_concepts(
    classifier_specs: list[ClassifierSpec] | None = None,
    document_ids: list[str] | None = None,
    config: Config | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> None:
    """Process document concept counts from S3 and update Vespa."""
    logger = get_run_logger()

    if not config:
        logger.info("no config provided, creating one")

        config = await Config.create()
    else:
        logger.info("config provided")

    logger.info(f"running with config: {config}")

    if classifier_specs is None:
        logger.info("no classifier specs. passed in, loading from file")
        classifier_specs = parse_spec_file(config.aws_env)

    logger.info(f"running with classifier specs.: {classifier_specs}")

    s3_accessor = s3_paths_or_s3_prefixes(
        classifier_specs,
        document_ids,
        config.cache_bucket,  # pyright: ignore[reportArgumentType]
        config.concepts_counts_prefix,
    )

    logger.info(f"s3_paths: {s3_accessor.paths}, s3_prefixes: {s3_accessor.prefixes}")

    documents_by_id = group_documents_uris(s3_accessor)

    logger.info(
        f"grouped {sum(len(docs) for docs in documents_by_id.values())} documents into {len(documents_by_id)} unique IDs"
    )

    # Convert dictionary items to list before batching
    documents_items = list(documents_by_id.items())
    documents_batches = iterate_batch(documents_items, batch_size=batch_size)

    has_failures = False

    for (
        documents_batch_num,
        documents_batch,
    ) in enumerate(documents_batches, start=1):
        logger.info(f"processing batch documents #{documents_batch_num}")

        load_update_document_groups_tasks = [
            load_update_document_concepts_counts_as(
                document_import_id,
                document_object_uris,
                batch_size,
                config.vespa_search_adapter,
                config.aws_env,
                config.as_deployment,
            )
            for document_import_id, document_object_uris in documents_batch
        ]

        logger.info(
            f"gathering {batch_size} load and update document groups tasks for batch {documents_batch_num}"
        )
        results = await asyncio.gather(
            *load_update_document_groups_tasks, return_exceptions=True
        )
        logger.info(
            f"gathered {batch_size} load and update document groups tasks for batch {documents_batch_num}"
        )

        for i, result in enumerate(results):
            document_import_id: DocumentImportId = documents_batch[i][0]

            if isinstance(result, Exception):
                logger.error(
                    f"failed to process group for document `{document_import_id}`: {str(result)}",
                )
                has_failures = True
                continue

            if isinstance(result, FlowRun):
                flow_run: FlowRun = result
                if flow_run.state.type != StateType.COMPLETED:
                    has_failures = True

            logger.info(f"processed batch documents #{documents_batch_num}")

    if has_failures:
        raise ValueError("there was at least 1 failure")

    return None
