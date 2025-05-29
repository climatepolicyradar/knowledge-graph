import json
import tempfile
from typing import Any, Final

import boto3
import httpx
from cpr_sdk.models.search import Passage as VespaPassage
from prefect import flow
from prefect.logging import get_run_logger
from pydantic import PositiveInt
from vespa.application import VespaAsync
from vespa.io import VespaResponse

from flows.aggregate_inference_results import DocumentImportId, S3Uri
from flows.boundary import (
    VESPA_MAX_TIMEOUT_MS,
    TextBlockId,
    VespaDataId,
    VespaHitId,
    get_document_passages_from_vespa__generator,
    get_vespa_search_adapter_from_aws_secrets,
)

DEFAULT_INDEXER_MAX_CONNECTIONS: Final[PositiveInt] = 50


def load_json_data_from_s3(s3_uri: S3Uri) -> dict[str, Any]:
    """Load JSON data from an S3 URI."""

    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=s3_uri.bucket, Key=s3_uri.key)
    body = response["Body"].read().decode("utf-8")
    return json.loads(body)


async def _update_vespa_passage_concepts(
    vespa_data_id: VespaDataId,
    serialised_concepts: list[dict[str, Any]],
    vespa_connection_pool: VespaAsync,
) -> VespaResponse:
    """Update a passage in Vespa with the given concepts."""

    # FIXME: If the data id doesn't exist here this silently fails.
    response: VespaResponse = await vespa_connection_pool.update_data(
        schema="document_passage",
        namespace="doc_search",
        data_id=vespa_data_id,
        fields={"concepts": serialised_concepts},
    )

    return response


@flow
async def index_aggregate_inference_results_to_vespa_passages(
    s3_uri: S3Uri,
    vespa_connection_pool: VespaAsync,
) -> None:
    """Index aggregated inference results from S3 into Vespa for a document."""

    logger = get_run_logger()

    document_import_id = DocumentImportId(s3_uri.stem)

    logger.info(f"Loading aggregated inference results from S3: {s3_uri}")
    aggregated_inference_results = load_json_data_from_s3(s3_uri=s3_uri)

    logger.info(
        "Loading passages from Vespa for document import ID: %s", document_import_id
    )
    passages_generator = get_document_passages_from_vespa__generator(
        document_import_id=document_import_id,
        vespa_connection_pool=vespa_connection_pool,
    )

    passages_in_vespa: dict[TextBlockId, tuple[VespaHitId, VespaPassage]] = {}
    async for passage_batch in passages_generator:
        passages_in_vespa.update(passage_batch)

    logger.info(
        "Loading aggregated inference results for document import ID: %s",
        document_import_id,
    )
    for text_block_id, concepts in aggregated_inference_results.items():
        vespa_hit_id: VespaHitId = passages_in_vespa[TextBlockId(text_block_id)][0]
        vespa_data_id: VespaDataId = VespaDataId(vespa_hit_id.split("::")[-1])

        response = await _update_vespa_passage_concepts(
            vespa_data_id=vespa_data_id,
            serialised_concepts=concepts,
            vespa_connection_pool=vespa_connection_pool,
        )
        if not response.is_successful():
            # TODO: Collect error
            raise ValueError(
                f"Failed to update text block {text_block_id} in Vespa: {response}"
            )


@flow
async def index_aggregate_inference_results_to_vespa_families(
    s3_uri: S3Uri,
    vespa_connection_pool: VespaAsync,
) -> None:
    """Update concept counts on the family index in vespa from aggregated inference results."""
    pass


@flow
async def index_aggregate_inference_results_to_vespa(
    s3_uri_list: list[S3Uri],
) -> None:
    """Index aggregated inference results from a list of S3 URIs into Vespa."""

    logger = get_run_logger()
    logger.info("Starting indexing from aggregate results...")

    temp_dir = tempfile.TemporaryDirectory()
    vespa_search_adapter = get_vespa_search_adapter_from_aws_secrets(
        cert_dir=temp_dir.name,
        vespa_private_key_param_name="VESPA_PRIVATE_KEY_FULL_ACCESS",
        vespa_public_cert_param_name="VESPA_PUBLIC_CERT_FULL_ACCESS",
    )

    async with (
        vespa_search_adapter.client.asyncio(
            connections=DEFAULT_INDEXER_MAX_CONNECTIONS,  # How many tasks to have running at once
            timeout=httpx.Timeout(VESPA_MAX_TIMEOUT_MS / 1_000),  # Seconds
        ) as vespa_connection_pool
    ):
        for s3_uri in s3_uri_list:
            logger.info(
                f"Indexing aggregated results from S3 URI to vespa passages: {s3_uri}"
            )
            await index_aggregate_inference_results_to_vespa_passages(
                s3_uri=s3_uri,
                vespa_connection_pool=vespa_connection_pool,
            )

            logger.info(
                "Indexing concept counts from aggregated results from S3 URI to vespa families: {s3_uri}"
            )
            await index_aggregate_inference_results_to_vespa_families(
                s3_uri=s3_uri,
                vespa_connection_pool=vespa_connection_pool,
            )

            logger.info(f"Finished indexing aggregated results from S3 URI: {s3_uri}")
