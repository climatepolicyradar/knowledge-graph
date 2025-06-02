import json
import os
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

from flows.aggregate_inference_results import (
    Config,
    DocumentImportId,
    RunOutputIdentifier,
)
from flows.boundary import (
    VESPA_MAX_TIMEOUT_MS,
    TextBlockId,
    VespaDataId,
    VespaHitId,
    get_data_id_from_vespa_hit_id,
    get_document_passages_from_vespa__generator,
    get_vespa_search_adapter_from_aws_secrets,
)
from flows.utils import collect_unique_file_stems_under_prefix

DEFAULT_INDEXER_MAX_CONNECTIONS: Final[PositiveInt] = 50


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

    response: VespaResponse = await vespa_connection_pool.update_data(
        schema="document_passage",
        namespace="doc_search",
        data_id=vespa_data_id,
        fields={"concepts": serialised_concepts},
    )

    return response


@flow
async def index_aggregate_results_from_s3_to_vespa(
    config: Config,
    run_output_identifier: RunOutputIdentifier,
    document_import_id: DocumentImportId,
    vespa_connection_pool: VespaAsync,
) -> None:
    """Index aggregated inference results from S3 into Vespa for a document."""

    logger = get_run_logger()

    aggregated_results_key = os.path.join(
        config.aggregate_inference_results_prefix,
        run_output_identifier,
        f"{document_import_id}.json",
    )
    logger.info(
        f"Loading aggregated inference results from S3: {aggregated_results_key}"
    )
    aggregated_inference_results = load_json_data_from_s3(
        bucket=config.cache_bucket_str, key=aggregated_results_key
    )

    passages_generator = get_document_passages_from_vespa__generator(
        document_import_id=document_import_id,
        vespa_connection_pool=vespa_connection_pool,
    )

    passages_in_vespa: dict[TextBlockId, tuple[VespaHitId, VespaPassage]] = {}
    async for passage_batch in passages_generator:
        passages_in_vespa.update(passage_batch)

    logger.info(
        f"Updating concepts for document import ID: {document_import_id}, "
        f"found {len(passages_in_vespa)} passages in Vespa",
    )
    text_blocks_not_in_vespa = []
    update_errors = []
    for text_block_id, concepts in aggregated_inference_results.items():
        if TextBlockId(text_block_id) not in list(passages_in_vespa.keys()):
            text_blocks_not_in_vespa.append(text_block_id)
            continue

        vespa_hit_id: VespaHitId = passages_in_vespa[TextBlockId(text_block_id)][0]
        vespa_data_id: VespaDataId = get_data_id_from_vespa_hit_id(vespa_hit_id)

        response = await _update_vespa_passage_concepts(
            vespa_data_id=vespa_data_id,
            serialised_concepts=concepts,
            vespa_connection_pool=vespa_connection_pool,
        )

        if not response.is_successful():
            update_errors.append((text_block_id, response.get_json()))

    if text_blocks_not_in_vespa or update_errors:
        raise ValueError(
            f"Error with {document_import_id}: {text_blocks_not_in_vespa=}, {update_errors=}"
        )

    logger.info(
        "Successfully indexed all aggregated inference results for document import ID: "
        f"{document_import_id} into Vespa. "
        f"Total passages updated: {len(aggregated_inference_results)}"
    )


@flow
async def run_indexing_from_aggregate_results(
    run_output_identifier: RunOutputIdentifier,
    document_import_ids: list[DocumentImportId],
    config: Config | None = None,
) -> None:
    """Index aggregated inference results from a list of S3 URIs into Vespa."""

    logger = get_run_logger()

    if config is None:
        config = await Config.create()

    if not document_import_ids:
        logger.info(
            "No document import ids provided. "
            f"Running on all documents under run_output_identifier: {run_output_identifier}"
        )
        document_import_ids = [
            DocumentImportId(i)
            for i in collect_unique_file_stems_under_prefix(
                bucket_name=config.cache_bucket_str,
                prefix=os.path.join(
                    config.aggregate_inference_results_prefix, run_output_identifier
                ),
            )
        ]
        logger.info(f"Found {len(document_import_ids)} document import ids to process.")

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
        for document_import_id in document_import_ids:
            logger.info(
                f"Indexing aggregated results for document: {document_import_id}"
            )
            await index_aggregate_results_from_s3_to_vespa(
                config=config,
                run_output_identifier=run_output_identifier,
                document_import_id=document_import_id,
                vespa_connection_pool=vespa_connection_pool,
            )
