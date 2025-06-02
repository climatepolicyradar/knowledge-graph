import json
from typing import Any

import boto3
from cpr_sdk.models.search import Passage as VespaPassage
from prefect import flow
from prefect.logging import get_run_logger
from vespa.application import VespaAsync
from vespa.io import VespaResponse

from flows.aggregate_inference_results import DocumentImportId, S3Uri
from flows.boundary import (
    TextBlockId,
    VespaDataId,
    VespaHitId,
    get_data_id_from_vespa_hit_id,
    get_document_passages_from_vespa__generator,
)


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
    s3_uri: S3Uri,
    document_import_id: DocumentImportId,
    vespa_connection_pool: VespaAsync,
) -> None:
    """Index aggregated inference results from S3 into Vespa for a document."""

    logger = get_run_logger()

    logger.info(f"Loading aggregated inference results from S3: {s3_uri}")
    aggregated_inference_results = load_json_data_from_s3(
        bucket=s3_uri.bucket, key=s3_uri.key
    )

    passages_generator = get_document_passages_from_vespa__generator(
        document_import_id=document_import_id,
        vespa_connection_pool=vespa_connection_pool,
    )

    passages_in_vespa: dict[TextBlockId, tuple[VespaHitId, VespaPassage]] = {}
    async for passage_batch in passages_generator:
        passages_in_vespa.update(passage_batch)

    logger.info(
        f"Updating concepts for document import ID: {document_import_id}, found {len(passages_in_vespa)} passages in Vespa",
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
        f"Successfully indexed all aggregated inference results for document import ID: {document_import_id} into Vespa. Total passages updated: {len(aggregated_inference_results)}"
    )
