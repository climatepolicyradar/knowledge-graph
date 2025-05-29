import json
from typing import Any

import boto3
from prefect import flow
from vespa.application import VespaAsync
from vespa.io import VespaResponse

from flows.aggregate_inference_results import DocumentImportId, S3Uri
from flows.boundary import TextBlockId, get_data_id_for_passage_from_vespa


async def _update_vespa_passage_concepts(
    text_block_id: TextBlockId,
    document_import_id: DocumentImportId,
    serialised_concepts: list[dict[str, Any]],
    vespa_connection_pool: VespaAsync,
) -> VespaResponse:
    """Update a passage in Vespa with the given concepts."""

    # FIXME: If the data id doesn't exist here this silently fails.
    vespa_hit_id = await get_data_id_for_passage_from_vespa(
        document_import_id=document_import_id,
        text_block_id=text_block_id,
        vespa_connection_pool=vespa_connection_pool,
    )

    response: VespaResponse = await vespa_connection_pool.update_data(
        schema="document_passage",
        namespace="doc_search",
        data_id=vespa_hit_id,
        fields={"concepts": serialised_concepts},
    )

    return response


# TODO: Add logging and retries, time outs?
@flow
async def index_aggregate_results_from_s3_to_vespa(
    s3_uri: S3Uri,
    vespa_connection_pool: VespaAsync,
) -> None:
    """Index aggregated inference results from S3 into Vespa for a document."""
    # TOOD: Move to a function
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=s3_uri.bucket, Key=s3_uri.key)
    body = response["Body"].read().decode("utf-8")
    aggregated_inference_results = json.loads(body)

    document_import_id = DocumentImportId(s3_uri.stem)

    for text_block_id, concepts in aggregated_inference_results.items():
        response = await _update_vespa_passage_concepts(
            text_block_id=text_block_id,
            document_import_id=document_import_id,
            serialised_concepts=concepts,
            vespa_connection_pool=vespa_connection_pool,
        )
        if not response.is_successful():
            # TODO: Collect error
            raise ValueError(
                f"Failed to update text block {text_block_id} in Vespa: {response}"
            )
