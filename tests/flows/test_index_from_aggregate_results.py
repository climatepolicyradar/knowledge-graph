import json

import pytest
from cpr_sdk.search_adaptors import VespaSearchAdapter

from flows.aggregate_inference_results import S3Uri
from flows.boundary import (
    DocumentImportId,
    get_document_passages_from_vespa__generator,
)
from flows.index_from_aggregate_results import index_aggregate_results_from_s3_to_vespa


@pytest.mark.asyncio
async def test_index_from_aggregated_inference_results(
    vespa_app,
    local_vespa_search_adapter: VespaSearchAdapter,
    mock_s3_client,
    mock_bucket: str,
    mock_bucket_inference_results: list[str],
    s3_prefix_inference_results: str,
) -> None:
    """Test that we loaded the inference results from the mock bucket."""
    async with local_vespa_search_adapter.client.asyncio() as vespa_connection_pool:
        file_keys = mock_bucket_inference_results
        for file_key in file_keys:
            response = mock_s3_client.get_object(Bucket=mock_bucket, Key=file_key)
            body = response["Body"].read().decode("utf-8")
            aggregated_inference_results = json.loads(body)
            document_import_id = DocumentImportId(
                file_key.split("/")[-1].replace(".json", "")
            )

            # Get the original vespa passages
            passages_generator = get_document_passages_from_vespa__generator(
                document_import_id=document_import_id,
                vespa_connection_pool=vespa_connection_pool,
                grouping_max=5_000,
            )

            responses = []
            async for vespa_passages in passages_generator:
                responses.append(vespa_passages)

            # Index the aggregated inference results from S3 to Vespa
            await index_aggregate_results_from_s3_to_vespa(
                s3_uri=S3Uri(bucket=mock_bucket, key=file_key),
                vespa_connection_pool=vespa_connection_pool,
            )

            # Get the final vespa passages
            final_passages_generator = get_document_passages_from_vespa__generator(
                document_import_id=document_import_id,
                vespa_connection_pool=vespa_connection_pool,
                grouping_max=5_000,
            )

            final_responses = []
            async for vespa_passages in final_passages_generator:
                final_responses.append(vespa_passages)

            # Assert the final responses are not the same as the original responses
            assert len(final_responses) == len(responses)
            assert final_responses != responses

            # Assert that for each passage that the only concepts present were those from the
            # aggregated inference results.
            print(aggregated_inference_results)
            # for response in final_responses:
            #     breakpoint()
            #     passage_concepts_in_s3 = aggregated_inference_results[
            #         response.text_block_id.value
            #     ]
            #     # vespa_response_concepts = response[]
            #     assert response
