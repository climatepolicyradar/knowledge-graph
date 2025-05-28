import pytest
from cpr_sdk.search_adaptors import VespaSearchAdapter

from flows.aggregate_inference_results import S3Uri
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
            await index_aggregate_results_from_s3_to_vespa(
                s3_uri=S3Uri(bucket=mock_bucket, key=file_key),
                vespa_connection_pool=vespa_connection_pool,
            )

    # Assert that for each passage that the only concepts present were those from the
    # aggregated inference results.
