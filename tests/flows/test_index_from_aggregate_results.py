def test_index_from_aggregated_inference_results(
    mock_s3_client,
    mock_bucket,
    mock_bucket_inference_results,
    s3_prefix_inference_results,
) -> None:
    """Test that we loaded the inference results from the mock bucket."""

    # TODO: Remove when implementing functionality.
    response = mock_s3_client.list_objects_v2(
        Bucket=mock_bucket, Prefix=s3_prefix_inference_results
    )
    assert "Contents" in response and len(response["Contents"]) > 0

    # TODO: Run the flow to index the results.
