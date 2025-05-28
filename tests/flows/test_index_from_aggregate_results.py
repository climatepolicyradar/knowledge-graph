def test_index_from_aggregated_inference_results(
    mock_s3_client,
    mock_bucket,
    mock_bucket_inference_results,
    s3_prefix_inference_results,
) -> None:
    """Test that we loaded the inference results from the mock bucket."""

    # TODO: The aggregate inference results function returns a unique s3 sub prefix,
    # move this functionality into a parent flow that takes the prefix as a parameter
    # and passes into the subflow once we have developed it.
    response = mock_s3_client.list_objects_v2(
        Bucket=mock_bucket, Prefix=s3_prefix_inference_results
    )
    assert "Contents" in response, "No objects found in bucket"
    assert len(response["Contents"]) > 0, "Empty response contents"

    file_keys = [
        obj["Key"] for obj in response["Contents"] if obj["Key"].endswith(".json")
    ]
    assert file_keys, "No JSON files found in bucket"

    for file_key in file_keys:
        response = mock_s3_client.get_object(Bucket=mock_bucket, Key=file_key)

        # TODO: Run the flow to index the results.
