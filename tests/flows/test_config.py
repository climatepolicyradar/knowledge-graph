import pytest
from pydantic import ValidationError

from flows.config import Config


def test_invalid_s3_prefix_without_trailing_slash():
    """Test that S3 prefixes without trailing slash are rejected."""
    with pytest.raises(ValidationError) as exc_info:
        Config(
            cache_bucket="test-bucket",
            aggregate_document_source_prefix="invalid_prefix",
        )
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("aggregate_document_source_prefix",)
    assert "must end with '/'" in errors[0]["msg"]


def test_valid_s3_prefix_with_trailing_slash():
    """Test that valid S3 prefixes with trailing slash are accepted."""
    config = Config(
        cache_bucket="test-bucket",
        aggregate_document_source_prefix="valid_prefix/",
        aggregate_inference_results_prefix="another_valid_prefix/",
        inference_document_source_prefix="embeddings/",
        inference_document_target_prefix="labelled/",
        pipeline_state_prefix="input/",
    )
    assert config.aggregate_document_source_prefix == "valid_prefix/"
    assert config.aggregate_inference_results_prefix == "another_valid_prefix/"
    assert config.inference_document_source_prefix == "embeddings/"
    assert config.inference_document_target_prefix == "labelled/"
    assert config.pipeline_state_prefix == "input/"


def test_invalid_s3_prefix_with_leading_slash():
    """Test that S3 prefixes with leading slash are rejected."""
    with pytest.raises(ValidationError) as exc_info:
        Config(
            cache_bucket="test-bucket",
            aggregate_document_source_prefix="/invalid_prefix",
        )
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("aggregate_document_source_prefix",)
    assert "should not start with '/'" in errors[0]["msg"]


def test_invalid_empty_s3_prefix():
    """Test that empty S3 prefixes are rejected."""
    with pytest.raises(ValidationError) as exc_info:
        Config(
            cache_bucket="test-bucket",
            aggregate_document_source_prefix="",
        )
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("aggregate_document_source_prefix",)
    assert "cannot be empty" in errors[0]["msg"]


def test_multiple_invalid_prefixes():
    """Test that multiple invalid prefixes are all caught."""
    with pytest.raises(ValidationError) as exc_info:
        Config(
            cache_bucket="test-bucket",
            aggregate_document_source_prefix="/invalid1",
            inference_document_source_prefix="",
            pipeline_state_prefix="/invalid2",
        )
    errors = exc_info.value.errors()
    assert len(errors) == 3
    error_locs = {e["loc"] for e in errors}
    assert ("aggregate_document_source_prefix",) in error_locs
    assert ("inference_document_source_prefix",) in error_locs
    assert ("pipeline_state_prefix",) in error_locs


def test_default_values_are_valid():
    """Test that the default S3 prefix values pass validation."""
    config = Config(cache_bucket="test-bucket")
    assert config.aggregate_document_source_prefix
    assert config.aggregate_inference_results_prefix
    assert config.inference_document_source_prefix
    assert config.inference_document_target_prefix
    assert config.pipeline_state_prefix


def test_nested_s3_prefix_without_trailing_slash():
    """Test that nested S3 prefixes without trailing slash are rejected."""
    with pytest.raises(ValidationError) as exc_info:
        Config(
            cache_bucket="test-bucket",
            aggregate_document_source_prefix="path/to/documents",
        )
    errors = exc_info.value.errors()
    assert len(errors) == 1
    assert errors[0]["loc"] == ("aggregate_document_source_prefix",)
    assert "must end with '/'" in errors[0]["msg"]


def test_nested_s3_prefix_with_trailing_slash():
    """Test that nested S3 prefixes with trailing slash are accepted."""
    config = Config(
        cache_bucket="test-bucket",
        aggregate_document_source_prefix="path/to/documents/",
    )
    assert config.aggregate_document_source_prefix == "path/to/documents/"
