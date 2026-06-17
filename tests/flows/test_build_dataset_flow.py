import io
from contextlib import contextmanager
from unittest.mock import patch

import pandas as pd
import pytest
from pydantic import SecretStr

from flows.build_dataset import (
    COMBINED_S3_KEY,
    SAMPLED_S3_KEY,
    build_dataset_flow,
)


@pytest.fixture
def fake_combined_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "text_block.text": ["passage one", "passage two", "passage three"],
            "text_block.type": ["text", "text", "text"],
            "document_id": ["doc1", "doc2", "doc3"],
            "document_content_type": ["Laws and Policies"] * 3,
            "document_name": ["Doc One", "Doc Two", "Doc Three"],
            "document_slug": ["doc-1", "doc-2", "doc-3"],
            "translated": [False, False, True],
            "document_metadata.corpus_type_name": ["Laws and Policies"] * 3,
            "world_bank_region": ["Europe", "Africa", None],
        }
    )


@pytest.fixture
def fake_sampled_df(fake_combined_df) -> pd.DataFrame:
    return fake_combined_df.iloc[:2].copy()


@pytest.fixture
def mock_feather_bucket(mock_aws_creds, mock_s3_client, test_config) -> str:
    mock_s3_client.create_bucket(
        Bucket=test_config.dataset_s3_bucket,
        CreateBucketConfiguration={"LocationConstraint": "eu-west-1"},
    )
    return test_config.dataset_s3_bucket


def _read_feather_from_s3(s3_client, bucket: str, key: str) -> pd.DataFrame:
    response = s3_client.get_object(Bucket=bucket, Key=key)
    return pd.read_feather(io.BytesIO(response["Body"].read()))


@contextmanager
def _run_flow(fake_combined_df, fake_sampled_df, test_config):
    with (
        patch(
            "flows.build_dataset._set_up_build_dataset_environment",
            return_value=(
                test_config,
                "test_account",
                "test_user",
                SecretStr("fake_key"),
            ),
        ),
        patch(
            "flows.build_dataset.run_build_dataset",
            return_value=(fake_combined_df, fake_sampled_df),
        ),
    ):
        yield


@pytest.mark.asyncio
@pytest.mark.no_xdist
async def test_build_dataset_flow_uploads_both_files(
    fake_combined_df,
    fake_sampled_df,
    mock_feather_bucket,
    mock_s3_client,
    test_config,
):
    with _run_flow(fake_combined_df, fake_sampled_df, test_config):
        await build_dataset_flow.fn(aws_env=test_config.aws_env)

    combined = _read_feather_from_s3(
        mock_s3_client, mock_feather_bucket, COMBINED_S3_KEY
    )
    sampled = _read_feather_from_s3(mock_s3_client, mock_feather_bucket, SAMPLED_S3_KEY)

    assert not combined.empty
    assert not sampled.empty


@pytest.mark.asyncio
@pytest.mark.no_xdist
async def test_build_dataset_flow_s3_files_are_valid_feathers(
    fake_combined_df,
    fake_sampled_df,
    mock_feather_bucket,
    mock_s3_client,
    test_config,
):
    with _run_flow(fake_combined_df, fake_sampled_df, test_config):
        await build_dataset_flow.fn(aws_env=test_config.aws_env)

    combined = _read_feather_from_s3(
        mock_s3_client, mock_feather_bucket, COMBINED_S3_KEY
    )
    sampled = _read_feather_from_s3(mock_s3_client, mock_feather_bucket, SAMPLED_S3_KEY)

    for df in (combined, sampled):
        assert "text_block.text" in df.columns
        assert "document_id" in df.columns
        assert "world_bank_region" in df.columns


@pytest.mark.asyncio
@pytest.mark.no_xdist
async def test_build_dataset_flow_combined_larger_than_sampled(
    fake_combined_df,
    fake_sampled_df,
    mock_feather_bucket,
    mock_s3_client,
    test_config,
):
    with _run_flow(fake_combined_df, fake_sampled_df, test_config):
        await build_dataset_flow.fn(aws_env=test_config.aws_env)

    combined = _read_feather_from_s3(
        mock_s3_client, mock_feather_bucket, COMBINED_S3_KEY
    )
    sampled = _read_feather_from_s3(mock_s3_client, mock_feather_bucket, SAMPLED_S3_KEY)

    assert len(combined) >= len(sampled)


@pytest.mark.asyncio
@pytest.mark.no_xdist
async def test_build_dataset_flow_uses_correct_s3_keys(
    fake_combined_df,
    fake_sampled_df,
    mock_feather_bucket,
    mock_s3_client,
    test_config,
):
    with _run_flow(fake_combined_df, fake_sampled_df, test_config):
        await build_dataset_flow.fn(aws_env=test_config.aws_env)

    objects = mock_s3_client.list_objects_v2(Bucket=mock_feather_bucket)
    keys = {obj["Key"] for obj in objects.get("Contents", [])}

    assert COMBINED_S3_KEY in keys
    assert SAMPLED_S3_KEY in keys
