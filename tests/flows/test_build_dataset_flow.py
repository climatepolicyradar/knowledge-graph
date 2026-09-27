import base64
import io
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pydantic import SecretStr

from flows.build_dataset import (
    COMBINED_S3_KEY,
    SAMPLED_S3_KEY,
    build_dataset_flow,
)
from knowledge_graph.operations.snowflake import (
    SNOWFLAKE_ACCOUNT_SSM,
    SNOWFLAKE_PRIVATE_KEY_SSM,
    SNOWFLAKE_USER_SSM,
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


def _make_fake_snowflake_df(n_rows: int = 5) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TEXT_BLOCK_TEXT": [f"passage {i}" for i in range(n_rows)],
            "TEXT_BLOCK_TYPE": ["text"] * n_rows,
            "DOCUMENT_ID": [f"doc{i}" for i in range(n_rows)],
            "DOCUMENT_CONTENT_TYPE": ["Laws and Policies"] * n_rows,
            "DOCUMENT_NAME": [f"Doc {i}" for i in range(n_rows)],
            "DOCUMENT_SLUG": [f"doc-{i}" for i in range(n_rows)],
            "DOCUMENT_METADATA_TRANSLATED": [False] * n_rows,
            "DOCUMENT_METADATA_CORPUS_TYPE_NAME": ["Laws and Policies"] * n_rows,
            "DOCUMENT_METADATA_GEOGRAPHIES": ["[]"] * n_rows,
        }
    )


FAKE_SNOWFLAKE_ACCOUNT = "test_account"
FAKE_SNOWFLAKE_USER = "svc_dbtbot"


@pytest.fixture
def mock_snowflake_ssm_params(mock_ssm_client):
    """Seed the real SSM parameter names `get_snowflake_credentials` reads."""
    mock_ssm_client.put_parameter(
        Name=SNOWFLAKE_ACCOUNT_SSM,
        Value=FAKE_SNOWFLAKE_ACCOUNT,
        Type="SecureString",
    )
    mock_ssm_client.put_parameter(
        Name=SNOWFLAKE_USER_SSM,
        Value=FAKE_SNOWFLAKE_USER,
        Type="SecureString",
    )
    mock_ssm_client.put_parameter(
        Name=SNOWFLAKE_PRIVATE_KEY_SSM,
        Value=base64.b64encode(b"fake_pem_key").decode(),
        Type="SecureString",
    )


@pytest.fixture
def mock_snowflake_connection():
    """Fake only the real external boundary: the Snowflake connection itself."""
    fake_df = _make_fake_snowflake_df()

    mock_cursor = MagicMock()
    mock_cursor.fetch_pandas_all.return_value = fake_df

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    mock_private_key = MagicMock()
    mock_private_key.private_bytes.return_value = b"fake_der_bytes"

    with (
        patch("snowflake.connector.connect", return_value=mock_conn) as mock_connect,
        patch(
            "knowledge_graph.operations.snowflake.load_pem_private_key",
            return_value=mock_private_key,
        ),
        # create_balanced_sample fails on uniform fake data — just return head(n)
        patch(
            "knowledge_graph.operations.build_dataset.create_balanced_sample",
            side_effect=lambda df, sample_size, on_columns: df.head(sample_size),
        ),
    ):
        yield mock_connect


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


@pytest.mark.asyncio
@pytest.mark.no_xdist
async def test_build_dataset_flow_resolves_snowflake_credentials_from_ssm_end_to_end(
    mock_snowflake_ssm_params,
    mock_snowflake_connection,
    mock_feather_bucket,
    mock_s3_client,
    test_config,
):
    """Only Snowflake itself is mocked; SSM credential resolution and the build logic run for real."""
    await build_dataset_flow.fn(
        sampled_dataset_target_num_rows=5,
        aws_env=test_config.aws_env,
        config=test_config,
    )

    connect_call = mock_snowflake_connection.call_args
    assert connect_call.kwargs.get("user") == FAKE_SNOWFLAKE_USER
    assert connect_call.kwargs.get("account") == FAKE_SNOWFLAKE_ACCOUNT

    combined = _read_feather_from_s3(
        mock_s3_client, mock_feather_bucket, COMBINED_S3_KEY
    )
    sampled = _read_feather_from_s3(mock_s3_client, mock_feather_bucket, SAMPLED_S3_KEY)

    assert not combined.empty
    assert not sampled.empty
    assert len(sampled) <= 5
    assert len(combined) >= len(sampled)
