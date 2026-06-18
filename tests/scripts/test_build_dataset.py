from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from scripts.build_dataset import get_world_bank_region, run_build_dataset


def _make_fake_snowflake_df(n_rows: int = 20) -> pd.DataFrame:
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


@pytest.fixture
def mock_snowflake_connection():
    fake_df = _make_fake_snowflake_df(n_rows=20)

    mock_cursor = MagicMock()
    mock_cursor.fetch_pandas_all.return_value = fake_df

    mock_conn = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    # Mock load_pem_private_key so tests don't need a real PEM key
    mock_private_key = MagicMock()
    mock_private_key.private_bytes.return_value = b"fake_der_bytes"

    with (
        patch("snowflake.connector.connect", return_value=mock_conn) as mock_connect,
        patch(
            "scripts.build_dataset.load_pem_private_key", return_value=mock_private_key
        ),
        # create_balanced_sample fails on uniform fake data — just return head(n)
        patch(
            "scripts.build_dataset.create_balanced_sample",
            side_effect=lambda df, sample_size, on_columns: df.head(sample_size),
        ),
    ):
        yield mock_connect, mock_conn, mock_cursor


def test_run_build_dataset_returns_two_dataframes(mock_snowflake_connection):
    combined_df, sampled_df = run_build_dataset(n=5)

    assert isinstance(combined_df, pd.DataFrame)
    assert isinstance(sampled_df, pd.DataFrame)


def test_run_build_dataset_combined_has_expected_columns(mock_snowflake_connection):
    combined_df, _ = run_build_dataset(n=5)

    for col in [
        "text_block.text",
        "document_id",
        "world_bank_region",
        "document_metadata.corpus_type_name",
        "translated",
    ]:
        assert col in combined_df.columns


def test_run_build_dataset_sampled_has_expected_columns(mock_snowflake_connection):
    _, sampled_df = run_build_dataset(n=5)

    for col in [
        "text_block.text",
        "document_id",
        "world_bank_region",
        "document_metadata.corpus_type_name",
        "translated",
    ]:
        assert col in sampled_df.columns


def test_run_build_dataset_sampled_does_not_exceed_n(mock_snowflake_connection):
    n = 5
    _, sampled_df = run_build_dataset(n=n)

    assert len(sampled_df) <= n


def test_run_build_dataset_combined_larger_than_sampled(mock_snowflake_connection):
    combined_df, sampled_df = run_build_dataset(n=5)

    assert len(combined_df) >= len(sampled_df)


def test_run_build_dataset_uses_explicit_credentials_when_provided(
    mock_snowflake_connection,
):
    mock_connect, _, _ = mock_snowflake_connection

    run_build_dataset(
        n=5,
        snowflake_user="svc_user",
        snowflake_private_key="fake_pem_key",
        snowflake_account="test_account",
    )

    connect_call = mock_connect.call_args
    assert "connection_name" not in (connect_call.kwargs or {})
    assert connect_call.kwargs.get("user") == "svc_user"
    assert connect_call.kwargs.get("account") == "test_account"


def test_run_build_dataset_falls_back_to_local_without_credentials(
    mock_snowflake_connection,
):
    mock_connect, _, _ = mock_snowflake_connection

    run_build_dataset(n=5)

    connect_call = mock_connect.call_args
    assert connect_call.kwargs.get("connection_name") == "local_dev"


def test_get_world_bank_region_returns_none_for_none_input():
    assert get_world_bank_region(None) is None


def test_get_world_bank_region_returns_none_for_empty_list():
    assert get_world_bank_region([]) is None


def test_get_world_bank_region_returns_none_for_unknown_iso_code():
    assert get_world_bank_region(["UNKNOWN_XYZ"]) is None


def test_get_world_bank_region_returns_string_for_valid_iso_code():
    result = get_world_bank_region(["GBR"])
    assert result is not None
    assert isinstance(result, str)


def test_get_world_bank_region_uses_first_iso_code_only():
    result_first = get_world_bank_region(["GBR"])
    result_second = get_world_bank_region(["GBR", "UNKNOWN_XYZ"])
    assert result_first == result_second


def test_get_world_bank_region_handles_non_list_input():
    assert get_world_bank_region("not-a-list") is None


def test_run_build_dataset_includes_corpus_type_in_sql_when_provided(
    mock_snowflake_connection,
):
    _, _, mock_cursor = mock_snowflake_connection

    run_build_dataset(n=5, corpus_type="Litigation")

    execute_calls = mock_cursor.execute.call_args_list
    assert len(execute_calls) == 2
    for single_call in execute_calls:
        sql = single_call[0][0]
        assert "Litigation" in sql, (
            f"Expected corpus type filter in SQL but got: {sql[:200]}"
        )


def test_run_build_dataset_omits_corpus_type_filter_when_not_provided(
    mock_snowflake_connection,
):
    _, _, mock_cursor = mock_snowflake_connection

    run_build_dataset(n=5)

    execute_calls = mock_cursor.execute.call_args_list
    for single_call in execute_calls:
        sql = single_call[0][0]
        assert "METADATA_CORPUS_TYPE_NAME =" not in sql
