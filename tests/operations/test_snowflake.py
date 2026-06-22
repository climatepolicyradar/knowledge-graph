import base64
from unittest.mock import MagicMock, patch

import pytest
import snowflake.connector
from pydantic import SecretStr

from knowledge_graph.cloud import AwsEnv
from knowledge_graph.operations.snowflake import (
    SNOWFLAKE_ACCOUNT_SSM,
    SNOWFLAKE_DATABASE,
    SNOWFLAKE_PRIVATE_KEY_SSM,
    SNOWFLAKE_SCHEMA,
    SNOWFLAKE_USER_SSM,
    SNOWFLAKE_WAREHOUSE,
    connect_to_snowflake,
    get_snowflake_credentials,
)


@pytest.fixture
def mock_connect():
    """Patch the snowflake driver and PEM loader; yields the connect mock."""
    mock_private_key = MagicMock()
    mock_private_key.private_bytes.return_value = b"fake_der_bytes"

    with (
        patch("snowflake.connector.connect", return_value=MagicMock()) as connect_mock,
        patch(
            "knowledge_graph.operations.snowflake.load_pem_private_key",
            return_value=mock_private_key,
        ),
    ):
        yield connect_mock


def test_connect_uses_key_pair_when_all_credentials_provided(mock_connect):
    connect_to_snowflake(
        snowflake_user="svc_user",
        snowflake_private_key="fake_pem_key",
        snowflake_account="test_account",
    )

    kwargs = mock_connect.call_args.kwargs
    assert "connection_name" not in kwargs
    assert kwargs["user"] == "svc_user"
    assert kwargs["account"] == "test_account"
    assert kwargs["private_key"] == b"fake_der_bytes"
    assert kwargs["warehouse"] == SNOWFLAKE_WAREHOUSE
    assert kwargs["database"] == SNOWFLAKE_DATABASE
    assert kwargs["schema"] == SNOWFLAKE_SCHEMA


def test_connect_falls_back_to_local_when_no_credentials(mock_connect):
    connect_to_snowflake()

    assert mock_connect.call_args.kwargs.get("connection_name") == "local_dev"


def test_connect_raises_on_partial_credentials(mock_connect):
    # Only one of the three credentials present → ambiguous config, must not
    # silently fall back to local_dev.
    with pytest.raises(ValueError, match="Partial Snowflake credentials"):
        connect_to_snowflake(snowflake_user="svc_user")

    mock_connect.assert_not_called()


def test_connect_reraises_local_connection_error():
    with patch(
        "snowflake.connector.connect",
        side_effect=snowflake.connector.errors.Error("boom"),
    ):
        with pytest.raises(snowflake.connector.errors.Error):
            connect_to_snowflake()


def test_get_snowflake_credentials_resolves_and_decodes():
    ssm_values = {
        SNOWFLAKE_ACCOUNT_SSM: "test_account",
        SNOWFLAKE_USER_SSM: "svc_user",
        # Private key is stored base64-encoded in SSM and decoded on the way out.
        SNOWFLAKE_PRIVATE_KEY_SSM: base64.b64encode(b"PEM_KEY_DATA").decode("utf-8"),
    }

    with patch(
        "knowledge_graph.operations.snowflake.get_aws_ssm_param",
        side_effect=lambda name, aws_env: ssm_values[name],
    ):
        account, user, private_key = get_snowflake_credentials(AwsEnv.production)

    assert account == "test_account"
    assert user == "svc_user"
    assert isinstance(private_key, SecretStr)
    assert private_key.get_secret_value() == "PEM_KEY_DATA"


def test_get_snowflake_credentials_passes_aws_env_through():
    with patch(
        "knowledge_graph.operations.snowflake.get_aws_ssm_param",
        return_value=base64.b64encode(b"x").decode("utf-8"),
    ) as mock_ssm:
        get_snowflake_credentials(AwsEnv.staging)

    assert mock_ssm.call_count == 3
    assert all(
        call.kwargs["aws_env"] == AwsEnv.staging for call in mock_ssm.call_args_list
    )
