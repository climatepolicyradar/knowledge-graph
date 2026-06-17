import base64
from unittest.mock import patch

from pydantic import SecretStr

from flows.snowflake import (
    SNOWFLAKE_ACCOUNT_SSM,
    SNOWFLAKE_PRIVATE_KEY_SSM,
    SNOWFLAKE_USER_SSM,
    get_snowflake_credentials,
)
from knowledge_graph.cloud import AwsEnv


def test_get_snowflake_credentials_resolves_and_decodes():
    ssm_values = {
        SNOWFLAKE_ACCOUNT_SSM: "test_account",
        SNOWFLAKE_USER_SSM: "svc_user",
        # Private key is stored base64-encoded in SSM and decoded on the way out.
        SNOWFLAKE_PRIVATE_KEY_SSM: base64.b64encode(b"PEM_KEY_DATA").decode("utf-8"),
    }

    with patch(
        "flows.snowflake.get_aws_ssm_param",
        side_effect=lambda name, aws_env: ssm_values[name],
    ):
        account, user, private_key = get_snowflake_credentials(AwsEnv.production)

    assert account == "test_account"
    assert user == "svc_user"
    assert isinstance(private_key, SecretStr)
    assert private_key.get_secret_value() == "PEM_KEY_DATA"


def test_get_snowflake_credentials_passes_aws_env_through():
    with patch(
        "flows.snowflake.get_aws_ssm_param",
        return_value=base64.b64encode(b"x").decode("utf-8"),
    ) as mock_ssm:
        get_snowflake_credentials(AwsEnv.staging)

    assert mock_ssm.call_count == 3
    assert all(
        call.kwargs["aws_env"] == AwsEnv.staging for call in mock_ssm.call_args_list
    )
