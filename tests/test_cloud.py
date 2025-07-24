from unittest.mock import Mock, patch

import botocore
import pytest
from moto import mock_aws

from scripts.cloud import (
    AwsEnv,
    function_to_flow_name,
    generate_deployment_name,
    is_logged_in,
    parse_aws_env,
)


def test_function_to_flow_name():
    assert function_to_flow_name(is_logged_in) == "is-logged-in"


def test_init_awsenv():
    assert AwsEnv.staging == AwsEnv("dev")


@pytest.mark.parametrize(
    "aws_env, use_aws_profiles, is_logged_in_result",
    [
        (AwsEnv.labs, True, True),
        (AwsEnv.sandbox, True, False),
        (AwsEnv.staging, True, True),
        (AwsEnv.production, True, False),
        (AwsEnv.labs, False, True),
        (AwsEnv.sandbox, False, False),
    ],
)
@mock_aws
def test_is_logged_in(aws_env, use_aws_profiles, is_logged_in_result):
    with patch("scripts.cloud.get_sts_client") as mock_get_sts_client:
        mock_sts = Mock()
        mock_get_sts_client.return_value = mock_sts

        if is_logged_in_result:
            mock_sts.get_caller_identity.return_value = {"Account": "123456789012"}
        else:
            mock_sts.get_caller_identity.side_effect = botocore.exceptions.ClientError(
                {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}},
                "GetCallerIdentity",
            )

        assert is_logged_in(aws_env, use_aws_profiles) == is_logged_in_result

        expected_aws_env = aws_env if use_aws_profiles else None
        mock_get_sts_client.assert_called_once_with(expected_aws_env)
        mock_sts.get_caller_identity.assert_called_once()


@pytest.mark.parametrize(
    "invalid_input",
    ["invalid", "test"],
)
def test_parse_aws_env_invalid(invalid_input):
    with pytest.raises(ValueError, match=f"'{invalid_input}' is not one of"):
        parse_aws_env(invalid_input)


@pytest.mark.parametrize(
    "flow_name, aws_env, expected",
    [
        ("inference", AwsEnv.labs, "kg-inference-labs"),
        ("aggregate", AwsEnv.sandbox, "kg-aggregate-sandbox"),
        ("index", AwsEnv.staging, "kg-index-staging"),
        ("full-pipeline", AwsEnv.production, "kg-full-pipeline-prod"),
    ],
)
def test_generate_deployment_name(flow_name, aws_env, expected):
    assert generate_deployment_name(flow_name, aws_env) == expected
