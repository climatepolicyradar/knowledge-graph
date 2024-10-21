from unittest.mock import Mock, patch

import botocore
import pytest
from moto import mock_aws

from scripts.cloud import AwsEnv, is_logged_in


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
