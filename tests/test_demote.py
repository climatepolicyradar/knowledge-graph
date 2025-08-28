import os
from unittest.mock import Mock, patch

import pytest
import typer
from moto import mock_aws

from scripts.demote import main
from src.cloud import AwsEnv


@pytest.mark.parametrize(
    ("wikibase_id", "aws_env", "logged_in", "expected_exception"),
    [
        ("Q123", AwsEnv.labs, True, None),
        ("Q456", AwsEnv.staging, True, None),
        ("Q789", AwsEnv.labs, False, typer.BadParameter),
    ],
)
@mock_aws
def test_main(wikibase_id, aws_env, logged_in, expected_exception, monkeypatch):
    os.environ["USE_AWS_PROFILES"] = "false"

    # Mock wandb artifact
    artifact_mock = Mock()
    artifact_mock.aliases = Mock()
    artifact_mock.aliases.remove = Mock()
    artifact_mock.save = Mock()

    # Mock wandb API
    api_mock = Mock()
    api_mock.artifact.return_value = artifact_mock

    monkeypatch.setattr("wandb.Api", lambda: api_mock)

    with patch("scripts.demote.is_logged_in", return_value=logged_in):
        if expected_exception:
            with pytest.raises(expected_exception):
                main(wikibase_id=wikibase_id, aws_env=aws_env)
                api_mock.artifact.assert_not_called()

        else:
            main(wikibase_id=wikibase_id, aws_env=aws_env)
            artifact_mock.aliases.remove.assert_called_once_with(aws_env.value)
