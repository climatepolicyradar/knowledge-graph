import os
from contextlib import nullcontext
from unittest.mock import Mock, patch

import pytest
import typer
from moto import mock_aws

from knowledge_graph.cloud import AwsEnv
from scripts.demote import main


@pytest.mark.parametrize(
    (
        "wikibase_id",
        "aws_env",
        "logged_in",
        "expected_exception",
        "wandb_registry_version",
    ),
    [
        ("Q123", AwsEnv.labs, True, None, None),
        ("Q456", AwsEnv.staging, True, None, None),
        ("Q789", AwsEnv.labs, False, typer.BadParameter, None),
        ("Q999", AwsEnv.labs, True, None, "v10"),
    ],
)
@mock_aws
def test_main(
    wikibase_id,
    aws_env,
    logged_in,
    expected_exception,
    wandb_registry_version,
    monkeypatch,
):
    os.environ["USE_AWS_PROFILES"] = "false"
    os.environ["WANDB_API_KEY"] = "test_wandb_api_key"

    # Mock wandb artifact
    artifact_mock = Mock()
    artifact_mock.tags = Mock()
    artifact_mock.metadata = {
        "aws_env": aws_env.value,
    }
    artifact_mock.tags.remove = Mock()
    artifact_mock.save = Mock()

    # Mock wandb API
    api_mock = Mock()
    artifacts_mock = [
        Mock(version="v1", metadata={"aws_env": aws_env.value}),
        Mock(version="v2", metadata={"aws_env": aws_env.value}),
    ]
    mock_registries = Mock()
    api_mock.registries.return_value = mock_registries

    mock_collections = Mock()
    mock_registries.collections.return_value = mock_collections

    mock_collections.versions.return_value = artifacts_mock

    # Mock run
    run_mock = Mock()
    run_mock.use_artifact.return_value = artifact_mock

    init_mock = Mock(return_value=nullcontext(run_mock))

    monkeypatch.setattr("wandb.Api", lambda: api_mock)
    monkeypatch.setattr("wandb.init", init_mock)

    with patch("scripts.demote.is_logged_in", return_value=logged_in):
        if expected_exception:
            with pytest.raises(expected_exception):
                main(
                    wikibase_id=wikibase_id,
                    aws_env=aws_env,
                    wandb_registry_version=wandb_registry_version,
                )
                api_mock.registries.assert_not_called()
                artifact_mock.tags.remove.assert_not_called()

        else:
            main(
                wikibase_id=wikibase_id,
                aws_env=aws_env,
                wandb_registry_version=wandb_registry_version,
            )
            artifact_mock.tags.remove.assert_called_once_with(aws_env.value)
            if wandb_registry_version:
                api_mock.registries.assert_not_called()
            else:
                api_mock.registries.assert_called_once()
