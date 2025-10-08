import os
from contextlib import nullcontext
from unittest.mock import Mock, patch

import pytest
import typer
from moto import mock_aws

from knowledge_graph.cloud import AwsEnv
from scripts.promote import main


@pytest.mark.parametrize(
    ("test_case", "logged_in", "expected_exception"),
    [
        (
            {  # Labs environment, logged in
                "wikibase_id": "Q123",
                "classifier_id": "abcd2345",
                "classifier_version": "v2",
                "aws_env": AwsEnv.labs,
            },
            True,
            None,
        ),
        (
            {  # Staging environment, logged in
                "wikibase_id": "Q456",
                "classifier_id": "abcd2345",
                "classifier_version": "v2",
                "aws_env": AwsEnv.staging,
            },
            True,
            None,
        ),
        (
            {  # Labs environment, logged in
                "wikibase_id": "Q789",
                "classifier_id": "abcd2345",
                "classifier_version": "v10",
                "aws_env": AwsEnv.labs,
            },
            True,
            None,
        ),
        (
            {  # Labs environment, not logged in
                "wikibase_id": "Q789",
                "classifier_id": "abcd2345",
                "classifier_version": "v10",
                "aws_env": AwsEnv.labs,
            },
            False,
            typer.BadParameter,
        ),
    ],
)
@mock_aws
def test_main(test_case, logged_in, expected_exception, monkeypatch):
    os.environ["USE_AWS_PROFILES"] = "false"
    os.environ["WANDB_API_KEY"] = "test_wandb_api_key"

    init_mock = Mock(return_value=nullcontext(Mock()))

    artifact_mock = Mock()
    artifact_mock.version = "v1"
    artifact_mock.metadata = {
        "classifiers_profiles": ["test_profile"],
        "aws_env": test_case["aws_env"].value,
    }
    artifact_mock.tags = []
    artifact_mock.save = Mock()

    run_mock = Mock()
    run_mock.use_artifact.return_value = artifact_mock
    run_mock.link_artifact = Mock()

    api_mock = Mock()
    api_mock.artifact_collection_exists.return_value = False

    init_mock = Mock(return_value=nullcontext(run_mock))
    monkeypatch.setattr("wandb.init", init_mock)
    monkeypatch.setattr("wandb.Artifact", lambda **kwargs: artifact_mock)
    monkeypatch.setattr("os.environ.__setitem__", lambda *args: None)

    with patch("scripts.promote.is_logged_in", return_value=logged_in):
        if expected_exception:
            with pytest.raises(expected_exception):
                main(**test_case)
        else:
            main(**test_case)
            assert test_case["aws_env"].value in artifact_mock.tags
            artifact_mock.save.assert_called()
