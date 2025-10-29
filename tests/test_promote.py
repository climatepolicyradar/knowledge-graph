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
                "aws_env": AwsEnv.labs,
            },
            True,
            None,
        ),
        (
            {  # Staging environment, logged in
                "wikibase_id": "Q456",
                "classifier_id": "abcd2345",
                "aws_env": AwsEnv.staging,
            },
            True,
            None,
        ),
        (
            {  # Labs environment, logged in
                "wikibase_id": "Q789",
                "classifier_id": "abcd2345",
                "aws_env": AwsEnv.labs,
            },
            True,
            None,
        ),
        (
            {  # Labs environment, not logged in
                "wikibase_id": "Q789",
                "classifier_id": "abcd2345",
                "aws_env": AwsEnv.labs,
            },
            False,
            typer.BadParameter,
        ),
        (
            {  # Labs environment, logged in, adding extra classifier profile
                "wikibase_id": "Q123",
                "classifier_id": "abcd2345",
                "aws_env": AwsEnv.labs,
                "add_classifiers_profiles": ["new_profile"],
            },
            True,
            typer.BadParameter,
        ),
        (
            {  # Labs environment, logged in, add classifier profile that exists
                "wikibase_id": "Q123",
                "classifier_id": "abcd2345",
                "aws_env": AwsEnv.labs,
                "add_classifiers_profiles": ["test_profile"],
            },
            True,
            None,
        ),
        (
            {  # Labs environment, logged in, remove existing classifier profile, add new one
                "wikibase_id": "Q123",
                "classifier_id": "abcd2345",
                "aws_env": AwsEnv.labs,
                "remove_classifiers_profiles": ["test_profile"],
                "add_classifiers_profiles": ["new_profile"],
            },
            True,
            None,
        ),
        (
            {  # Labs environment, logged in, add and remove same classifier profile
                "wikibase_id": "Q123",
                "classifier_id": "abcd2345",
                "aws_env": AwsEnv.labs,
                "remove_classifiers_profiles": ["new_profile"],
                "add_classifiers_profiles": ["new_profile"],
            },
            True,
            typer.BadParameter,
        ),
    ],
)
@mock_aws
def test_main(test_case, logged_in, expected_exception, monkeypatch):
    os.environ["USE_AWS_PROFILES"] = "false"
    os.environ["WANDB_API_KEY"] = "test_wandb_api_key"
    artifact_profile = "test_profile"
    # Mock the wandb.init function
    init_mock = Mock(return_value=nullcontext(Mock()))

    # Mock the artifact returned by use_artifact
    artifact_mock = Mock()
    artifact_mock.version = "v1"
    artifact_mock.metadata = {
        "classifiers_profiles": [artifact_profile],
    }
    artifact_mock.tags = []
    artifact_mock.save = Mock()

    # Mock the run object returned by wandb.init
    run_mock = Mock()
    run_mock.use_artifact.return_value = artifact_mock
    run_mock.link_artifact = Mock()

    # Mock the wandb.Api object
    api_mock = Mock()
    api_mock.artifact_collection_exists.return_value = False
    artifacts_mock = [
        Mock(version="v1", metadata={"aws_env": test_case["aws_env"].value}),
        Mock(version="v2", metadata={"aws_env": test_case["aws_env"].value}),
        Mock(version="v3", metadata={"aws_env": "other_env"}),
    ]
    api_mock.artifacts.return_value = artifacts_mock

    init_mock = Mock(return_value=nullcontext(run_mock))
    # Patch objects
    monkeypatch.setattr("wandb.init", init_mock)
    monkeypatch.setattr("wandb.Api", lambda: api_mock)
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
            api_mock.artifacts.asset_called_once_with(
                type_name="model",
                name=f"{test_case['wikibase_id']}/{test_case['classifier_id']}",
            )
            assert list(
                set(artifact_mock.metadata.get("classifiers_profiles"))
            ) == list(
                set(
                    (test_case.get("add_classifiers_profiles") or [])
                    + [artifact_profile]
                )
                - set(test_case.get("remove_classifiers_profiles") or [])
            )
