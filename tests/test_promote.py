import os
from contextlib import nullcontext
from unittest.mock import Mock, patch

import pytest
import typer
from moto import mock_aws

from scripts.cloud import AwsEnv
from scripts.promote import (
    check_existing_artifact_aliases,
    find_artifact_in_registry,
)
from src.identifiers import Identifier


@pytest.mark.parametrize(
    ("test_case", "logged_in", "expected_exception"),
    [
        (
            {  # Labs environment, logged in
                "wikibase_id": "Q123",
                "classifier_id": "abcd2345",
                "aws_env": AwsEnv.labs,
                "primary": False,
            },
            True,
            None,
        ),
        (
            {  # Staging environment, logged in, primary
                "wikibase_id": "Q456",
                "classifier_id": "abcd2345",
                "aws_env": AwsEnv.staging,
                "primary": True,
            },
            True,
            None,
        ),
        (
            {  # Labs environment, logged in
                "wikibase_id": "Q789",
                "classifier_id": "abcd2345",
                "aws_env": AwsEnv.labs,
                "primary": False,
            },
            True,
            None,
        ),
        (
            {  # Labs environment, not logged in
                "wikibase_id": "Q789",
                "classifier_id": "abcd2345",
                "aws_env": AwsEnv.labs,
                "primary": False,
            },
            False,
            typer.BadParameter,
        ),
    ],
)
@mock_aws
def test_main(test_case, logged_in, expected_exception, monkeypatch):
    from scripts.promote import main

    os.environ["USE_AWS_PROFILES"] = "false"
    os.environ["WANDB_API_KEY"] = "test_wandb_api_key"

    init_mock = Mock(return_value=nullcontext(Mock()))

    artifact_mock = Mock()
    artifact_mock.version = "v1"

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


@pytest.mark.parametrize(
    "collection_exists,artifact_aliases,classifier_id,aws_env,target_path,expected_error",
    [
        # Collection doesn't exist
        (
            False,
            [],
            Identifier("abcd2345"),
            AwsEnv.labs,
            "test/path",
            None,
        ),
        # Artifact exists with no conflicting aliases
        (
            True,
            ["labs"],
            Identifier("abcd2345"),
            AwsEnv.labs,
            "test/path",
            None,
        ),
        # Artifact exists with conflicting alias
        (
            True,
            ["staging", "labs"],
            Identifier("abcd2345"),
            AwsEnv.labs,
            "test/path",
            "Something has gone wrong with the source artifact",
        ),
    ],
)
def test_check_existing_artifact_aliases(
    collection_exists,
    artifact_aliases,
    classifier_id,
    aws_env,
    target_path,
    expected_error,
):
    """Test checking for existing artifacts with conflicting aliases."""
    mock_api = Mock()
    mock_api.artifact_collection_exists.return_value = collection_exists

    if collection_exists:
        other_mock_artifact = Mock(
            aliases=[],
            source_name="abcd2345:v2",
        )
        mock_artifact = Mock(
            aliases=artifact_aliases,
            source_name=f"{classifier_id}:v1",
        )
        mock_collection = Mock()
        mock_collection.artifacts.return_value = [other_mock_artifact, mock_artifact]
        mock_api.artifact_collection.return_value = mock_collection

    if expected_error:
        with pytest.raises(typer.BadParameter, match=expected_error):
            check_existing_artifact_aliases(
                mock_api,
                target_path,
                classifier_id,
                aws_env,
            )
    else:
        check_existing_artifact_aliases(
            mock_api,
            target_path,
            classifier_id,
            aws_env,
        )


@pytest.mark.parametrize(
    "artifacts,aws_env,expected",
    [
        # No artifacts
        ([], AwsEnv.labs, None),
        # One matching artifact
        (
            [Mock(source_name="abcd2345:v1", aliases=["prod"])],
            AwsEnv.production,
            "snapshot1",
        ),
        # Multiple artifacts, one match
        (
            [
                Mock(source_name="abcd2345:v1", aliases=["prod"]),
                Mock(source_name="abcd2345:v2", aliases=["staging"]),
            ],
            AwsEnv.production,
            "snapshot1",
        ),
        # Multiple artifacts, no match
        (
            [
                Mock(source_name="abcd2345:v1", aliases=["prod"]),
                Mock(source_name="abcd2345:v2", aliases=["staging"]),
            ],
            AwsEnv.sandbox,
            None,
        ),
    ],
)
def test_find_artifact_in_registry(artifacts, aws_env, expected):
    """Test finding artifacts by aws_env."""
    mock_collection = Mock()
    mock_collection.artifacts.return_value = artifacts

    result = find_artifact_in_registry(mock_collection, Identifier("abcd2345"), aws_env)

    if expected is None:
        assert result is None
    else:
        assert result
