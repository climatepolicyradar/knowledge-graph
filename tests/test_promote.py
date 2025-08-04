import os
from unittest.mock import Mock

import pytest
import typer
from moto import mock_aws

from scripts.cloud import AwsEnv
from scripts.promote import (
    check_existing_artifact_aliases,
    find_artifact_by_version,
)
from src.version import Version


@pytest.mark.parametrize(
    "wikibase_id, classifier, version, aws_env, primary, expected_exception",
    [
        (
            "Q123",
            "TestClassifier",
            Version("v1"),
            AwsEnv.labs,
            False,
            None,
        ),
        (
            "Q456",
            "AnotherClassifier",
            Version("v2"),
            AwsEnv.staging,
            True,
            None,
        ),
        (
            "Q789",
            "ThirdClassifier",
            Version("v3"),
            AwsEnv.labs,
            False,
            None,
        ),
    ],
)
@mock_aws
def test_main(
    wikibase_id,
    classifier,
    version,
    aws_env,
    primary,
    expected_exception,
    monkeypatch,
):
    from scripts.promote import main

    os.environ["USE_AWS_PROFILES"] = "false"
    os.environ["WANDB_API_KEY"] = "test_wandb_api_key"

    init_mock = Mock()
    init_mock.return_value.__enter__ = Mock()
    init_mock.return_value.__exit__ = Mock()

    monkeypatch.setattr("wandb.init", init_mock)
    monkeypatch.setattr("wandb.Artifact", lambda **kwargs: Mock())

    monkeypatch.setattr("os.environ.__setitem__", lambda *args: None)

    if expected_exception:
        with pytest.raises(expected_exception):
            main(
                wikibase_id=wikibase_id,
                classifier=classifier,
                version=version,
                aws_env=aws_env,
                primary=primary,
            )
    else:
        main(
            wikibase_id=wikibase_id,
            classifier=classifier,
            version=version,
            aws_env=aws_env,
            primary=primary,
        )


@pytest.mark.parametrize(
    "collection_exists,artifact_aliases,version,aws_env,target_path,expected_error",
    [
        # Collection doesn't exist
        (
            False,
            [],
            Version("v1"),
            AwsEnv.labs,
            "test/path",
            None,
        ),
        # Artifact exists with no conflicting aliases
        (
            True,
            ["labs"],
            Version("v1"),
            AwsEnv.labs,
            "test/path",
            None,
        ),
        # Artifact exists with conflicting alias
        (
            True,
            ["staging"],
            Version("v1"),
            AwsEnv.labs,
            "test/path",
            "An artifact already exists with AWS environment aliases {'staging'}",
        ),
    ],
)
def test_check_existing_artifact_aliases(
    collection_exists,
    artifact_aliases,
    version,
    aws_env,
    target_path,
    expected_error,
):
    """Test checking for existing artifacts with conflicting aliases."""
    mock_api = Mock()
    mock_api.artifact_collection_exists.return_value = collection_exists

    if collection_exists:
        mock_artifact = Mock(aliases=artifact_aliases, version=str(version))
        mock_collection = Mock()
        mock_collection.artifacts.return_value = [mock_artifact]
        mock_api.artifact_collection.return_value = mock_collection

    if expected_error:
        with pytest.raises(typer.BadParameter, match=expected_error):
            check_existing_artifact_aliases(
                mock_api,
                target_path,
                version,
                aws_env,
            )
    else:
        check_existing_artifact_aliases(
            mock_api,
            target_path,
            version,
            aws_env,
        )


@pytest.mark.parametrize(
    "artifacts,version,expected",
    [
        # No artifacts
        ([], "v1", None),
        # One matching artifact
        ([Mock(version="v1", aliases=["prod"])], "v1", "snapshot1"),
        # Multiple artifacts, one match
        (
            [
                Mock(version="v1", aliases=["prod"]),
                Mock(version="v2", aliases=["staging"]),
            ],
            "v1",
            "snapshot1",
        ),
        # Multiple artifacts, no match
        (
            [
                Mock(version="v1", aliases=["prod"]),
                Mock(version="v2", aliases=["staging"]),
            ],
            "v3",
            None,
        ),
    ],
)
def test_find_artifact_by_version(artifacts, version, expected):
    """Test finding artifacts by version."""
    mock_collection = Mock()
    mock_collection.artifacts.return_value = artifacts

    result = find_artifact_by_version(mock_collection, Version(version))

    if expected is None:
        assert result is None
    else:
        assert result
