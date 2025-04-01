import os
from unittest.mock import ANY, Mock, call, patch

import pytest
import typer
from moto import mock_aws

from scripts.cloud import AwsEnv
from scripts.promote import (
    Across,
    Version,
    Within,
    check_existing_artifact_aliases,
    download,
    find_artifact_by_version,
    get_aliases,
    get_bucket_name_for_aws_env,
    get_object_key,
    upload,
    validate_logins,
)


@pytest.mark.parametrize("input_version", ["v1", "v10", "v0"])
def test_version_valid(input_version):
    assert str(Version(input_version)) == input_version


@pytest.mark.parametrize("invalid_version", ["1", "v", "v1.0", "latest"])
def test_version_invalid(invalid_version):
    with pytest.raises(ValueError):
        Version(invalid_version)


@pytest.mark.parametrize(
    "version_a,version_b,expected",
    [
        ("v1", "v2", True),
        ("v2", "v1", False),
        ("v10", "v2", False),
    ],
)
def test_version_comparison(version_a, version_b, expected):
    assert (Version(version_a) < Version(version_b)) == expected


def test_version_sorting():
    versions = [Version("v3"), Version("v1"), Version("v10"), Version("v2")]
    sorted_versions = sorted(versions)
    assert [str(v) for v in sorted_versions] == ["v1", "v2", "v3", "v10"]


def test_version_sorting_with_larger_numbers():
    versions = [
        Version("v3"),
        Version("v1"),
        Version("v10"),
        Version("v2"),
        Version("v20"),
    ]
    sorted_versions = sorted(versions)
    assert [str(v) for v in sorted_versions] == ["v1", "v2", "v3", "v10", "v20"]


@pytest.mark.parametrize(
    "wikibase_id, classifier, version, from_aws_env, to_aws_env, within_aws_env, primary, expected_exception",
    [
        (
            "Q123",
            "TestClassifier",
            Version("v1"),
            AwsEnv.labs,
            AwsEnv.staging,
            None,
            False,
            None,
        ),
        (
            "Q456",
            "AnotherClassifier",
            Version("v2"),
            AwsEnv.staging,
            AwsEnv.production,
            None,
            True,
            None,
        ),
        (
            "Q789",
            "ThirdClassifier",
            Version("v3"),
            None,
            None,
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
    from_aws_env,
    to_aws_env,
    within_aws_env,
    primary,
    expected_exception,
    monkeypatch,
):
    from scripts.promote import main

    os.environ["USE_AWS_PROFILES"] = "false"
    os.environ["WANDB_API_KEY"] = "test_wandb_api_key"

    mock_validate_login = Mock(return_value=None)

    monkeypatch.setattr(
        "scripts.promote.copy_across_aws_envs",
        lambda *args: ("mock-bucket", "mock-key"),
    )

    monkeypatch.setattr("wandb.init", lambda **kwargs: Mock())
    monkeypatch.setattr("wandb.Artifact", lambda **kwargs: Mock())

    monkeypatch.setattr("os.environ.__setitem__", lambda *args: None)

    with patch(
        "scripts.promote.validate_logins",
        new=mock_validate_login,
    ):
        if expected_exception:
            with pytest.raises(expected_exception):
                main(
                    wikibase_id=wikibase_id,
                    classifier=classifier,
                    version=version,
                    from_aws_env=from_aws_env,
                    to_aws_env=to_aws_env,
                    within_aws_env=within_aws_env,
                    primary=primary,
                )
        else:
            if from_aws_env is not None or to_aws_env is not None:
                with pytest.raises(
                    NotImplementedError,
                    match="Promotion across AWS environments is not yet implemented",
                ):
                    main(
                        wikibase_id=wikibase_id,
                        classifier=classifier,
                        version=version,
                        from_aws_env=from_aws_env,
                        to_aws_env=to_aws_env,
                        within_aws_env=within_aws_env,
                        primary=primary,
                    )
            else:
                main(
                    wikibase_id=wikibase_id,
                    classifier=classifier,
                    version=version,
                    from_aws_env=from_aws_env,
                    to_aws_env=to_aws_env,
                    within_aws_env=within_aws_env,
                    primary=primary,
                )


def test_version_latest_not_supported():
    with pytest.raises(ValueError, match="`latest` isn't yet supported"):
        Version("latest")


def test_version_equality():
    assert Version("v1") == Version("v1")
    assert Version("v1") == "v1"
    assert Version("v1") != Version("v2")
    assert Version("v1") != "v2"


def test_version_hash():
    versions = {Version("v1"), Version("v2"), Version("v1")}
    assert len(versions) == 2


@pytest.mark.parametrize(
    "promotion",
    [
        (Within(value=AwsEnv.labs, primary=False)),
        (Within(value=AwsEnv.labs, primary=True)),
        (Within(value=AwsEnv.staging, primary=True)),
        (Across(src=AwsEnv.labs, dst=AwsEnv.staging, primary=False)),
        (Across(src=AwsEnv.labs, dst=AwsEnv.staging, primary=True)),
        (Across(src=AwsEnv.staging, dst=AwsEnv.production, primary=True)),
    ],
)
def test_get_aliases(promotion, snapshot):
    assert get_aliases(promotion) == snapshot


@pytest.mark.parametrize(
    "aws_env, expected_bucket",
    [
        (AwsEnv.labs, "cpr-labs-models"),
        (AwsEnv.sandbox, "cpr-sandbox-models"),
        (AwsEnv.staging, "cpr-staging-models"),
        (AwsEnv.production, "cpr-prod-models"),
    ],
)
def test_get_bucket_name_for_aws_env(aws_env, expected_bucket):
    assert get_bucket_name_for_aws_env(aws_env) == expected_bucket


@pytest.mark.parametrize(
    "concept, classifier, version",
    [
        (
            "Q123",
            "TestClassifier",
            Version("v1"),
        ),
        (
            "Q456",
            "AnotherClassifier",
            Version("v2"),
        ),
        (
            "Q789",
            "ThirdClassifier",
            Version("v10"),
        ),
    ],
)
def test_get_object_key(concept, classifier, version, snapshot):
    assert str(get_object_key(concept, classifier, version)) == snapshot


@pytest.mark.parametrize(
    "promotion, login_states, expected_exception",
    [
        (
            Across(src=AwsEnv.labs, dst=AwsEnv.staging),
            {AwsEnv.labs: True, AwsEnv.staging: True},
            None,
        ),
        (
            Across(src=AwsEnv.labs, dst=AwsEnv.staging),
            {AwsEnv.labs: True, AwsEnv.staging: False},
            AwsEnv.staging,
        ),
        (
            Within(value=AwsEnv.labs),
            {AwsEnv.labs: True},
            None,
        ),
        (
            Within(value=AwsEnv.labs),
            {AwsEnv.labs: False},
            AwsEnv.labs,
        ),
    ],
)
def test_validate_logins(promotion, login_states, expected_exception):
    with patch("scripts.promote.is_logged_in") as mock_is_logged_in:
        mock_is_logged_in.side_effect = lambda env, _: login_states.get(env, False)

        if expected_exception:
            with pytest.raises(
                typer.BadParameter,
                match=f"you're not logged into {expected_exception.value}",
            ):
                validate_logins(promotion, False)
        else:
            validate_logins(promotion, False)

        expected_calls = (
            [((promotion.value, False),)]
            if isinstance(promotion, Within)
            else [((promotion.src, False),), ((promotion.dst, False),)]
        )
        assert mock_is_logged_in.call_args_list == expected_calls


@pytest.mark.parametrize(
    "collection_exists,artifact_aliases,version,promotion,target_path,expected_error",
    [
        # Collection doesn't exist
        (
            False,
            [],
            Version("v1"),
            Within(value=AwsEnv.labs, primary=True),
            "test/path",
            None,
        ),
        # Artifact exists with no conflicting aliases
        (
            True,
            ["labs"],
            Version("v1"),
            Within(value=AwsEnv.labs, primary=True),
            "test/path",
            None,
        ),
        # Artifact exists with conflicting alias
        (
            True,
            ["staging"],
            Version("v1"),
            Within(value=AwsEnv.labs, primary=True),
            "test/path",
            "An artifact already exists with AWS environment aliases {'staging'}",
        ),
    ],
)
def test_check_existing_artifact_aliases(
    collection_exists,
    artifact_aliases,
    version,
    promotion,
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
                promotion,
            )
    else:
        check_existing_artifact_aliases(
            mock_api,
            target_path,
            version,
            promotion,
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


@pytest.mark.parametrize(
    "promotion, concept, classifier, from_version, content",
    [
        (
            Across(src=AwsEnv.labs, dst=AwsEnv.staging),
            "Q123",
            "TestClassifier",
            Version("v1"),
            b"test content",
        ),
        (
            Across(src=AwsEnv.staging, dst=AwsEnv.production),
            "Q456",
            "AnotherClassifier",
            Version("v2"),
            b"another test content",
        ),
    ],
)
@mock_aws
def test_copy_across_aws_envs(
    promotion,
    concept,
    classifier,
    from_version,
    content,
):
    from scripts.promote import copy_across_aws_envs

    with patch("scripts.promote.get_s3_client") as mock_get_s3_client:
        mock_from_s3 = Mock()
        mock_to_s3 = Mock()
        mock_get_s3_client.side_effect = [mock_from_s3, mock_to_s3]

        from_bucket = get_bucket_name_for_aws_env(promotion.src)
        to_bucket = get_bucket_name_for_aws_env(promotion.dst)

        to_version = from_version.increment()

        # Mock the necessary S3 operations
        mock_from_s3.head_object.return_value = {"ContentLength": len(content)}
        mock_to_s3.head_object.return_value = {"ContentLength": len(content)}

        with (
            patch("scripts.promote.download") as mock_download,
            patch("scripts.promote.upload") as mock_upload,
        ):
            # Call the function
            result_bucket, result_key = copy_across_aws_envs(
                promotion,
                concept,
                classifier,
                from_version,
                to_version,
                False,
            )

            from_object_key = get_object_key(concept, classifier, from_version)
            to_object_key = get_object_key(concept, classifier, to_version)

            # Verify the results
            assert result_bucket == to_bucket
            assert result_key == to_object_key

            # Verify that get_s3_client was called correctly
            mock_get_s3_client.assert_has_calls(
                [call(None, "eu-west-1"), call(None, "eu-west-1")]
            )

            # Verify that download and upload were called with correct arguments
            mock_download.assert_called_once_with(
                mock_from_s3, from_bucket, from_object_key, ANY
            )
            mock_upload.assert_called_once_with(
                mock_to_s3, ANY, to_bucket, to_object_key
            )


@pytest.mark.parametrize(
    "from_bucket, object_key, content_length",
    [
        ("test-bucket", "test/object.pkl", 996),
        ("another-bucket", "models/model.pkl", 4992),
    ],
)
def test_download(from_bucket, object_key, content_length, tmp_path):
    # Create a mock S3 client
    mock_s3_client = Mock()

    # Create test content
    test_content = b"test content" * (content_length // 12)

    # Mock the head_object method to return the content length
    mock_s3_client.head_object.return_value = {"ContentLength": content_length}

    # Mock the download_file method to write the test content to a file
    def mock_download_file(Bucket, Key, Filename, Callback):
        with open(Filename, "wb") as f:
            f.write(test_content)
        Callback(content_length)

    mock_s3_client.download_file.side_effect = mock_download_file

    # Use a temporary file for the download
    temp_file = tmp_path / "downloaded_file"

    # Call the download function
    download(mock_s3_client, from_bucket, object_key, temp_file)

    # Verify that the head_object method was called correctly
    mock_s3_client.head_object.assert_called_once_with(
        Bucket=from_bucket,
        Key=object_key,
    )

    # Verify that the download_file method was called correctly
    mock_s3_client.download_file.assert_called_once_with(
        from_bucket,
        object_key,
        str(temp_file),
        Callback=ANY,
    )

    # Verify the file was downloaded correctly
    assert temp_file.read_bytes() == test_content


@pytest.mark.parametrize(
    "to_bucket, object_key, content_length",
    [
        ("test-bucket", "test/object.pkl", 996),
        ("another-bucket", "models/model.pkl", 4992),
    ],
)
def test_upload(to_bucket, object_key, content_length, tmp_path):
    # Create a mock S3 client
    mock_s3_client = Mock()

    # Create a test file to upload
    test_content = b"test content" * (content_length // 12)
    temp_file = tmp_path / "upload_file"
    temp_file.write_bytes(test_content)

    # Call the upload function
    upload(mock_s3_client, temp_file, to_bucket, object_key)

    # Verify that the upload_file method was called correctly
    mock_s3_client.upload_file.assert_called_once_with(
        str(temp_file),
        to_bucket,
        object_key,
        Callback=ANY,
    )

    # Verify that the Callback function was called with the correct total size
    callback = mock_s3_client.upload_file.call_args[1]["Callback"]
    callback(content_length)
