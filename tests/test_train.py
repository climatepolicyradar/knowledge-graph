import os
from unittest.mock import Mock, patch

import boto3
import pytest
from moto import mock_aws

import wandb
from scripts.train import (
    AwsEnv,
    Namespace,
    StorageLink,
    StorageUpload,
    get_next_version,
    get_s3_client,
    link_model_artifact,
    main,
    upload_model_artifact,
)
from src.identifiers import WikibaseID


@pytest.mark.parametrize(
    "aws_env, expected_profile",
    [
        (AwsEnv.labs, "labs"),
        (AwsEnv.sandbox, "sandbox"),
        (AwsEnv.staging, "staging"),
        (AwsEnv.production, "production"),
    ],
)
def test_get_s3_client(aws_env, expected_profile):
    with patch("boto3.Session") as mock_session:
        mock_client = Mock()
        mock_session.return_value.client.return_value = mock_client

        region_name = "eu-west-1"
        client = get_s3_client(aws_env, region_name)

        mock_session.assert_called_once_with(profile_name=expected_profile)
        mock_session.return_value.client.assert_called_once_with(
            "s3", region_name=region_name
        )

        assert client == mock_client


@pytest.mark.parametrize(
    "aws_env, expected_bucket",
    [
        (AwsEnv.labs, "cpr-labs-models"),
        (AwsEnv.sandbox, "cpr-sandbox-models"),
        (AwsEnv.staging, "cpr-staging-models"),
        (AwsEnv.production, "cpr-production-models"),
    ],
)
@mock_aws
def test_upload_model_artifact(
    aws_credentials,
    aws_env,
    expected_bucket,
    tmp_path,
):
    # Create a mock classifier
    mock_classifier = Mock()
    mock_classifier.name = "test_classifier"

    # Create a temporary file to upload
    test_file_path = os.path.join(tmp_path, "test_model.pickle")
    with open(test_file_path, "w") as f:
        f.write("test model content")

    # Set up the S3 client
    s3_client = boto3.client("s3", region_name="us-east-1")
    s3_client.create_bucket(Bucket=expected_bucket)

    storage_upload = StorageUpload(
        s3_client=s3_client,
        next_version="v3",
        aws_env=aws_env,
    )

    # Call the function
    bucket, key = upload_model_artifact(
        classifier=mock_classifier,
        classifier_path=test_file_path,
        storage_upload=storage_upload,
        namespace=Namespace(project=WikibaseID("Q123"), entity="test_entity"),
    )

    # Assert the correct bucket was used
    assert bucket == expected_bucket

    # Assert the key structure is correct
    assert key == "Q123/test_classifier/v3/model.pickle"

    # Check if the file was uploaded to S3
    response = s3_client.get_object(Bucket=bucket, Key=key)
    content = response["Body"].read().decode("utf-8")
    assert content == "test model content"


def test_main_track_false_upload_true():
    with pytest.raises(
        ValueError,
        match="you can only upload a model artifact, if you're also tracking the run",
    ):
        main(
            wikibase_id="Q123",
            track=False,
            upload=True,
            aws_env=AwsEnv.labs,
        )


@mock_aws
def test_link_model_artifact(aws_credentials):
    # Given there's a model that's been uploaded to S3
    mock_run = Mock()
    mock_classifier = Mock()
    mock_classifier.name = "test_classifier"
    region_name = "eu-west-1"
    bucket = "cpr-labs-models"
    key = "Q123/test_classifier/v3/model.pickle"
    aws_env = AwsEnv.labs

    session = boto3.Session()
    s3_client = session.client("s3", region_name=region_name)

    s3_client.create_bucket(
        Bucket=bucket,
        CreateBucketConfiguration={"LocationConstraint": region_name},
    )

    # When it's linked from S3 to a W&B artifact
    with patch("wandb.Artifact") as mock_artifact_class:
        mock_artifact_instance = Mock()
        mock_artifact_class.return_value = mock_artifact_instance

        storage_link = StorageLink(
            bucket=bucket,
            key=key,
            aws_env=aws_env,
        )

        # When it's linked from S3 to a W&B artifact
        artifact = link_model_artifact(
            mock_run,
            mock_classifier,
            storage_link,
        )

        # Then W&B internally has gotten the checksums from S3. This is
        # done internally, which is why we use `mock_aws`, and need to
        # setup the bucket).
        mock_run.log_artifact.assert_called_once()

        # Then the artifact was logged in W&b.
        mock_artifact_class.assert_called_once_with(
            name=mock_classifier.name,
            type="model",
            metadata={"aws_env": aws_env.value},
        )

        # Then the artifact returned is the mocked instance
        assert artifact == mock_artifact_instance


@patch("wandb.Api")
def test_get_next_version_with_existing(mock_api):
    mock_artifact = Mock()
    mock_artifact._version = "v2"
    mock_api.return_value.artifact.return_value = mock_artifact

    namespace = Namespace(project=WikibaseID("Q123"), entity="test_entity")
    mock_classifier = Mock()
    mock_classifier.name = "test_classifier"
    wikibase_id = "Q123"

    next_version = get_next_version(namespace, wikibase_id, mock_classifier)

    assert next_version == "v3"


@patch("wandb.Api")
def test_get_next_version_with_default(mock_api):
    namespace = Namespace(project=WikibaseID("Q123"), entity="test_entity")
    mock_classifier = Mock()
    mock_classifier.name = "test_classifier"
    wikibase_id = "Q123"

    mock_api.side_effect = wandb.errors.CommError(
        "artifact 'test_classifier:latest' not found in 'test_entity/Q123'"
    )

    next_version = get_next_version(namespace, wikibase_id, mock_classifier)

    assert next_version == "v0"
