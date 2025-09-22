import os
from pathlib import Path
from unittest.mock import ANY, Mock, patch

import pytest
from wandb.errors.errors import CommError

from knowledge_graph.classifier.targets import TargetClassifier
from knowledge_graph.cloud import AwsEnv
from knowledge_graph.identifiers import WikibaseID
from scripts.train import (
    Namespace,
    StorageLink,
    StorageUpload,
    create_and_link_model_artifact,
    get_next_version,
    run_training,
    upload_model_artifact,
)


@pytest.mark.parametrize(
    "aws_env, expected_bucket",
    [
        (AwsEnv.labs, "cpr-labs-models"),
        (AwsEnv.sandbox, "cpr-sandbox-models"),
        (AwsEnv.staging, "cpr-staging-models"),
        (AwsEnv.production, "cpr-prod-models"),
    ],
)
def test_upload_model_artifact(aws_env, expected_bucket, tmp_path):
    # Create a mock classifier
    mock_classifier = Mock()
    mock_classifier.name = "test_classifier"

    # Create a temporary file to upload
    test_file_path = Path(os.path.join(tmp_path, "test_model.pickle"))
    with open(test_file_path, "w") as f:
        f.write("test model content")

    # Create a mock S3 client
    mock_s3_client = Mock()

    target_path = "Q123/v4prnc54"
    storage_upload = StorageUpload(
        target_path=target_path,
        next_version="v3",
        aws_env=aws_env,
    )

    # Call the function
    bucket, key = upload_model_artifact(
        classifier=mock_classifier,
        classifier_path=test_file_path,
        storage_upload=storage_upload,
        s3_client=mock_s3_client,
    )

    # Assert the correct bucket was used
    assert bucket == expected_bucket

    # Assert the key structure is correct
    assert key == "Q123/v4prnc54/v3/model.pickle"

    # Verify that the upload_file method was called with correct arguments
    mock_s3_client.upload_file.assert_called_once_with(
        test_file_path,
        expected_bucket,
        key,
        ExtraArgs={"ContentType": "application/octet-stream"},
        Callback=ANY,
    )


@pytest.mark.parametrize(
    ("classifier_class_to_spec", "extra_metadata"),
    [
        (None, {}),
        (TargetClassifier, {"compute_environment": {"gpu": True}}),
    ],
)
@pytest.mark.asyncio
async def test_run_training(
    classifier_class_to_spec, extra_metadata, MockedWikibaseSession, mock_s3_client
):
    mock_s3_client.create_bucket(
        Bucket="cpr-labs-models",
        CreateBucketConfiguration={"LocationConstraint": "eu-west-1"},
    )

    # Setup test data
    mock_classifier = Mock(spec=classifier_class_to_spec)
    mock_classifier.fit.return_value = None
    mock_classifier.save.return_value = None
    mock_classifier.id = "aaaa2222"
    mock_classifier.version = None

    mock_concept = Mock()
    mock_concept.id = "5d4xcy5g"
    mock_concept.wikibase_revision = 12300
    mock_classifier.concept = mock_concept

    mock_path = Path("tests/fixtures/data/processed/classifiers")
    mock_artifact = Mock(_version="v0")
    mock_artifact_instance = Mock()

    with (
        patch(
            "knowledge_graph.classifier.ClassifierFactory.create",
            return_value=mock_classifier,
        ),
        patch("knowledge_graph.config.classifier_dir", mock_path),
        patch("scripts.train.validate_params"),
        patch("wandb.init"),
        patch(
            "wandb.Api", return_value=Mock(artifact=Mock(return_value=mock_artifact))
        ),
        patch(
            "wandb.Artifact", return_value=mock_artifact_instance
        ) as mock_artifact_class,
        patch("scripts.get_concept.ArgillaSession") as mock_argilla_session,
    ):
        mock_argilla_instance = Mock()
        mock_argilla_instance.pull_labelled_passages.return_value = []
        mock_argilla_session.return_value = mock_argilla_instance

        result = await run_training(
            wikibase_id=WikibaseID("Q787"),
            track=True,
            upload=True,
            aws_env=AwsEnv.labs,
            s3_client=mock_s3_client,
        )

        # Verify artifact was created with correct metadata
        mock_artifact_class.assert_called_once_with(
            name=mock_classifier.id,
            type="model",
            metadata={
                "aws_env": "labs",
                "classifier_name": mock_classifier.name,
                "concept_id": mock_classifier.concept.id,
                "concept_wikibase_revision": mock_classifier.concept.wikibase_revision,
                **extra_metadata,
            },
        )

    assert result == mock_classifier


@pytest.mark.asyncio
async def test_run_training__valueerror():
    with pytest.raises(
        ValueError,
        match="you can only upload a model artifact, if you're also tracking the run",
    ):
        await run_training(
            wikibase_id=WikibaseID("Q123"),
            track=False,
            upload=True,
            aws_env=AwsEnv.labs,
        )


def test_create_and_link_model_artifact():
    # Given there's a model that's been uploaded to S3
    mock_run = Mock()
    mock_classifier = Mock()
    mock_classifier.name = "test_classifier"
    mock_classifier.concept = Mock()
    mock_classifier.concept.id = "5d4xcy5g"
    mock_classifier.concept.wikibase_revision = 12300
    bucket = "cpr-labs-models"
    key = "Q123/v4prnc54/v3/model.pickle"
    aws_env = AwsEnv.labs

    storage_link = StorageLink(
        bucket=bucket,
        key=key,
        aws_env=aws_env,
    )

    # When it's linked from S3 to a W&B artifact
    with patch("wandb.Artifact") as mock_artifact_class:
        mock_artifact_instance = Mock()
        mock_artifact_class.return_value = mock_artifact_instance

        create_and_link_model_artifact(
            mock_run,
            mock_classifier,
            storage_link,
        )

        # Then the artifact was created with correct parameters
        mock_artifact_class.assert_called_once_with(
            name=mock_classifier.id,
            type="model",
            metadata={
                "aws_env": aws_env.value,
                "classifier_name": "test_classifier",
                "concept_id": "5d4xcy5g",
                "concept_wikibase_revision": 12300,
            },
        )

        # Then the S3 reference was added to the artifact
        mock_artifact_instance.add_reference.assert_called_once_with(
            uri=f"s3://{bucket}/{key}", checksum=True
        )

        # Then the artifact was logged in W&B
        mock_run.log_artifact.assert_called_once_with(
            mock_artifact_instance,
            aliases=[aws_env.value],
        )

        # Then the artifact was waited for
        mock_run.log_artifact.return_value.wait.assert_called_once()


@patch("wandb.Api")
def test_get_next_version_with_existing(mock_api):
    mock_artifact = Mock()
    mock_artifact._version = "v2"
    mock_api.return_value.artifact.return_value = mock_artifact

    namespace = Namespace(project=WikibaseID("Q123"), entity="test_entity")
    mock_classifier = Mock()
    mock_classifier.concept.wikibase_id = "Q123"

    wandb_target_entity = f"{namespace.project}/{mock_classifier.id}"
    next_version = get_next_version(namespace, wandb_target_entity, mock_classifier)

    assert next_version == "v3"


@patch("wandb.Api")
def test_get_next_version_with_default(mock_api):
    namespace = Namespace(project=WikibaseID("Q123"), entity="test_entity")
    mock_classifier = Mock()
    mock_classifier.concept.wikibase_id = "Q123"

    mock_api.side_effect = CommError(
        "artifact membership 'test_classifier:latest' not found in 'test_entity/Q123'"
    )
    wandb_target_entity = f"{namespace.project}/{mock_classifier.id}"
    next_version = get_next_version(namespace, wandb_target_entity, mock_classifier)

    assert next_version == "v0"
