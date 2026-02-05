import os
from pathlib import Path
from unittest.mock import ANY, MagicMock, Mock, patch

import pytest
from wandb.errors.errors import CommError

from knowledge_graph.classifier.targets import TargetClassifier
from knowledge_graph.cloud import AwsEnv
from knowledge_graph.config import wandb_model_artifact_filename
from knowledge_graph.identifiers import ClassifierID, WikibaseID
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
    test_file_path = Path(os.path.join(tmp_path, wandb_model_artifact_filename))
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
    assert key == f"Q123/v4prnc54/v3/{wandb_model_artifact_filename}"

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
    # Set model_name to a non-OpenRouter model to avoid pricing lookups
    mock_classifier.model_name = "gpt-4"

    mock_concept = Mock()
    mock_concept.id = "5d4xcy5g"
    mock_concept.wikibase_revision = 12300
    mock_concept.labelled_passages = []
    mock_classifier.concept = mock_concept

    mock_path = Path("tests/fixtures/data/processed/classifiers")
    mock_artifact = Mock(_version="v0")
    mock_artifact_instance = Mock()

    mock_file = MagicMock()
    mock_new_file_context_manager = MagicMock()
    mock_new_file_context_manager.__enter__ = Mock(return_value=mock_file)
    mock_new_file_context_manager.__exit__ = Mock(return_value=None)
    mock_artifact_instance.new_file.return_value = mock_new_file_context_manager

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
        patch("scripts.train.evaluate_classifier") as mock_evaluate,
        patch("scripts.train.classifier_exists_in_wandb", return_value=False),
    ):
        mock_argilla_instance = Mock()
        mock_argilla_instance.get_labelled_passages.return_value = []
        mock_argilla_session.return_value = mock_argilla_instance

        mock_metrics_df = Mock()
        mock_labelled_passages = []
        mock_confusion_matrix = Mock()
        mock_evaluate.return_value = (
            mock_metrics_df,
            mock_labelled_passages,
            mock_confusion_matrix,
        )

        result = await run_training(
            wikibase_id=WikibaseID("Q787"),
            track_and_upload=True,
            aws_env=AwsEnv.labs,
            s3_client=mock_s3_client,
        )

        # Two artifacts are created: a model and a set of labelled passages
        assert mock_artifact_class.call_count == 2

        model_artifact_call = mock_artifact_class.call_args_list[0]
        assert model_artifact_call[1]["name"] == mock_classifier.id
        assert model_artifact_call[1]["type"] == "model"
        assert model_artifact_call[1]["metadata"] == {
            "aws_env": "labs",
            "classifier_name": mock_classifier.name,
            "concept_id": mock_classifier.concept.id,
            "concept_wikibase_revision": mock_classifier.concept.wikibase_revision,
            **extra_metadata,
        }

        labelled_passages_call = mock_artifact_class.call_args_list[1]
        assert (
            labelled_passages_call[1]["name"]
            == f"{mock_classifier.id}-labelled-passages"
        )
        assert labelled_passages_call[1]["type"] == "labelled_passages"

    assert result == mock_classifier


def test_create_and_link_model_artifact():
    # Given there's a model that's been uploaded to S3
    mock_run = Mock()
    mock_classifier = Mock()
    mock_classifier.name = "test_classifier"
    mock_classifier.concept = Mock()
    mock_classifier.concept.id = "5d4xcy5g"
    mock_classifier.concept.wikibase_revision = 12300
    bucket = "cpr-labs-models"
    key = f"Q123/v4prnc54/v3/{wandb_model_artifact_filename}"
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
            aliases=[],
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


@pytest.mark.asyncio
async def test_run_training_uploads_labelled_passages_when_evaluate_is_true(
    MockedWikibaseSession, mock_s3_client
):
    """Test that labelled passages artifact is created and uploaded when evaluate=True and track_and_upload=True."""
    mock_s3_client.create_bucket(
        Bucket="cpr-labs-models",
        CreateBucketConfiguration={"LocationConstraint": "eu-west-1"},
    )

    # Setup test data
    mock_classifier = Mock()
    mock_classifier.fit.return_value = None
    mock_classifier.save.return_value = None
    mock_classifier.id = "aaaa2222"
    mock_classifier.version = None
    # Set model_name to a non-OpenRouter model to avoid pricing lookups
    mock_classifier.model_name = "gpt-4"

    mock_concept = Mock()
    mock_concept.id = "5d4xcy5g"
    mock_concept.wikibase_revision = 12300
    mock_concept.labelled_passages = []
    mock_classifier.concept = mock_concept

    mock_path = Path("tests/fixtures/data/processed/classifiers")
    mock_artifact = Mock(_version="v0")
    mock_run = MagicMock()
    mock_run.__enter__ = Mock(return_value=mock_run)
    mock_run.__exit__ = Mock(return_value=None)

    # Mock labelled passages that would be returned by evaluate_classifier
    mock_passage_1 = Mock()
    mock_passage_1.model_dump_json.return_value = '{"passage": "test passage 1"}'
    mock_passage_2 = Mock()
    mock_passage_2.model_dump_json.return_value = '{"passage": "test passage 2"}'
    mock_labelled_passages = [mock_passage_1, mock_passage_2]
    mock_metrics_df = Mock()

    with (
        patch(
            "knowledge_graph.classifier.ClassifierFactory.create",
            return_value=mock_classifier,
        ),
        patch("knowledge_graph.config.classifier_dir", mock_path),
        patch("scripts.train.validate_params"),
        patch("wandb.init", return_value=mock_run),
        patch(
            "wandb.Api", return_value=Mock(artifact=Mock(return_value=mock_artifact))
        ),
        patch("wandb.Artifact") as mock_artifact_class,
        patch("scripts.get_concept.ArgillaSession") as mock_argilla_session,
        patch("scripts.train.evaluate_classifier") as mock_evaluate,
        patch("scripts.train.classifier_exists_in_wandb", return_value=False),
    ):
        mock_argilla_instance = Mock()
        mock_argilla_instance.get_labelled_passages.return_value = []
        mock_argilla_session.return_value = mock_argilla_instance
        mock_confusion_matrix = Mock()

        # Configure evaluate_classifier to return mock data
        mock_evaluate.return_value = (
            mock_metrics_df,
            mock_labelled_passages,
            mock_confusion_matrix,
        )

        # Create mock artifact instances with proper context manager support
        mock_labelled_passages_artifact = Mock()
        mock_artifact_file = MagicMock()
        mock_context_manager = MagicMock()
        mock_context_manager.__enter__ = Mock(return_value=mock_artifact_file)
        mock_context_manager.__exit__ = Mock(return_value=None)
        mock_labelled_passages_artifact.new_file.return_value = mock_context_manager

        mock_artifact_class.return_value = mock_labelled_passages_artifact

        result = await run_training(
            wikibase_id=WikibaseID("Q787"),
            track_and_upload=True,
            aws_env=AwsEnv.labs,
            s3_client=mock_s3_client,
            evaluate=True,
        )

        assert mock_artifact_class.call_count == 2
        labelled_passages_call = mock_artifact_class.call_args_list[1]
        assert (
            labelled_passages_call[1]["name"]
            == f"{mock_classifier.id}-labelled-passages"
        )
        assert labelled_passages_call[1]["type"] == "labelled_passages"

        log_artifact_calls = mock_run.log_artifact.call_args_list
        assert len(log_artifact_calls) == 2

    assert result == mock_classifier


@pytest.mark.asyncio
async def test_whether_run_training_skips_classifier_when_it_already_exists_in_wandb_and_force_false_is_set(
    MockedWikibaseSession, mock_s3_client
):
    """Test that training is skipped when classifier already exists in W&B."""
    mock_s3_client.create_bucket(
        Bucket="cpr-labs-models",
        CreateBucketConfiguration={"LocationConstraint": "eu-west-1"},
    )

    # Setup test data
    mock_classifier = Mock()
    mock_classifier.fit = Mock()
    mock_classifier.save = Mock()
    mock_classifier.id = ClassifierID.generate("existing_classifier")
    mock_classifier.version = None

    mock_concept = Mock()
    mock_concept.id = "5d4xcy5g"
    mock_concept.wikibase_revision = 12300
    mock_concept.labelled_passages = []
    mock_classifier.concept = mock_concept

    mock_path = Path("tests/fixtures/data/processed/classifiers")

    with (
        patch(
            "knowledge_graph.classifier.ClassifierFactory.create",
            return_value=mock_classifier,
        ),
        patch("knowledge_graph.config.classifier_dir", mock_path),
        patch("scripts.train.validate_params"),
        patch("wandb.init"),
        patch("scripts.get_concept.ArgillaSession") as mock_argilla_session,
        patch("scripts.train.classifier_exists_in_wandb", return_value=True),
        patch("wandb.Artifact") as mock_artifact_class,
    ):
        mock_argilla_instance = Mock()
        mock_argilla_instance.get_labelled_passages.return_value = []
        mock_argilla_session.return_value = mock_argilla_instance

        result = await run_training(
            wikibase_id=WikibaseID("Q787"),
            track_and_upload=True,
            aws_env=AwsEnv.labs,
            s3_client=mock_s3_client,
            force=False,
        )

        # Should return classifier without training
        assert result == mock_classifier

        # Verify training was skipped
        mock_classifier.fit.assert_not_called()
        mock_classifier.save.assert_not_called()

        # Verify no artifacts were created
        mock_artifact_class.assert_not_called()


@pytest.mark.asyncio
async def test_run_training_continues_when_openrouter_api_fails(
    MockedWikibaseSession, mock_s3_client
):
    """Test that training continues successfully even when OpenRouter API fails."""
    mock_s3_client.create_bucket(
        Bucket="cpr-labs-models",
        CreateBucketConfiguration={"LocationConstraint": "eu-west-1"},
    )

    mock_classifier = Mock()
    mock_classifier.fit.return_value = None
    mock_classifier.save.return_value = None
    mock_classifier.id = "aaaa2222"
    mock_classifier.version = None
    mock_classifier.model_name = "openrouter:openai/gpt-4"  # OpenRouter model

    mock_concept = Mock()
    mock_concept.id = "5d4xcy5g"
    mock_concept.wikibase_revision = 12300
    mock_concept.labelled_passages = []
    mock_classifier.concept = mock_concept

    mock_path = Path("tests/fixtures/data/processed/classifiers")
    mock_artifact = Mock(_version="v0")
    mock_artifact_instance = Mock()

    mock_file = MagicMock()
    mock_new_file_context_manager = MagicMock()
    mock_new_file_context_manager.__enter__ = Mock(return_value=mock_file)
    mock_new_file_context_manager.__exit__ = Mock(return_value=None)
    mock_artifact_instance.new_file.return_value = mock_new_file_context_manager

    mock_run = MagicMock()

    with (
        patch(
            "knowledge_graph.classifier.ClassifierFactory.create",
            return_value=mock_classifier,
        ),
        patch("knowledge_graph.config.classifier_dir", mock_path),
        patch("scripts.train.validate_params"),
        patch("wandb.init", return_value=mock_run),
        patch(
            "wandb.Api", return_value=Mock(artifact=Mock(return_value=mock_artifact))
        ),
        patch("wandb.Artifact", return_value=mock_artifact_instance) as _,
        patch("scripts.get_concept.ArgillaSession") as mock_argilla_session,
        patch("scripts.train.evaluate_classifier") as mock_evaluate,
        patch("scripts.train.classifier_exists_in_wandb", return_value=False),
        patch(
            "scripts.train.get_openrouter_pricing", return_value=None
        ) as mock_pricing,
    ):
        mock_argilla_instance = Mock()
        mock_argilla_instance.get_labelled_passages.return_value = []
        mock_argilla_session.return_value = mock_argilla_instance

        mock_metrics_df = Mock()
        mock_labelled_passages = []
        mock_confusion_matrix = Mock()
        mock_evaluate.return_value = (
            mock_metrics_df,
            mock_labelled_passages,
            mock_confusion_matrix,
        )

        result = await run_training(
            wikibase_id=WikibaseID("Q787"),
            track_and_upload=True,
            aws_env=AwsEnv.labs,
            s3_client=mock_s3_client,
        )

        # Verify the pricing function was called
        mock_pricing.assert_called_once_with("openrouter:openai/gpt-4")

        # Verify training still completed successfully
        assert result == mock_classifier
        mock_classifier.fit.assert_called_once()
        mock_classifier.save.assert_called_once()


@pytest.mark.asyncio
async def test_run_training_includes_pricing_when_openrouter_api_succeeds(
    MockedWikibaseSession, mock_s3_client
):
    """Test that OpenRouter pricing is included in W&B config when API succeeds."""
    mock_s3_client.create_bucket(
        Bucket="cpr-labs-models",
        CreateBucketConfiguration={"LocationConstraint": "eu-west-1"},
    )

    # Setup test data with an OpenRouter model
    mock_classifier = Mock()
    mock_classifier.fit.return_value = None
    mock_classifier.save.return_value = None
    mock_classifier.id = "aaaa2222"
    mock_classifier.version = None
    mock_classifier.model_name = "openrouter:anthropic/claude-3-5-sonnet"

    mock_concept = Mock()
    mock_concept.id = "5d4xcy5g"
    mock_concept.wikibase_revision = 12300
    mock_concept.labelled_passages = []
    mock_classifier.concept = mock_concept

    mock_path = Path("tests/fixtures/data/processed/classifiers")
    mock_artifact = Mock(_version="v0")
    mock_artifact_instance = Mock()

    mock_file = MagicMock()
    mock_new_file_context_manager = MagicMock()
    mock_new_file_context_manager.__enter__ = Mock(return_value=mock_file)
    mock_new_file_context_manager.__exit__ = Mock(return_value=None)
    mock_artifact_instance.new_file.return_value = mock_new_file_context_manager

    mock_run = MagicMock()

    from knowledge_graph.openrouter_pricing import ModelPricing

    mock_pricing_data = ModelPricing(prompt_price=0.00003, completion_price=0.00015)

    with (
        patch(
            "knowledge_graph.classifier.ClassifierFactory.create",
            return_value=mock_classifier,
        ),
        patch("knowledge_graph.config.classifier_dir", mock_path),
        patch("scripts.train.validate_params"),
        patch("wandb.init", return_value=mock_run) as mock_wandb_init,
        patch(
            "wandb.Api", return_value=Mock(artifact=Mock(return_value=mock_artifact))
        ),
        patch("wandb.Artifact", return_value=mock_artifact_instance) as _,
        patch("scripts.get_concept.ArgillaSession") as mock_argilla_session,
        patch("scripts.train.evaluate_classifier") as mock_evaluate,
        patch("scripts.train.classifier_exists_in_wandb", return_value=False),
        patch(
            "scripts.train.get_openrouter_pricing", return_value=mock_pricing_data
        ) as mock_pricing_func,
    ):
        mock_argilla_instance = Mock()
        mock_argilla_instance.get_labelled_passages.return_value = []
        mock_argilla_session.return_value = mock_argilla_instance

        mock_metrics_df = Mock()
        mock_labelled_passages = []
        mock_confusion_matrix = Mock()
        mock_evaluate.return_value = (
            mock_metrics_df,
            mock_labelled_passages,
            mock_confusion_matrix,
        )

        result = await run_training(
            wikibase_id=WikibaseID("Q787"),
            track_and_upload=True,
            aws_env=AwsEnv.labs,
            s3_client=mock_s3_client,
        )

        # Verify the pricing function was called
        mock_pricing_func.assert_called_once_with(
            "openrouter:anthropic/claude-3-5-sonnet"
        )

        # Verify training completed successfully
        assert result == mock_classifier

        # Verify wandb.init was called with config including pricing
        mock_wandb_init.assert_called_once()
        call_kwargs = mock_wandb_init.call_args[1]
        wandb_config = call_kwargs["config"]

        # Check that pricing fields were added to the config
        assert "prompt_price_usd" in wandb_config
        assert wandb_config["prompt_price_usd"] == 0.00003
        assert "completion_price_usd" in wandb_config
        assert wandb_config["completion_price_usd"] == 0.00015


@pytest.mark.asyncio
async def test_run_training_skips_pricing_for_non_openrouter_models(
    MockedWikibaseSession, mock_s3_client
):
    """Test that OpenRouter pricing is not fetched for non-OpenRouter models."""
    mock_s3_client.create_bucket(
        Bucket="cpr-labs-models",
        CreateBucketConfiguration={"LocationConstraint": "eu-west-1"},
    )

    mock_classifier = Mock()
    mock_classifier.fit.return_value = None
    mock_classifier.save.return_value = None
    mock_classifier.id = "aaaa2222"
    mock_classifier.version = None
    mock_classifier.model_name = "gpt-4"  # Not an OpenRouter model

    mock_concept = Mock()
    mock_concept.id = "5d4xcy5g"
    mock_concept.wikibase_revision = 12300
    mock_concept.labelled_passages = []
    mock_classifier.concept = mock_concept

    mock_path = Path("tests/fixtures/data/processed/classifiers")
    mock_artifact = Mock(_version="v0")
    mock_artifact_instance = Mock()

    mock_file = MagicMock()
    mock_new_file_context_manager = MagicMock()
    mock_new_file_context_manager.__enter__ = Mock(return_value=mock_file)
    mock_new_file_context_manager.__exit__ = Mock(return_value=None)
    mock_artifact_instance.new_file.return_value = mock_new_file_context_manager

    mock_run = MagicMock()

    with (
        patch(
            "knowledge_graph.classifier.ClassifierFactory.create",
            return_value=mock_classifier,
        ),
        patch("knowledge_graph.config.classifier_dir", mock_path),
        patch("scripts.train.validate_params"),
        patch("wandb.init", return_value=mock_run) as mock_wandb_init,
        patch(
            "wandb.Api", return_value=Mock(artifact=Mock(return_value=mock_artifact))
        ),
        patch("wandb.Artifact", return_value=mock_artifact_instance) as _,
        patch("scripts.get_concept.ArgillaSession") as mock_argilla_session,
        patch("scripts.train.evaluate_classifier") as mock_evaluate,
        patch("scripts.train.classifier_exists_in_wandb", return_value=False),
        patch("scripts.train.get_openrouter_pricing") as mock_pricing_func,
    ):
        mock_argilla_instance = Mock()
        mock_argilla_instance.get_labelled_passages.return_value = []
        mock_argilla_session.return_value = mock_argilla_instance

        mock_metrics_df = Mock()
        mock_labelled_passages = []
        mock_confusion_matrix = Mock()
        mock_evaluate.return_value = (
            mock_metrics_df,
            mock_labelled_passages,
            mock_confusion_matrix,
        )

        result = await run_training(
            wikibase_id=WikibaseID("Q787"),
            track_and_upload=True,
            aws_env=AwsEnv.labs,
            s3_client=mock_s3_client,
        )

        mock_pricing_func.assert_not_called()

        # Verify training completed successfully
        assert result == mock_classifier

        # Verify wandb.init was called without pricing fields
        mock_wandb_init.assert_called_once()
        call_kwargs = mock_wandb_init.call_args[1]
        wandb_config = call_kwargs["config"]

        assert "prompt_price_usd" not in wandb_config
        assert "completion_price_usd" not in wandb_config
