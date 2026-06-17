from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from flows.modify_threshold import modify_threshold
from knowledge_graph.cloud import AwsEnv


@pytest.fixture
def mock_classifier():
    return MagicMock()


@pytest.fixture
def patched_tasks(test_config, mock_classifier):
    original_metadata = {"some": "metadata"}
    artifact_path = "climatepolicyradar/Q913/rsgz5ygh:v1"

    with (
        patch("flows.modify_threshold.Config") as mock_config_cls,
        patch("flows.modify_threshold.login_to_wandb") as mock_login,
        patch(
            "flows.modify_threshold.load_classifier_task", new_callable=AsyncMock
        ) as mock_load,
        patch("flows.modify_threshold.upload_modified_classifier_task") as mock_upload,
    ):
        mock_config_cls.create = AsyncMock(return_value=test_config)
        mock_load.return_value = (mock_classifier, original_metadata)
        mock_upload.return_value = artifact_path
        yield {
            "login": mock_login,
            "load": mock_load,
            "upload": mock_upload,
            "classifier": mock_classifier,
            "metadata": original_metadata,
            "artifact_path": artifact_path,
        }


@pytest.mark.asyncio
async def test_modify_threshold_logs_in_to_wandb_when_api_key_present(
    patched_tasks, test_config
):
    """login_to_wandb should be called with the API key from config."""
    await modify_threshold(
        wandb_path="climatepolicyradar/Q913/rsgz5ygh:v0",
        threshold=0.5,
        config=test_config,
    )

    patched_tasks["login"].assert_called_once_with(
        wandb_api_key=test_config.wandb_api_key
    )


@pytest.mark.asyncio
async def test_modify_threshold_skips_login_when_no_api_key(patched_tasks, test_config):
    """login_to_wandb should be skipped when wandb_api_key is not set."""
    test_config.wandb_api_key = None

    await modify_threshold(
        wandb_path="climatepolicyradar/Q913/rsgz5ygh:v0",
        threshold=0.5,
        config=test_config,
    )

    patched_tasks["login"].assert_not_called()


@pytest.mark.asyncio
async def test_modify_threshold_passes_wandb_path_to_load(patched_tasks, test_config):
    """load_classifier_task should receive the wandb_path."""
    wandb_path = "climatepolicyradar/Q913/rsgz5ygh:v0"

    await modify_threshold(
        wandb_path=wandb_path,
        threshold=0.5,
        config=test_config,
    )

    _, kwargs = patched_tasks["load"].call_args
    assert kwargs["wandb_path"] == wandb_path


@pytest.mark.asyncio
async def test_modify_threshold_passes_classifier_and_metadata_to_upload(
    patched_tasks, test_config
):
    """upload_modified_classifier_task should receive exactly what load_classifier_task returned."""
    await modify_threshold(
        wandb_path="climatepolicyradar/Q913/rsgz5ygh:v0",
        threshold=0.5,
        config=test_config,
    )

    _, kwargs = patched_tasks["upload"].call_args
    assert kwargs["classifier"] is patched_tasks["classifier"]
    assert kwargs["original_metadata"] is patched_tasks["metadata"]


@pytest.mark.asyncio
async def test_modify_threshold_passes_threshold_to_upload(patched_tasks, test_config):
    """Threshold should be forwarded to upload_modified_classifier_task."""
    await modify_threshold(
        wandb_path="climatepolicyradar/Q913/rsgz5ygh:v0",
        threshold=0.42,
        config=test_config,
    )

    _, kwargs = patched_tasks["upload"].call_args
    assert kwargs["threshold"] == 0.42


@pytest.mark.asyncio
async def test_modify_threshold_passes_source_wandb_path_to_upload(
    patched_tasks, test_config
):
    """source_wandb_path passed to upload should match the input wandb_path."""
    wandb_path = "climatepolicyradar/Q913/rsgz5ygh:v0"

    await modify_threshold(
        wandb_path=wandb_path,
        threshold=0.5,
        config=test_config,
    )

    _, kwargs = patched_tasks["upload"].call_args
    assert kwargs["source_wandb_path"] == wandb_path


@pytest.mark.asyncio
async def test_modify_threshold_passes_aws_env_to_upload(patched_tasks, test_config):
    """aws_env should be forwarded to upload_modified_classifier_task."""
    await modify_threshold(
        wandb_path="climatepolicyradar/Q913/rsgz5ygh:v0",
        threshold=0.5,
        aws_env=AwsEnv.sandbox,
        config=test_config,
    )

    _, kwargs = patched_tasks["upload"].call_args
    assert kwargs["aws_env"] == AwsEnv.sandbox


@pytest.mark.asyncio
async def test_modify_threshold_returns_artifact_path(patched_tasks, test_config):
    """The flow should return the artifact path from upload_modified_classifier_task."""
    result = await modify_threshold(
        wandb_path="climatepolicyradar/Q913/rsgz5ygh:v0",
        threshold=0.5,
        config=test_config,
    )

    assert result == patched_tasks["artifact_path"]
