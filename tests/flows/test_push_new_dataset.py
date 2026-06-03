from unittest.mock import AsyncMock, patch

import pytest

from flows.push_new_dataset import push_new_dataset
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage


@pytest.fixture
def labelled_passages():
    return [LabelledPassage(text=f"passage {i}", spans=[]) for i in range(10)]


@pytest.fixture
def patched_push_dependencies(labelled_passages, test_config):
    with (
        patch("flows.push_new_dataset.load_labelled_passages_from_wandb") as mock_load,
        patch("flows.push_new_dataset.wandb.login") as mock_wandb_login,
        patch("flows.push_new_dataset.Config") as mock_config_cls,
        patch("flows.push_new_dataset.push_passages_to_argilla") as mock_push,
    ):
        mock_load.return_value = labelled_passages
        mock_config_cls.create = AsyncMock(return_value=test_config)

        yield {
            "load": mock_load,
            "wandb_login": mock_wandb_login,
            "push": mock_push,
        }


@pytest.mark.asyncio
async def test_push_new_dataset_loads_passages_from_wandb_artifact(
    patched_push_dependencies, test_config
):
    """The flow should load passages from the given W&B artifact path."""
    await push_new_dataset(
        wikibase_id=WikibaseID("Q787"),
        wandb_artifact_path="climatepolicyradar/Q787/labelled-passages:v0",
        config=test_config,
    )

    patched_push_dependencies["load"].assert_called_once_with(
        wandb_path="climatepolicyradar/Q787/labelled-passages:v0"
    )


@pytest.mark.asyncio
async def test_push_new_dataset_logs_into_wandb_when_api_key_present(
    patched_push_dependencies, test_config
):
    """The flow should call wandb.login when a W&B API key is in Config."""
    await push_new_dataset(
        wikibase_id=WikibaseID("Q787"),
        wandb_artifact_path="climatepolicyradar/Q787/labelled-passages:v0",
        config=test_config,
    )

    patched_push_dependencies["wandb_login"].assert_called_once_with(
        key=test_config.wandb_api_key.get_secret_value()
    )


@pytest.mark.asyncio
async def test_push_new_dataset_skips_wandb_login_without_api_key(
    patched_push_dependencies, test_config
):
    """The flow should not call wandb.login when no W&B API key is configured."""
    config_no_key = test_config.model_copy(update={"wandb_api_key": None})

    await push_new_dataset(
        wikibase_id=WikibaseID("Q787"),
        wandb_artifact_path="climatepolicyradar/Q787/labelled-passages:v0",
        config=config_no_key,
    )

    patched_push_dependencies["wandb_login"].assert_not_called()


@pytest.mark.asyncio
async def test_push_new_dataset_calls_push_with_correct_args(
    patched_push_dependencies, labelled_passages, test_config
):
    """The flow should call push_passages_to_argilla with passages and credentials from Config."""
    await push_new_dataset(
        wikibase_id=WikibaseID("Q787"),
        wandb_artifact_path="climatepolicyradar/Q787/labelled-passages:v0",
        workspace_name="my-workspace",
        limit=5,
        config=test_config,
    )

    patched_push_dependencies["push"].assert_called_once_with(
        labelled_passages=labelled_passages,
        wikibase_id=WikibaseID("Q787"),
        workspace_name="my-workspace",
        limit=5,
        argilla_api_url=test_config.argilla_api_url,
        argilla_api_key=test_config.argilla_api_key.get_secret_value(),
        wikibase_username=test_config.wikibase_username,
        wikibase_password=test_config.wikibase_password.get_secret_value(),
        wikibase_url=test_config.wikibase_url,
    )
