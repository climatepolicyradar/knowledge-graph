import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from flows.push_new_dataset import push_new_dataset
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage


@pytest.fixture
def labelled_passages():
    return [LabelledPassage(text=f"passage {i}", spans=[]) for i in range(10)]


@pytest.fixture
def mock_concept():
    concept = MagicMock()
    concept.wikibase_id = WikibaseID("Q787")
    return concept


@pytest.fixture
def patched_push_dependencies(labelled_passages, mock_concept, test_config):
    with (
        patch("flows.push_new_dataset.load_labelled_passages_from_wandb") as mock_load,
        patch("flows.push_new_dataset.wandb.login") as mock_wandb_login,
        patch("flows.push_new_dataset.Config") as mock_config_cls,
        patch("flows.push_new_dataset.ArgillaSession") as mock_argilla_cls,
        patch("flows.push_new_dataset.WikibaseSession") as mock_wikibase_cls,
        patch(
            "flows.push_new_dataset.create_llm_ensemble_for_prelabelling_validation_dataset"
        ) as mock_create_ensemble,
    ):
        mock_load.return_value = labelled_passages
        mock_config_cls.create = AsyncMock(return_value=test_config)

        mock_argilla = MagicMock()
        mock_argilla_cls.return_value = mock_argilla

        mock_wikibase = MagicMock()
        mock_wikibase_cls.return_value = mock_wikibase
        mock_wikibase.get_concept_async = AsyncMock(return_value=mock_concept)

        yield {
            "load": mock_load,
            "wandb_login": mock_wandb_login,
            "argilla": mock_argilla,
            "argilla_cls": mock_argilla_cls,
            "wikibase_cls": mock_wikibase_cls,
            "create_ensemble": mock_create_ensemble,
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
    """The flow should call wandb.login when the config has a W&B API key."""
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
    """The flow should not call wandb.login when the config has no W&B API key."""
    config_no_key = test_config.model_copy(update={"wandb_api_key": None})

    await push_new_dataset(
        wikibase_id=WikibaseID("Q787"),
        wandb_artifact_path="climatepolicyradar/Q787/labelled-passages:v0",
        config=config_no_key,
    )

    patched_push_dependencies["wandb_login"].assert_not_called()


@pytest.mark.asyncio
async def test_push_new_dataset_applies_limit(
    patched_push_dependencies, labelled_passages, test_config
):
    """Passages should be sliced to the limit before being pushed to Argilla."""
    await push_new_dataset(
        wikibase_id=WikibaseID("Q787"),
        wandb_artifact_path="climatepolicyradar/Q787/labelled-passages:v0",
        limit=3,
        config=test_config,
    )

    call_kwargs = patched_push_dependencies[
        "argilla"
    ].add_labelled_passages.call_args.kwargs
    assert len(call_kwargs["labelled_passages"]) == 3
    assert call_kwargs["labelled_passages"] == labelled_passages[:3]


@pytest.mark.asyncio
async def test_push_new_dataset_passes_argilla_credentials_from_config(
    patched_push_dependencies, test_config
):
    """ArgillaSession should be initialised with credentials from Config."""
    await push_new_dataset(
        wikibase_id=WikibaseID("Q787"),
        wandb_artifact_path="climatepolicyradar/Q787/labelled-passages:v0",
        config=test_config,
    )

    patched_push_dependencies["argilla_cls"].assert_called_once_with(
        api_url=test_config.argilla_api_url,
        api_key=test_config.argilla_api_key.get_secret_value(),
    )


@pytest.mark.asyncio
async def test_push_new_dataset_passes_wikibase_credentials_from_config(
    patched_push_dependencies, test_config
):
    """WikibaseSession should be initialised with credentials from Config."""
    await push_new_dataset(
        wikibase_id=WikibaseID("Q787"),
        wandb_artifact_path="climatepolicyradar/Q787/labelled-passages:v0",
        config=test_config,
    )

    patched_push_dependencies["wikibase_cls"].assert_called_once_with(
        username=test_config.wikibase_username,
        password=test_config.wikibase_password.get_secret_value(),
        url=test_config.wikibase_url,
    )


@pytest.mark.asyncio
async def test_push_new_dataset_prelabels_with_an_llm_ensemble_by_default(
    patched_push_dependencies, mock_concept, test_config, monkeypatch
):
    """The ensemble is built for the concept and passed on as the suggestion model."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    await push_new_dataset(
        wikibase_id=WikibaseID("Q787"),
        wandb_artifact_path="climatepolicyradar/Q787/labelled-passages:v0",
        config=test_config,
    )

    create_ensemble = patched_push_dependencies["create_ensemble"]
    create_ensemble.assert_called_once_with(mock_concept)

    call_kwargs = patched_push_dependencies[
        "argilla"
    ].add_labelled_passages.call_args.kwargs
    assert call_kwargs["suggestion_model"] is create_ensemble.return_value

    # pydantic-ai reads the OpenRouter key from the environment
    assert (
        os.environ["OPENROUTER_API_KEY"]
        == test_config.openrouter_api_key.get_secret_value()
    )


@pytest.mark.asyncio
async def test_push_new_dataset_skips_prelabelling_when_disabled(
    patched_push_dependencies, test_config
):
    """With the flag off, no ensemble is built and no suggestions are pushed."""
    await push_new_dataset(
        wikibase_id=WikibaseID("Q787"),
        wandb_artifact_path="climatepolicyradar/Q787/labelled-passages:v0",
        prelabel_with_llm_ensemble=False,
        config=test_config,
    )

    patched_push_dependencies["create_ensemble"].assert_not_called()

    call_kwargs = patched_push_dependencies[
        "argilla"
    ].add_labelled_passages.call_args.kwargs
    assert call_kwargs["suggestion_model"] is None
