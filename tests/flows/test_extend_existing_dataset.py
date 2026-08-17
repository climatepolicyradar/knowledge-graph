import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from flows.extend_existing_dataset import extend_existing_dataset
from knowledge_graph.identifiers import Identifier, WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage

WIKIBASE_ID = WikibaseID("Q2419")
ARTIFACT = "climatepolicyradar/Q2419/labelled-passages:v4"


@pytest.fixture
def labelled_passages():
    return [LabelledPassage(text=f"passage {i}", spans=[]) for i in range(10)]


@pytest.fixture
def mock_concept():
    concept = MagicMock()
    concept.wikibase_id = WIKIBASE_ID
    return concept


EXISTING_IDS = {Identifier.generate(f"existing {i}") for i in range(3)}


@pytest.fixture
def patched_extend_dependencies(labelled_passages, mock_concept, test_config):
    with (
        patch(
            "flows.extend_existing_dataset.load_labelled_passages_from_wandb"
        ) as mock_load,
        patch("flows.extend_existing_dataset.wandb.login") as mock_wandb_login,
        patch("flows.extend_existing_dataset.Config") as mock_config_cls,
        patch("flows.extend_existing_dataset.WikibaseSession") as mock_wikibase_cls,
        patch(
            "flows.extend_existing_dataset.create_llm_ensemble_for_prelabelling_validation_dataset"
        ) as mock_create_ensemble,
        patch("flows.extend_existing_dataset.run_extend_dataset") as mock_run_extend,
        patch(
            "flows.extend_existing_dataset.get_passage_ids_in_argilla",
            return_value=EXISTING_IDS,
        ) as mock_get_ids,
        patch(
            "flows.extend_existing_dataset.sample",
            new_callable=AsyncMock,
            return_value=ARTIFACT,
        ) as mock_sample,
    ):
        mock_load.return_value = labelled_passages
        mock_config_cls.create = AsyncMock(return_value=test_config)
        mock_run_extend.return_value = labelled_passages[:3]

        mock_wikibase = MagicMock()
        mock_wikibase_cls.return_value = mock_wikibase
        mock_wikibase.get_concept_async = AsyncMock(return_value=mock_concept)

        yield {
            "load": mock_load,
            "wandb_login": mock_wandb_login,
            "wikibase_cls": mock_wikibase_cls,
            "create_ensemble": mock_create_ensemble,
            "run_extend": mock_run_extend,
            "get_ids": mock_get_ids,
            "sample": mock_sample,
        }


@pytest.mark.asyncio
async def test_samples_when_given_no_artifact_path(
    patched_extend_dependencies, test_config
):
    """
    With no artifact the flow samples for itself — one command, not two.

    sample_size defaults to n_new_passages because the passages already in Argilla are
    excluded from the pool, so there is nothing to overshoot.
    """
    await extend_existing_dataset(
        wikibase_id=WIKIBASE_ID, n_new_passages=50, config=test_config
    )

    kwargs = patched_extend_dependencies["sample"].call_args.kwargs
    assert kwargs["sample_size"] == 50
    assert kwargs["exclude_passage_ids"] == sorted(EXISTING_IDS)
    assert kwargs["track_and_upload"] is True
    patched_extend_dependencies["load"].assert_called_once_with(wandb_path=ARTIFACT)


@pytest.mark.asyncio
async def test_does_not_sample_when_given_an_artifact_path(
    patched_extend_dependencies, test_config
):
    await extend_existing_dataset(
        wikibase_id=WIKIBASE_ID, wandb_artifact_path=ARTIFACT, config=test_config
    )

    patched_extend_dependencies["sample"].assert_not_called()


@pytest.mark.asyncio
async def test_explicit_sample_size_overrides_n_new_passages(
    patched_extend_dependencies, test_config
):
    await extend_existing_dataset(
        wikibase_id=WIKIBASE_ID,
        n_new_passages=50,
        sample_size=200,
        config=test_config,
    )

    assert patched_extend_dependencies["sample"].call_args.kwargs["sample_size"] == 200


@pytest.mark.asyncio
async def test_raises_when_sample_returns_no_artifact(
    patched_extend_dependencies, test_config
):
    patched_extend_dependencies["sample"].return_value = None

    with pytest.raises(RuntimeError, match="did not return an artifact path"):
        await extend_existing_dataset(wikibase_id=WIKIBASE_ID, config=test_config)

    patched_extend_dependencies["run_extend"].assert_not_called()


@pytest.mark.asyncio
async def test_raises_when_no_size_is_given_to_sample_with(
    patched_extend_dependencies, test_config
):
    with pytest.raises(ValueError, match="No sample size"):
        await extend_existing_dataset(
            wikibase_id=WIKIBASE_ID,
            n_new_passages=None,
            sample_size=None,
            config=test_config,
        )

    patched_extend_dependencies["sample"].assert_not_called()


@pytest.mark.asyncio
async def test_reads_argilla_once_and_reuses_the_ids(
    patched_extend_dependencies, test_config
):
    """The IDs are fetched up front and handed to both sampling and the extend op."""
    await extend_existing_dataset(wikibase_id=WIKIBASE_ID, config=test_config)

    patched_extend_dependencies["get_ids"].assert_called_once()
    kwargs = patched_extend_dependencies["run_extend"].call_args.kwargs
    assert kwargs["existing_passage_ids"] == EXISTING_IDS


@pytest.mark.asyncio
async def test_loads_candidate_passages_from_the_given_artifact(
    patched_extend_dependencies, test_config
):
    """The flow reads the output of a prior sample run."""
    await extend_existing_dataset(
        wikibase_id=WIKIBASE_ID, wandb_artifact_path=ARTIFACT, config=test_config
    )

    patched_extend_dependencies["load"].assert_called_once_with(wandb_path=ARTIFACT)


@pytest.mark.asyncio
async def test_forwards_limit_workspace_and_require_full_limit(
    patched_extend_dependencies, labelled_passages, test_config
):
    """n_new_passages becomes the operation's limit."""
    await extend_existing_dataset(
        wikibase_id=WIKIBASE_ID,
        wandb_artifact_path=ARTIFACT,
        n_new_passages=50,
        require_full_limit=True,
        workspace_name="my-workspace",
        config=test_config,
    )

    kwargs = patched_extend_dependencies["run_extend"].call_args.kwargs
    assert kwargs["limit"] == 50
    assert kwargs["require_full_limit"] is True
    assert kwargs["workspace"] == "my-workspace"
    assert kwargs["labelled_passages"] == labelled_passages


@pytest.mark.asyncio
async def test_passes_argilla_credentials_from_config(
    patched_extend_dependencies, test_config
):
    """The flow resolves credentials; the operation builds the session."""
    await extend_existing_dataset(
        wikibase_id=WIKIBASE_ID, wandb_artifact_path=ARTIFACT, config=test_config
    )

    kwargs = patched_extend_dependencies["run_extend"].call_args.kwargs
    assert kwargs["argilla_api_url"] == test_config.argilla_api_url
    assert kwargs["argilla_api_key"] == test_config.argilla_api_key.get_secret_value()


@pytest.mark.asyncio
async def test_raises_when_argilla_credentials_are_missing(
    patched_extend_dependencies, test_config, monkeypatch
):
    """
    Config swallows a missing Argilla SSM param outside labs.

    Without this guard the run would fail later on an opaque auth error. pyproject sets
    both vars for the test session, so they have to be removed here.
    """
    monkeypatch.delenv("ARGILLA_API_URL", raising=False)
    monkeypatch.delenv("ARGILLA_API_KEY", raising=False)
    config_without_argilla = test_config.model_copy(
        update={"argilla_api_url": None, "argilla_api_key": None}
    )

    with pytest.raises(ValueError, match="Missing Argilla credentials"):
        await extend_existing_dataset(
            wikibase_id=WIKIBASE_ID,
            wandb_artifact_path=ARTIFACT,
            config=config_without_argilla,
        )

    patched_extend_dependencies["run_extend"].assert_not_called()


@pytest.mark.asyncio
async def test_logs_into_wandb_when_api_key_present(
    patched_extend_dependencies, test_config
):
    await extend_existing_dataset(
        wikibase_id=WIKIBASE_ID, wandb_artifact_path=ARTIFACT, config=test_config
    )

    patched_extend_dependencies["wandb_login"].assert_called_once_with(
        key=test_config.wandb_api_key.get_secret_value()
    )


@pytest.mark.asyncio
async def test_skips_wandb_login_without_api_key(
    patched_extend_dependencies, test_config
):
    config_no_key = test_config.model_copy(update={"wandb_api_key": None})

    await extend_existing_dataset(
        wikibase_id=WIKIBASE_ID, wandb_artifact_path=ARTIFACT, config=config_no_key
    )

    patched_extend_dependencies["wandb_login"].assert_not_called()


@pytest.mark.asyncio
async def test_skips_prelabelling_by_default(patched_extend_dependencies, test_config):
    await extend_existing_dataset(
        wikibase_id=WIKIBASE_ID, wandb_artifact_path=ARTIFACT, config=test_config
    )

    patched_extend_dependencies["create_ensemble"].assert_not_called()
    patched_extend_dependencies["wikibase_cls"].assert_not_called()
    kwargs = patched_extend_dependencies["run_extend"].call_args.kwargs
    assert kwargs["suggestion_model"] is None


@pytest.mark.asyncio
async def test_prelabels_with_an_llm_ensemble_when_enabled(
    patched_extend_dependencies, mock_concept, test_config, monkeypatch
):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    await extend_existing_dataset(
        wikibase_id=WIKIBASE_ID,
        wandb_artifact_path=ARTIFACT,
        prelabel_with_llm_ensemble=True,
        config=test_config,
    )

    create_ensemble = patched_extend_dependencies["create_ensemble"]
    create_ensemble.assert_called_once_with(mock_concept)

    kwargs = patched_extend_dependencies["run_extend"].call_args.kwargs
    assert kwargs["suggestion_model"] is create_ensemble.return_value

    # pydantic-ai reads the OpenRouter key from the environment
    assert (
        os.environ["OPENROUTER_API_KEY"]
        == test_config.openrouter_api_key.get_secret_value()
    )


@pytest.mark.asyncio
async def test_returns_the_number_of_passages_added(
    patched_extend_dependencies, test_config
):
    added = await extend_existing_dataset(
        wikibase_id=WIKIBASE_ID, wandb_artifact_path=ARTIFACT, config=test_config
    )

    assert added == 3
