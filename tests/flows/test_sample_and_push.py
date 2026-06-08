from unittest.mock import AsyncMock, patch

import pytest

from flows.sample_and_push import sample_and_push
from knowledge_graph.identifiers import WikibaseID


@pytest.fixture
def patched_subflows(test_config):
    with (
        patch("flows.sample_and_push.sample", new_callable=AsyncMock) as mock_sample,
        patch(
            "flows.sample_and_push.push_new_dataset", new_callable=AsyncMock
        ) as mock_push,
        patch("flows.sample_and_push.Config") as mock_config_cls,
    ):
        mock_config_cls.create = AsyncMock(return_value=test_config)
        yield {
            "sample": mock_sample,
            "push": mock_push,
        }


@pytest.mark.asyncio
async def test_sample_and_push_calls_sample_then_push(patched_subflows, test_config):
    """Sample should be called before push_new_dataset."""
    await sample_and_push(wikibase_id=WikibaseID("Q787"), config=test_config)

    assert patched_subflows["sample"].called
    assert patched_subflows["push"].called


@pytest.mark.asyncio
async def test_sample_and_push_forces_track_and_upload(patched_subflows, test_config):
    """track_and_upload must always be True so the artifact exists for push_new_dataset."""
    await sample_and_push(wikibase_id=WikibaseID("Q787"), config=test_config)

    _, kwargs = patched_subflows["sample"].call_args
    assert kwargs["track_and_upload"] is True


@pytest.mark.asyncio
async def test_sample_and_push_uses_latest_artifact(patched_subflows, test_config):
    """push_new_dataset should receive the :latest artifact path for the concept."""
    await sample_and_push(wikibase_id=WikibaseID("Q787"), config=test_config)

    _, kwargs = patched_subflows["push"].call_args
    assert kwargs["wandb_artifact_path"] == (
        "climatepolicyradar/Q787/labelled-passages:latest"
    )


@pytest.mark.asyncio
async def test_sample_and_push_passes_wikibase_id_to_both_flows(
    patched_subflows, test_config
):
    """Both sub-flows should receive the same wikibase_id."""
    await sample_and_push(wikibase_id=WikibaseID("Q787"), config=test_config)

    _, sample_kwargs = patched_subflows["sample"].call_args
    _, push_kwargs = patched_subflows["push"].call_args
    assert sample_kwargs["wikibase_id"] == WikibaseID("Q787")
    assert push_kwargs["wikibase_id"] == WikibaseID("Q787")


@pytest.mark.asyncio
async def test_sample_and_push_passes_workspace_name(patched_subflows, test_config):
    """workspace_name should be forwarded to push_new_dataset."""
    await sample_and_push(
        wikibase_id=WikibaseID("Q787"),
        workspace_name="my-workspace",
        config=test_config,
    )

    _, kwargs = patched_subflows["push"].call_args
    assert kwargs["workspace_name"] == "my-workspace"


@pytest.mark.asyncio
async def test_sample_and_push_passes_sample_params(patched_subflows, test_config):
    """Non-default sample params should be forwarded to the sample flow."""
    await sample_and_push(
        wikibase_id=WikibaseID("Q787"),
        dataset_name="combined",
        sample_size=50,
        min_negative_proportion=0.2,
        config=test_config,
    )

    _, kwargs = patched_subflows["sample"].call_args
    assert kwargs["dataset_name"] == "combined"
    assert kwargs["sample_size"] == 50
    assert kwargs["min_negative_proportion"] == 0.2
