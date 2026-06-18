from unittest.mock import AsyncMock, patch

import pytest

from flows.create_evaluation_dataset_in_argilla import (
    create_evaluation_dataset_in_argilla,
)
from knowledge_graph.identifiers import WikibaseID


@pytest.fixture
def patched_subflows(test_config):
    with (
        patch(
            "flows.create_evaluation_dataset_in_argilla.sample", new_callable=AsyncMock
        ) as mock_sample,
        patch(
            "flows.create_evaluation_dataset_in_argilla.push_new_dataset",
            new_callable=AsyncMock,
        ) as mock_push,
        patch("flows.create_evaluation_dataset_in_argilla.Config") as mock_config_cls,
    ):
        mock_config_cls.create = AsyncMock(return_value=test_config)
        yield {
            "sample": mock_sample,
            "push": mock_push,
        }


@pytest.mark.asyncio
async def test_create_evaluation_dataset_in_argilla_calls_sample_then_push(
    patched_subflows, test_config
):
    """Sample must be called before push_new_dataset."""
    call_order = []

    async def record_sample(*args, **kwargs):
        call_order.append("sample")
        return "climatepolicyradar/Q787/labelled-passages:v0"

    async def record_push(*args, **kwargs):
        call_order.append("push")

    patched_subflows["sample"].side_effect = record_sample
    patched_subflows["push"].side_effect = record_push

    await create_evaluation_dataset_in_argilla(
        wikibase_id=WikibaseID("Q787"), config=test_config
    )

    assert call_order == ["sample", "push"]


@pytest.mark.asyncio
async def test_create_evaluation_dataset_in_argilla_forces_track_and_upload(
    patched_subflows, test_config
):
    """track_and_upload must always be True so the artifact exists for push_new_dataset."""
    await create_evaluation_dataset_in_argilla(
        wikibase_id=WikibaseID("Q787"), config=test_config
    )

    _, kwargs = patched_subflows["sample"].call_args
    assert kwargs["track_and_upload"] is True


@pytest.mark.asyncio
async def test_create_evaluation_dataset_in_argilla_passes_artifact_path_from_sample(
    patched_subflows, test_config
):
    """push_new_dataset should receive the exact artifact path returned by the sample flow."""
    artifact_path = "climatepolicyradar/Q787/labelled-passages:v3"
    patched_subflows["sample"].return_value = artifact_path

    await create_evaluation_dataset_in_argilla(
        wikibase_id=WikibaseID("Q787"), config=test_config
    )

    _, kwargs = patched_subflows["push"].call_args
    assert kwargs["wandb_artifact_path"] == artifact_path


@pytest.mark.asyncio
async def test_create_evaluation_dataset_in_argilla_raises_if_sample_returns_none(
    patched_subflows, test_config
):
    """If sample returns None (no artifact version assigned), the flow should raise rather than push."""
    patched_subflows["sample"].return_value = None

    with pytest.raises(RuntimeError, match="did not return an artifact path"):
        await create_evaluation_dataset_in_argilla(
            wikibase_id=WikibaseID("Q787"), config=test_config
        )

    patched_subflows["push"].assert_not_called()


@pytest.mark.asyncio
async def test_create_evaluation_dataset_in_argilla_passes_wikibase_id_to_both_flows(
    patched_subflows, test_config
):
    """Both sub-flows should receive the same wikibase_id."""
    await create_evaluation_dataset_in_argilla(
        wikibase_id=WikibaseID("Q787"), config=test_config
    )

    _, sample_kwargs = patched_subflows["sample"].call_args
    _, push_kwargs = patched_subflows["push"].call_args
    assert sample_kwargs["wikibase_id"] == WikibaseID("Q787")
    assert push_kwargs["wikibase_id"] == WikibaseID("Q787")


@pytest.mark.asyncio
async def test_create_evaluation_dataset_in_argilla_passes_workspace_name(
    patched_subflows, test_config
):
    """workspace_name should be forwarded to push_new_dataset."""
    await create_evaluation_dataset_in_argilla(
        wikibase_id=WikibaseID("Q787"),
        workspace_name="my-workspace",
        config=test_config,
    )

    _, kwargs = patched_subflows["push"].call_args
    assert kwargs["workspace_name"] == "my-workspace"


@pytest.mark.asyncio
async def test_create_evaluation_dataset_in_argilla_passes_sample_params(
    patched_subflows, test_config
):
    """Non-default sample params should be forwarded to the sample flow."""
    await create_evaluation_dataset_in_argilla(
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
