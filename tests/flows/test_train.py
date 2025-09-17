from unittest.mock import AsyncMock, patch

import pytest

from flows.train import train_on_gpu
from knowledge_graph.cloud import AwsEnv
from knowledge_graph.identifiers import WikibaseID


@pytest.mark.asyncio
async def test_train_on_gpu(mock_s3_client, mock_wandb, test_config):
    with patch(
        "flows.train.run_training", return_value=AsyncMock()
    ) as mock_run_training:
        pass_through_kwargs = {
            "wikibase_id": WikibaseID("Q1"),
            "track": False,
            "upload": False,
        }

        _ = await train_on_gpu(
            wikibase_id=pass_through_kwargs["wikibase_id"],
            track=pass_through_kwargs["track"],
            upload=pass_through_kwargs["upload"],
            aws_env=AwsEnv.labs,
            config=test_config,
        )

        mock_run_training.assert_called_once()
        got_kwargs = mock_run_training.call_args.kwargs
        for kwarg, value in pass_through_kwargs.items():
            assert got_kwargs[kwarg] == value, (
                f"run_training expected {pass_through_kwargs}, but got {got_kwargs}"
            )
