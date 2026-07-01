from io import BytesIO
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from flows.build_dataset import COMBINED_S3_KEY, SAMPLED_S3_KEY
from flows.sample import (
    load_dataset_from_s3,
    run_sampling_task,
    sample,
)
from knowledge_graph.cloud import AwsEnv
from knowledge_graph.identifiers import WikibaseID


def _feather_bytes(df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    df.to_feather(buffer)
    return buffer.getvalue()


def _mock_async_session(feather_bytes: bytes) -> MagicMock:
    """Build a mock aioboto3 session whose S3 client returns ``feather_bytes``."""
    mock_body = MagicMock()
    mock_body.read = AsyncMock(return_value=feather_bytes)

    mock_s3 = MagicMock()
    mock_s3.get_object = AsyncMock(return_value={"Body": mock_body})

    client_cm = MagicMock()
    client_cm.__aenter__ = AsyncMock(return_value=mock_s3)
    client_cm.__aexit__ = AsyncMock(return_value=None)

    mock_session = MagicMock()
    mock_session.client.return_value = client_cm
    return mock_session, mock_s3


@pytest.mark.asyncio
async def test_load_dataset_from_s3_reads_balanced_dataset_key(test_config):
    df = pd.DataFrame({"text_block.text": ["a", "b"]})
    mock_session, mock_s3 = _mock_async_session(_feather_bytes(df))

    with patch("flows.sample.get_async_session", return_value=mock_session):
        result = await load_dataset_from_s3.fn(
            dataset_name="balanced", config=test_config, aws_env=AwsEnv.sandbox
        )

    assert list(result["text_block.text"]) == ["a", "b"]
    assert mock_s3.get_object.call_args.kwargs["Key"] == SAMPLED_S3_KEY
    assert (
        mock_s3.get_object.call_args.kwargs["Bucket"] == test_config.dataset_s3_bucket
    )


@pytest.mark.asyncio
async def test_load_dataset_from_s3_reads_combined_dataset_key(test_config):
    df = pd.DataFrame({"text_block.text": ["a"]})
    mock_session, mock_s3 = _mock_async_session(_feather_bytes(df))

    with patch("flows.sample.get_async_session", return_value=mock_session):
        await load_dataset_from_s3.fn(
            dataset_name="combined", config=test_config, aws_env=AwsEnv.sandbox
        )

    assert mock_s3.get_object.call_args.kwargs["Key"] == COMBINED_S3_KEY


@pytest.mark.asyncio
async def test_load_dataset_from_s3_raises_for_unknown_dataset_name(test_config):
    with pytest.raises(ValueError, match="Unknown dataset_name"):
        await load_dataset_from_s3.fn(
            dataset_name="nonsense", config=test_config, aws_env=AwsEnv.sandbox
        )


@pytest.mark.asyncio
async def test_load_dataset_from_s3_reraises_on_s3_error(test_config):
    mock_session = MagicMock()
    client_cm = MagicMock()
    client_cm.__aenter__ = AsyncMock(side_effect=RuntimeError("s3 down"))
    client_cm.__aexit__ = AsyncMock(return_value=None)
    mock_session.client.return_value = client_cm

    with patch("flows.sample.get_async_session", return_value=mock_session):
        with pytest.raises(RuntimeError, match="s3 down"):
            await load_dataset_from_s3.fn(
                dataset_name="balanced", config=test_config, aws_env=AwsEnv.sandbox
            )


@pytest.mark.asyncio
async def test_run_sampling_task_passes_through_to_run_sampling():
    dataset = pd.DataFrame({"text_block.text": ["a"]})

    with (
        patch(
            "flows.sample.run_sampling", return_value="entity/Q1/labelled-passages:v1"
        ) as mock_run_sampling,
        patch(
            "flows.sample.parse_kwargs_from_strings", return_value={"definition": "x"}
        ) as mock_parse,
    ):
        result = await run_sampling_task.fn(
            wikibase_id=WikibaseID("Q1"),
            dataset=dataset,
            dataset_name="balanced",
            sample_size=130,
            min_negative_proportion=0.1,
            corpus_types_include=None,
            corpus_types_exclude=None,
            max_size_to_sample_from=500_000,
            max_negative_proportion=None,
            track_and_upload=True,
            concept_override=["definition=x"],
            wikibase_username="user",
            wikibase_password="pass",
            wikibase_url="https://wikibase.test",
        )

    assert result == "entity/Q1/labelled-passages:v1"
    mock_parse.assert_called_once_with(["definition=x"])
    got = mock_run_sampling.call_args.kwargs
    assert got["wikibase_id"] == WikibaseID("Q1")
    assert got["sample_size"] == 130
    assert got["track_and_upload"] is True
    assert got["concept_overrides"] == {"definition": "x"}
    assert got["wikibase_username"] == "user"


@pytest.mark.asyncio
async def test_sample_flow_logs_in_to_wandb_when_tracking(test_config):
    with (
        patch(
            "flows.sample.load_dataset_from_s3",
            new_callable=AsyncMock,
            return_value=pd.DataFrame({"text_block.text": ["a"]}),
        ),
        patch(
            "flows.sample.run_sampling_task",
            new_callable=AsyncMock,
            return_value="artifact:v1",
        ),
        patch("flows.sample.login_to_wandb") as mock_login,
    ):
        result = await sample(
            wikibase_id=WikibaseID("Q1"),
            track_and_upload=True,
            config=test_config,
        )

    assert result == "artifact:v1"
    mock_login.assert_called_once()


@pytest.mark.asyncio
async def test_sample_flow_skips_wandb_login_when_not_tracking(test_config):
    with (
        patch(
            "flows.sample.load_dataset_from_s3",
            new_callable=AsyncMock,
            return_value=pd.DataFrame({"text_block.text": ["a"]}),
        ),
        patch(
            "flows.sample.run_sampling_task",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch("flows.sample.login_to_wandb") as mock_login,
    ):
        await sample(
            wikibase_id=WikibaseID("Q1"),
            track_and_upload=False,
            config=test_config,
        )

    mock_login.assert_not_called()


@pytest.mark.asyncio
async def test_sample_flow_passes_wikibase_credentials_from_config(test_config):
    with (
        patch(
            "flows.sample.load_dataset_from_s3",
            new_callable=AsyncMock,
            return_value=pd.DataFrame({"text_block.text": ["a"]}),
        ),
        patch(
            "flows.sample.run_sampling_task",
            new_callable=AsyncMock,
            return_value=None,
        ) as mock_run_sampling_task,
        patch("flows.sample.login_to_wandb"),
    ):
        await sample(
            wikibase_id=WikibaseID("Q1"),
            track_and_upload=False,
            config=test_config,
        )

    got = mock_run_sampling_task.call_args.kwargs
    assert got["wikibase_username"] == test_config.wikibase_username
    assert got["wikibase_password"] == test_config.wikibase_password.get_secret_value()
    assert got["wikibase_url"] == test_config.wikibase_url
