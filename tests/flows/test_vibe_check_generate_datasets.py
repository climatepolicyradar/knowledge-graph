from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from flows.vibe_check_generate_datasets import (
    _set_up_environment,
    generate_vibe_checker_datasets,
    upload_vibe_checker_files,
)


@pytest.mark.asyncio
async def test_generate_vibe_checker_datasets_wires_load_embed_upload(test_config):
    dataset_df = pd.DataFrame({"text_block.text": ["a", "b"]})
    embeddings = np.zeros((2, 4), dtype=np.float32)
    mock_s3_client = MagicMock()

    with (
        patch(
            "flows.vibe_check_generate_datasets._set_up_environment",
            new_callable=AsyncMock,
            return_value=(test_config, mock_s3_client),
        ),
        patch(
            "flows.vibe_check_generate_datasets.load_dataset_from_s3",
            new_callable=AsyncMock,
            return_value=dataset_df,
        ) as mock_load,
        patch(
            "flows.vibe_check_generate_datasets.generate_embeddings",
            return_value=embeddings,
        ) as mock_generate,
        patch(
            "flows.vibe_check_generate_datasets.upload_vibe_checker_files",
        ) as mock_upload,
    ):
        await generate_vibe_checker_datasets(
            embedding_model_name="test-model",
            batch_size=8,
            config=test_config,
        )

    assert mock_load.call_args.kwargs["dataset_name"] == "balanced"
    assert mock_generate.call_args.kwargs == {
        "embedding_model_name": "test-model",
        "batch_size": 8,
    }
    upload_args = mock_upload.call_args
    assert upload_args.args[0] is mock_s3_client
    assert upload_args.args[1] is dataset_df
    assert upload_args.args[2] is embeddings


@pytest.mark.asyncio
async def test_set_up_environment_returns_config_and_client(test_config, monkeypatch):
    monkeypatch.delenv("USE_AWS_PROFILES", raising=False)
    mock_session = MagicMock()
    mock_session.client.return_value = "s3-client"

    with patch(
        "flows.vibe_check_generate_datasets.boto3.session.Session",
        return_value=mock_session,
    ) as mock_session_cls:
        config, s3_client = await _set_up_environment(config=test_config)

    assert config is test_config
    assert s3_client == "s3-client"
    # With USE_AWS_PROFILES unset, no profile should be passed.
    assert mock_session_cls.call_args.kwargs["profile_name"] is None
    assert mock_session_cls.call_args.kwargs["region_name"] == test_config.bucket_region


@pytest.mark.asyncio
async def test_set_up_environment_uses_profile_when_enabled(test_config, monkeypatch):
    monkeypatch.setenv("USE_AWS_PROFILES", "true")
    mock_session = MagicMock()

    with patch(
        "flows.vibe_check_generate_datasets.boto3.session.Session",
        return_value=mock_session,
    ) as mock_session_cls:
        await _set_up_environment(config=test_config)

    assert (
        mock_session_cls.call_args.kwargs["profile_name"] == test_config.aws_env.value
    )


def test_upload_vibe_checker_files_pushes_three_objects():
    df = pd.DataFrame({"text_block.text": ["a", "b", "c"]})
    embeddings = np.ones((3, 4), dtype=np.float32)
    s3_client = MagicMock()

    with (
        patch(
            "flows.vibe_check_generate_datasets.get_bucket_name_from_ssm",
            return_value="vibe-bucket",
        ),
        patch(
            "flows.vibe_check_generate_datasets.push_object_bytes_to_s3",
        ) as mock_push,
    ):
        upload_vibe_checker_files.fn(
            s3_client,
            df,
            embeddings,
            embedding_model_name="test-model",
            batch_size=8,
        )

    pushed_keys = [call.args[1] for call in mock_push.call_args_list]
    assert pushed_keys == [
        "passages_dataset.feather",
        "passages_embeddings.npy",
        "passages_embeddings_metadata.json",
    ]
    assert mock_push.call_count == 3
