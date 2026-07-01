import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from flows.train import (
    _set_up_training_environment,
    load_wikibase_ids_from_config,
    train_from_config,
    train_on_cpu,
    train_on_gpu,
)
from knowledge_graph.cloud import AwsEnv
from knowledge_graph.identifiers import WikibaseID


@pytest.mark.asyncio
@pytest.mark.parametrize("train_flow", [train_on_gpu, train_on_cpu])
async def test_train_flow(train_flow, mock_s3_client, mock_wandb, test_config):
    # Create a mock session that returns our mock_s3_client
    mock_session = MagicMock()
    mock_session.client.return_value = mock_s3_client

    with (
        patch(
            "flows.train.run_training", return_value=AsyncMock()
        ) as mock_run_training,
        patch("boto3.session.Session", return_value=mock_session),
    ):
        pass_through_kwargs = {
            "wikibase_id": WikibaseID("Q1"),
            "track_and_upload": False,
        }

        _ = await train_flow(
            wikibase_id=pass_through_kwargs["wikibase_id"],
            track_and_upload=pass_through_kwargs["track_and_upload"],
            aws_env=AwsEnv.labs,
            config=test_config,
        )

        mock_run_training.assert_called_once()
        got_kwargs = mock_run_training.call_args.kwargs
        for kwarg, value in pass_through_kwargs.items():
            assert got_kwargs[kwarg] == value, (
                f"run_training expected {pass_through_kwargs}, but got {got_kwargs}"
            )


@pytest.mark.asyncio
async def test_set_up_training_environment_builds_configs_from_config(
    test_config, monkeypatch
):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    mock_session = MagicMock()
    mock_session.client.return_value = "s3-client"

    with (
        patch("flows.train.wandb") as mock_wandb,
        patch("boto3.session.Session", return_value=mock_session),
    ):
        (
            config,
            wikibase_config,
            argilla_config,
            s3_client,
        ) = await _set_up_training_environment(config=test_config, aws_env=AwsEnv.labs)

    assert config is test_config
    assert wikibase_config.username == test_config.wikibase_username
    assert (
        wikibase_config.password.get_secret_value()
        == test_config.wikibase_password.get_secret_value()
    )
    assert str(wikibase_config.url) == test_config.wikibase_url
    assert argilla_config.api_key.get_secret_value() == (
        test_config.argilla_api_key.get_secret_value()
    )
    assert s3_client == "s3-client"
    mock_wandb.login.assert_called_once_with(
        key=test_config.wandb_api_key.get_secret_value()
    )
    assert os.environ["OPENROUTER_API_KEY"] == (
        test_config.openrouter_api_key.get_secret_value()
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "missing_field",
    [
        "wikibase_username",
        "wikibase_password",
        "wikibase_url",
        "argilla_api_key",
        "argilla_api_url",
    ],
)
async def test_set_up_training_environment_raises_when_config_incomplete(
    test_config, missing_field
):
    incomplete_config = test_config.model_copy(update={missing_field: None})

    with (
        patch("flows.train.wandb"),
        patch("boto3.session.Session"),
    ):
        with pytest.raises(ValueError, match="Missing values in config"):
            await _set_up_training_environment(
                config=incomplete_config, aws_env=AwsEnv.labs
            )


def test_load_wikibase_ids_from_config_returns_unique_ids(tmp_path):
    config_file = tmp_path / "concepts.yml"
    config_file.write_text("- Q1\n- Q2\n- Q2\n")

    wikibase_ids = load_wikibase_ids_from_config.fn(str(config_file))

    assert set(wikibase_ids) == {WikibaseID("Q1"), WikibaseID("Q2")}


def test_load_wikibase_ids_from_config_raises_when_file_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_wikibase_ids_from_config.fn(str(tmp_path / "does_not_exist.yml"))


def test_load_wikibase_ids_from_config_raises_for_invalid_yaml(tmp_path):
    config_file = tmp_path / "concepts.yml"
    config_file.write_text("- [unclosed\n")

    with pytest.raises(ValueError, match="valid YAML"):
        load_wikibase_ids_from_config.fn(str(config_file))


def test_load_wikibase_ids_from_config_raises_for_empty_config(tmp_path):
    config_file = tmp_path / "concepts.yml"
    config_file.write_text("[]\n")

    with pytest.raises(ValueError, match="No concepts found"):
        load_wikibase_ids_from_config.fn(str(config_file))


@pytest.mark.asyncio
async def test_train_from_config_aggregates_successes_and_failures(test_config):
    with (
        patch(
            "flows.train._set_up_training_environment",
            new_callable=AsyncMock,
            return_value=(test_config, MagicMock(), MagicMock(), MagicMock()),
        ),
        patch(
            "flows.train.load_wikibase_ids_from_config",
            return_value=[WikibaseID("Q1"), WikibaseID("Q2")],
        ),
        patch(
            "flows.train.run_training",
            new_callable=AsyncMock,
            side_effect=["classifier-Q1", RuntimeError("training failed")],
        ),
    ):
        results = await train_from_config(
            config_file_path="ignored.yml",
            track_and_upload=False,
            config=test_config,
        )

    assert len(results) == 2
    successes = [r for r in results if not isinstance(r, Exception)]
    failures = [r for r in results if isinstance(r, Exception)]
    assert successes == ["classifier-Q1"]
    assert len(failures) == 1
    assert isinstance(failures[0], RuntimeError)
