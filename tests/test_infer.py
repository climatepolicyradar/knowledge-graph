from unittest.mock import AsyncMock, patch

import pytest
import typer
from typer.testing import CliRunner

from scripts.infer import app, convert_classifier_specs, main

runner = CliRunner()


def test_convert_classifier_specs_with_name_only():
    input_specs = ["Q123"]
    result = convert_classifier_specs(input_specs)
    assert result == [{"name": "Q123", "alias": "latest"}]


def test_convert_classifier_specs_with_name_and_alias():
    input_specs = ["Q123:v1"]
    result = convert_classifier_specs(input_specs)
    assert result == [{"name": "Q123", "alias": "v1"}]


def test_convert_classifier_specs_multiple_specs():
    input_specs = ["Q123", "Q456:v2"]
    result = convert_classifier_specs(input_specs)
    assert result == [
        {"name": "Q123", "alias": "latest"},
        {"name": "Q456", "alias": "v2"},
    ]


def test_convert_classifier_specs_invalid_format():
    with pytest.raises(typer.BadParameter):
        convert_classifier_specs(["Q123:v1:extra"])


def test_cli_basic():
    """Test the CLI interface with basic options"""
    mock_run = AsyncMock()
    mock_run.return_value.id = "test-id"

    with patch("scripts.infer.run_deployment", new=mock_run):
        result = runner.invoke(app, ["--aws-env", "staging", "-c", "Q123"])
        assert result.exit_code == 0

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["parameters"]["classifier_specs"] == [
            {"name": "Q123", "alias": "latest"}
        ]
        assert call_kwargs["parameters"]["document_ids"] is None


def test_cli_with_documents():
    """Test the CLI interface with documents"""
    mock_run = AsyncMock()
    mock_run.return_value.id = "test-id"

    with patch("scripts.infer.run_deployment", new=mock_run):
        result = runner.invoke(
            app,
            [
                "--aws-env",
                "staging",
                "-c",
                "Q123:v1",
                "-d",
                "doc1",
                "-d",
                "doc2",
            ],
        )
        assert result.exit_code == 0

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["parameters"]["classifier_specs"] == [
            {"name": "Q123", "alias": "v1"}
        ]
        assert call_kwargs["parameters"]["document_ids"] == ["doc1", "doc2"]


def test_cli_no_options():
    """Test the CLI interface with minimal arguments"""
    mock_run = AsyncMock()
    mock_run.return_value.id = "test-id"

    with patch("scripts.infer.run_deployment", new=mock_run):
        result = runner.invoke(app, ["--aws-env", "staging"])
        assert result.exit_code == 0

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["parameters"]["classifier_specs"] is None
        assert call_kwargs["parameters"]["document_ids"] is None


@pytest.mark.asyncio
async def test_main_function_basic():
    """Test the core function with a classifier"""
    mock_run = AsyncMock()
    mock_run.return_value.id = "test-id"

    with patch("scripts.infer.run_deployment", new=mock_run):
        await main(
            aws_env="staging",
            classifiers=convert_classifier_specs(["Q123"]),
            documents=[],
        )

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["parameters"]["classifier_specs"] == [
            {"name": "Q123", "alias": "latest"}
        ]
        assert call_kwargs["parameters"]["document_ids"] is None


@pytest.mark.asyncio
async def test_main_function_with_documents():
    """Test the core function with both classifier and documents"""
    mock_run = AsyncMock()
    mock_run.return_value.id = "test-id"

    with patch("scripts.infer.run_deployment", new=mock_run):
        await main(
            aws_env="staging",
            classifiers=convert_classifier_specs(["Q123:v1"]),
            documents=["doc1", "doc2"],
        )

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["parameters"]["classifier_specs"] == [
            {"name": "Q123", "alias": "v1"}
        ]
        assert call_kwargs["parameters"]["document_ids"] == ["doc1", "doc2"]


@pytest.mark.asyncio
async def test_main_function_no_options():
    """Test the core function with minimal arguments"""
    mock_run = AsyncMock()
    mock_run.return_value.id = "test-id"

    with patch("scripts.infer.run_deployment", new=mock_run):
        await main(
            aws_env="staging",
            classifiers=[],
            documents=[],
        )

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["parameters"]["classifier_specs"] is None
        assert call_kwargs["parameters"]["document_ids"] is None
