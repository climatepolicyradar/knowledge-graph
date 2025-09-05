from unittest.mock import AsyncMock, patch

import pytest
import typer
from typer.testing import CliRunner

from flows.classifier_specs.spec_interface import ClassifierSpec
from knowledge_graph.cloud import AwsEnv
from scripts.infer import app, convert_classifier_specs, main

runner = CliRunner()


@pytest.fixture
def spec_json():
    return '{"wikibase_id": "Q787", "classifier_id": "393fk3km", "wandb_registry_version": "v2"}'


@pytest.fixture
def spec_json_with_gpu():
    return '{"wikibase_id": "Q787", "classifier_id": "393fk3km", "compute_environment": {"gpu": true}, "wandb_registry_version": "v2"}'


def test_convert_classifier_specs_with_name_and_alias(spec_json):
    input_specs = [spec_json]
    result = convert_classifier_specs(input_specs)
    assert len(result) == 1


def test_convert_classifier_specs_multiple_specs(spec_json, spec_json_with_gpu):
    input_specs = [spec_json, spec_json_with_gpu]
    result = convert_classifier_specs(input_specs)
    assert len(result) == 2


def test_convert_classifier_specs_invalid_format():
    with pytest.raises(typer.BadParameter) as e:
        convert_classifier_specs(['{"some_other_id": "Q787", "gaps"}']), f"raised: {e}"


def test_cli_basic(spec_json):
    """Test the CLI interface with basic options"""
    mock_run = AsyncMock()
    mock_run.return_value.id = "test-id"

    with patch("scripts.infer.run_deployment", new=mock_run):
        result = runner.invoke(app, ["--aws-env", "staging", "-c", spec_json])
        assert result.exit_code == 0

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["parameters"]["classifier_specs"] == [
            ClassifierSpec.model_validate_json(spec_json)
        ]
        assert call_kwargs["parameters"]["document_ids"] is None


def test_cli_with_documents(spec_json_with_gpu):
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
                spec_json_with_gpu,
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
            ClassifierSpec.model_validate_json(spec_json_with_gpu)
        ]
        assert call_kwargs["parameters"]["document_ids"] == ["doc1", "doc2"]


def test_cli_no_options():
    """Test the CLI interface with minimal arguments"""
    mock_run = AsyncMock()
    mock_run.return_value.id = "test-id"

    with patch("scripts.infer.run_deployment", new=mock_run):
        result = runner.invoke(app, ["--aws-env", "staging"])
        assert result.exit_code == 0, result.output

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["parameters"]["classifier_specs"] is None
        assert call_kwargs["parameters"]["document_ids"] is None


def test_main_function_basic(spec_json):
    """Test the core function with a classifier"""
    mock_run = AsyncMock()
    mock_run.return_value.id = "test-id"

    with patch("scripts.infer.run_deployment", new=mock_run):
        main(
            aws_env=AwsEnv.staging,
            classifiers=convert_classifier_specs([spec_json]),
            documents=[],
        )

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["parameters"]["classifier_specs"] == [
            ClassifierSpec.model_validate_json(spec_json)
        ]
        assert call_kwargs["parameters"]["document_ids"] is None


def test_main_function_with_documents(spec_json):
    """Test the core function with both classifier and documents"""
    mock_run = AsyncMock()
    mock_run.return_value.id = "test-id"

    with patch("scripts.infer.run_deployment", new=mock_run):
        main(
            aws_env=AwsEnv.staging,
            classifiers=convert_classifier_specs([spec_json]),
            documents=["doc1", "doc2"],
        )

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["parameters"]["classifier_specs"] == [
            ClassifierSpec.model_validate_json(spec_json)
        ]
        assert call_kwargs["parameters"]["document_ids"] == ["doc1", "doc2"]


def test_main_function_no_options():
    """Test the core function with minimal arguments"""
    mock_run = AsyncMock()
    mock_run.return_value.id = "test-id"

    with patch("scripts.infer.run_deployment", new=mock_run):
        main(
            aws_env=AwsEnv.staging,
            classifiers=[],
            documents=[],
        )

        mock_run.assert_called_once()
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["parameters"]["classifier_specs"] is None
        assert call_kwargs["parameters"]["document_ids"] is None
