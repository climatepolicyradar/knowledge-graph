import pytest
import typer
from typer.testing import CliRunner

from scripts.infer import app, convert_classifier_specs

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


@pytest.mark.asyncio
async def test_main_command_basic():
    result = runner.invoke(app, ["--aws-env", "staging", "-c", "Q123"])

    assert result.exit_code == 0


@pytest.mark.asyncio
async def test_main_command_with_documents():
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


@pytest.mark.asyncio
async def test_main_command_no_options():
    result = runner.invoke(app, ["--aws-env", "staging"])

    assert result.exit_code == 0
