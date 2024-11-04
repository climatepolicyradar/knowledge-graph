import os
from io import BytesIO
from pathlib import Path
from typing import Any, Generator
from unittest.mock import Mock, patch

import boto3
import pytest
from cpr_sdk.parser_models import (
    BaseParserOutput,
    BlockType,
    HTMLData,
    HTMLTextBlock,
    PDFData,
    PDFTextBlock,
)
from moto import mock_aws

from flows.inference import Config
from src.identifiers import WikibaseID

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture()
def test_config():
    yield Config(
        cache_bucket="test_bucket",
        wandb_model_registry="test_wandb_model_registry",
        wandb_entity="test_entity",
        wandb_api_key="test_wandb_api_key",
    )


@pytest.fixture(scope="function")
def mock_aws_creds():
    """Mocked AWS Credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = "test"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "test"
    os.environ["AWS_SECURITY_TOKEN"] = "test"
    os.environ["AWS_SESSION_TOKEN"] = "test"


@pytest.fixture
def mock_s3_client(mock_aws_creds) -> Generator[str, Any, Any]:
    with mock_aws():
        yield boto3.client("s3")


@pytest.fixture
def mock_bucket(
    mock_aws_creds, mock_s3_client, test_config
) -> Generator[str, Any, Any]:
    mock_s3_client.create_bucket(
        Bucket=test_config.cache_bucket,
        CreateBucketConfiguration={"LocationConstraint": "eu-west-1"},
    )
    yield test_config.cache_bucket


def load_fixture(file_name):
    fixture_path = FIXTURE_DIR / file_name
    with open(fixture_path) as f:
        return f.read()


@pytest.fixture
def mock_bucket_documents(mock_s3_client, mock_bucket):
    fixture_files = ["PDF.document.0.1.json", "HTML.document.0.1.json"]
    for file_name in fixture_files:
        data = load_fixture(file_name)
        body = BytesIO(data.encode("utf-8"))
        key = os.path.join("embeddings_input", file_name)
        mock_s3_client.put_object(
            Bucket=mock_bucket, Key=key, Body=body, ContentType="application/json"
        )
    yield fixture_files


@pytest.fixture
def mock_classifiers_dir(test_config):
    mock_dir = Path(FIXTURE_DIR) / "classifiers"
    with patch.object(test_config, "local_classifier_dir", new=mock_dir):
        yield mock_dir


@pytest.fixture
def local_classifier_id(mock_classifiers_dir):
    classifier_id = WikibaseID("Q788")
    full_path = mock_classifiers_dir / classifier_id
    assert full_path.exists()
    yield classifier_id


@pytest.fixture
def parser_output():
    yield BaseParserOutput(
        document_id="test id",
        document_metadata={},
        document_name="test name",
        document_slug="test slug",
        document_description="test description",
    )


@pytest.fixture
def parser_output_html(parser_output):
    parser_output.document_content_type = "text/html"
    parser_output.html_data = HTMLData(
        has_valid_text=True,
        text_blocks=[
            HTMLTextBlock(
                text=["test html text"],
                text_block_id="1",
            )
        ],
    )
    yield parser_output


@pytest.fixture
def parser_output_pdf(parser_output):
    # When the content type is pdf
    parser_output.document_content_type = "application/pdf"
    parser_output.html_data = None
    parser_output.pdf_data = PDFData(
        page_metadata=[],
        md5sum="",
        text_blocks=[
            PDFTextBlock(
                text=["test pdf text"],
                text_block_id="2",
                page_number=1,
                coords=[],
                type=BlockType.TEXT,
                type_confidence=0.5,
            )
        ],
    )
    yield parser_output


@pytest.fixture
def mock_wandb(mock_s3_client):
    with (
        patch("wandb.init") as mock_init,
        patch("wandb.login"),
    ):
        mock_run = Mock()
        mock_artifact = Mock()
        mock_init.return_value = mock_run
        mock_run.use_artifact.return_value = mock_artifact

        yield mock_init, mock_run, mock_artifact
