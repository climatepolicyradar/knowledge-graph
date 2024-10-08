import os
from io import BytesIO
from pathlib import Path
from typing import Any, Generator
from unittest.mock import patch

import boto3
import pytest
from moto import mock_aws

from flows.inference import config
from src.identifiers import WikibaseID

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


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
def mock_bucket(mock_aws_creds, mock_s3_client) -> Generator[str, Any, Any]:
    test_bucket_name = "test_bucket"
    with patch.object(config, "cache_bucket", new=test_bucket_name):
        mock_s3_client.create_bucket(
            Bucket=test_bucket_name,
            CreateBucketConfiguration={"LocationConstraint": "eu-west-1"},
        )
        yield test_bucket_name


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
def mock_classifiers_dir():
    mock_dir = Path(FIXTURE_DIR) / "classifiers"
    with patch.object(config, "local_classifier_dir", new=mock_dir):
        yield mock_dir


@pytest.fixture
def local_classifier_id(mock_classifiers_dir):
    classifier_id = WikibaseID("Q788")
    full_path = mock_classifiers_dir / classifier_id
    assert full_path.exists()
    yield classifier_id
