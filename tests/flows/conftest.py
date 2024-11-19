import json
import os
import subprocess
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Generator
from unittest.mock import Mock, patch

import boto3
import pytest
from cpr_sdk.models.search import Concept as VespaConcept
from cpr_sdk.parser_models import (
    BaseParserOutput,
    BlockType,
    HTMLData,
    HTMLTextBlock,
    PDFData,
    PDFTextBlock,
)
from cpr_sdk.search_adaptors import VespaSearchAdapter
from moto import mock_aws
from pydantic import SecretStr
from requests.exceptions import ConnectionError
from vespa.application import Vespa
from wandb.apis.public import ArtifactType

from flows.index import get_vespa_search_adapter_from_aws_secrets
from flows.inference import Config
from scripts.cloud import AwsEnv
from src.identifiers import WikibaseID
from src.labelled_passage import LabelledPassage

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture()
def test_config():
    yield Config(
        cache_bucket="test_bucket",
        wandb_model_registry="test_org/test_wandb_model_registry",
        wandb_entity="test_entity",
        wandb_api_key=SecretStr("test_wandb_api_key"),
        aws_env=AwsEnv("sandbox"),
    )


@pytest.fixture(scope="function")
def mock_aws_creds():
    """Mocked AWS Credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = "test"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "test"
    os.environ["AWS_SECURITY_TOKEN"] = "test"
    os.environ["AWS_SESSION_TOKEN"] = "test"


@pytest.fixture
def mock_s3_client(mock_aws_creds) -> Generator:
    with mock_aws():
        yield boto3.client("s3")


@pytest.fixture(scope="function")
def mock_ssm_client(mock_aws_creds) -> Generator:
    """Mocked boto3 ssm client."""
    with mock_aws():
        yield boto3.client("ssm", region_name="eu-west-1")


@pytest.fixture()
def mock_vespa_credentials() -> dict[str, str]:
    """Mocked vespa credentials."""
    return {
        "VESPA_INSTANCE_URL": "http://localhost:8080",
        "VESPA_PUBLIC_CERT_FULL_ACCESS": "UHVibGljIGNlcnQgY29udGVudAo=",  # "Public cert content"
        "VESPA_PRIVATE_KEY_FULL_ACCESS": "UHJpdmF0ZSBrZXkgY29udGVudAo=",  # "Private key content"
    }


@pytest.fixture
def create_vespa_params(mock_ssm_client, mock_vespa_credentials) -> None:
    """Creates the vespa parameters in the mock ssm client."""
    mock_ssm_client.put_parameter(
        Name="VESPA_INSTANCE_URL",
        Description="A test parameter for the vespa instance.",
        Value=mock_vespa_credentials["VESPA_INSTANCE_URL"],
        Type="SecureString",
    )
    mock_ssm_client.put_parameter(
        Name="VESPA_PUBLIC_CERT_FULL_ACCESS",
        Description="A test parameter for a vespa public cert",
        Value=mock_vespa_credentials["VESPA_PUBLIC_CERT_FULL_ACCESS"],
        Type="SecureString",
    )
    mock_ssm_client.put_parameter(
        Name="VESPA_PRIVATE_KEY_FULL_ACCESS",
        Description="A test parameter for a vespa private key",
        Value=mock_vespa_credentials["VESPA_PRIVATE_KEY_FULL_ACCESS"],
        Type="SecureString",
    )


@pytest.fixture(scope="function")
def vespa_app(
    mock_vespa_credentials,
):
    # Connection
    print("\nSetting up Vespa connection...")
    app = Vespa(mock_vespa_credentials["VESPA_INSTANCE_URL"])

    subprocess.run(["just", "vespa_feed_data"], capture_output=True, text=True)

    yield app  # This is where the test function will be executed

    # Teardown
    print("\nTearing down Vespa connection...")
    delete_all_documents(app)


def delete_all_documents(app):
    print("Deleting all documents...")
    response = app.delete_all_docs(
        content_cluster_name="family-document-passage", schema="search_weights"
    )
    print(f"Delete response: {response}")
    response = app.delete_all_docs(
        content_cluster_name="family-document-passage", schema="family_document"
    )
    print(f"Delete response: {response}")
    response = app.delete_all_docs(
        content_cluster_name="family-document-passage", schema="document_passage"
    )
    print(f"Delete response: {response}")


@pytest.fixture
def local_vespa_search_adapter(
    create_vespa_params, mock_vespa_credentials, tmp_path
) -> VespaSearchAdapter:
    """VespaSearchAdapter instantiated from mocked ssm params."""
    adapter = get_vespa_search_adapter_from_aws_secrets(
        cert_dir=str(tmp_path),
        vespa_public_cert_param_name="VESPA_PUBLIC_CERT_FULL_ACCESS",
        vespa_private_key_param_name="VESPA_PRIVATE_KEY_FULL_ACCESS",
    )
    try:
        adapter.client.get_application_status()
    except ConnectionError:
        pytest.fail(
            "Can't connect to a local vespa instance. See guidance here: "
            "`tests/local_vespa/README.md`"
        )

    yield adapter


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


def create_mock_new_and_updated_documents_json(
    mock_s3_client, mock_bucket, doc_names: tuple[str, str], timestamp: str
):
    first_doc, second_doc = doc_names
    content = {
        "new_documents": [
            {"import_id": first_doc},
        ],
        "updated_documents": {second_doc: []},
    }
    data = BytesIO(json.dumps(content).encode("utf-8"))
    key = os.path.join("input", timestamp, "new_and_updated_documents.json")
    mock_s3_client.put_object(
        Bucket=mock_bucket, Key=key, Body=data, ContentType="application/json"
    )


@pytest.fixture
def mock_bucket_new_and_updated_documents_json(mock_s3_client, mock_bucket):
    previous_docs = {"Previous.document.0.2", "Previous.document.0.1"}
    create_mock_new_and_updated_documents_json(
        mock_s3_client,
        mock_bucket,
        doc_names=previous_docs,
        timestamp="2023-01-1T01.01.01.000001",
    )

    latest_docs = {"Latest.document.0.2", "Latest.document.0.1"}
    create_mock_new_and_updated_documents_json(
        mock_s3_client,
        mock_bucket,
        doc_names=latest_docs,
        timestamp="2023-01-1T01.01.01.000001",
    )
    yield previous_docs, latest_docs


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
def s3_prefix_mock_bucket(
    mock_bucket: str,
) -> str:
    return f"s3://{mock_bucket}"
    """Returns the s3 prefix for the concepts."""


@pytest.fixture
def s3_prefix_mock_bucket_labelled_passages(
    mock_bucket: str,
    s3_prefix_labelled_passages: str,
) -> str:
    return f"s3://{mock_bucket}/{s3_prefix_labelled_passages}"
    """Returns the s3 prefix for the concepts."""


@pytest.fixture
def s3_prefix_labelled_passages() -> str:
    """Returns the s3 prefix for the concepts."""
    return "labelled_passages/Q788/latest"


@pytest.fixture()
def labelled_passage_fixture_ids() -> list[str]:
    """Returns the list of concept fixture files."""
    return [
        "CCLW.executive.10014.4470",
        "CCLW.executive.4934.1571",
    ]


@pytest.fixture()
def labelled_passage_fixture_files(labelled_passage_fixture_ids) -> list[str]:
    """Returns the list of concept fixture files."""
    return [f"{doc_id}.json" for doc_id in labelled_passage_fixture_ids]


@pytest.fixture
def mock_bucket_labelled_passages(
    mock_s3_client,
    mock_bucket,
    s3_prefix_labelled_passages,
    labelled_passage_fixture_files,
) -> None:
    """Puts the concept fixture files in the mock bucket."""
    for file_name in labelled_passage_fixture_files:
        data = load_fixture(file_name)
        body = BytesIO(data.encode("utf-8"))
        key = os.path.join(s3_prefix_labelled_passages, file_name)
        mock_s3_client.put_object(
            Bucket=mock_bucket, Key=key, Body=body, ContentType="application/json"
        )


@pytest.fixture
def document_passages_test_data_file_path() -> str:
    """Returns the path to the document passages test data file."""
    return "tests/local_vespa/test_documents/document_passage.json"


@pytest.fixture
def example_vespa_concepts() -> list[VespaConcept]:
    """Vespa concepts for testing."""
    return [
        VespaConcept(
            id="1457",
            name="wood industry",
            parent_concepts=[
                {"name": "forestry sector", "id": "Q788"},
                {"name": "lumber", "id": "Q789"},
            ],
            parent_concept_ids_flat="Q788,Q789",
            model='KeywordClassifier("wood industry")',
            end=100,
            start=0,
            timestamp=datetime.now(),
        ),
        VespaConcept(
            id="1273",
            name="manufacturing sector",
            parent_concepts=[
                {"name": "manufacturing", "id": "Q2"},
                {"name": "processing industry", "id": "Q3"},
            ],
            parent_concept_ids_flat="Q2,Q3",
            model="KeywordClassifier('manufacturing sector')",
            end=100,
            start=0,
            timestamp=datetime.now(),
        ),
    ]


@pytest.fixture
def example_labelled_passages(labelled_passage_fixture_files) -> list[LabelledPassage]:
    """Returns a list of example labelled passages."""
    labelled_passages = []
    for file_name in labelled_passage_fixture_files:
        data = json.loads(load_fixture(file_name))
        labelled_passages.extend([LabelledPassage.model_validate(i) for i in data])
    return labelled_passages


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


@pytest.fixture
def mock_wandb_api():
    with (
        patch("wandb.Api") as mock_api,
        patch("wandb.apis.public.ArtifactType") as mock_artifact_type,
    ):
        # Create a mock for the API instance
        api_instance = Mock()
        mock_api.return_value = api_instance

        # Create mock model collections
        collections = []
        for model_data in [
            {"name": "Q111", "env": "sandbox"},
            {"name": "Q222", "env": "sandbox"},
            {"name": "Q444", "env": "labs"},
            {"name": "some_other_model", "env": "sandbox"},
        ]:
            mock_artifact = Mock()
            mock_artifact.version = "v1"
            mock_artifact.metadata = {"aws_env": model_data["env"]}

            mock_collection = Mock()
            mock_collection.name = model_data["name"]
            mock_collection.artifacts.return_value = [mock_artifact]
            collections.append(mock_collection)

        mock_type_instance = Mock()
        mock_type_instance.collections.return_value = collections

        mock_artifact_type.return_value = mock_type_instance
        with (
            patch.object(ArtifactType, "load", return_value="mocked"),
            patch.object(ArtifactType, "collections", return_value=collections),
        ):
            yield mock_api
