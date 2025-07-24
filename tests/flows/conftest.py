import asyncio
import json
import os
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import AsyncGenerator, Generator
from contextlib import asynccontextmanager
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import aioboto3
import boto3
import pytest
import pytest_asyncio
from botocore.config import Config as BotoCoreConfig
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
from prefect import Flow, State
from prefect_aws.s3 import S3Bucket
from pydantic import SecretStr
from requests.exceptions import ConnectionError
from types_aiobotocore_s3.client import S3Client
from vespa.application import Vespa
from vespa.io import VespaQueryResponse

from flows.aggregate import Config as AggregateInferenceResultsConfig
from flows.inference import S3_BLOCK_RESULTS_CACHE
from flows.inference import Config as InferenceConfig
from flows.utils import DocumentStem
from flows.wikibase_to_s3 import Config as WikibaseToS3Config
from scripts.cloud import AwsEnv
from src.concept import Concept
from src.identifiers import WikibaseID
from src.labelled_passage import LabelledPassage

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture
def test_config():
    yield InferenceConfig(
        cache_bucket="test_bucket",
        wandb_model_registry="test_org/test_wandb_model_registry",
        wandb_entity="test_entity",
        wandb_api_key=SecretStr("test_wandb_api_key"),
        aws_env=AwsEnv("sandbox"),
    )


@pytest.fixture
def test_wikibase_to_s3_config():
    yield WikibaseToS3Config(
        cdn_bucket_name="test_bucket",
        aws_env=AwsEnv("sandbox"),
        wikibase_password=SecretStr("test_password"),
        wikibase_username="test_username",
        wikibase_url="https://test.test.test",
    )


@pytest.fixture
def test_aggregate_config():
    yield AggregateInferenceResultsConfig(
        cache_bucket="test_bucket",
        aws_env=AwsEnv.sandbox,
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


@pytest_asyncio.fixture
async def mock_s3_async_client(
    mock_aws_creds, moto_patch_session
) -> AsyncGenerator[S3Client, None]:
    with mock_aws():
        session = aioboto3.Session(region_name="eu-west-1")
        config = BotoCoreConfig(
            read_timeout=60, connect_timeout=60, retries={"max_attempts": 3}
        )
        async with session.client("s3", config=config) as client:
            yield client


@pytest.fixture(scope="function")
def mock_ssm_client(mock_aws_creds) -> Generator:
    """Mocked boto3 ssm client."""
    with mock_aws():
        yield boto3.client("ssm", region_name="eu-west-1")


@pytest.fixture
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

    subprocess.run(
        ["just", "vespa_feed_data"],
        capture_output=True,
        text=True,
        check=True,
        timeout=600,  # Seconds
    )

    yield app  # This is where the test function will be executed

    # Teardown
    print("\nTearing down Vespa connection...")
    subprocess.run(
        ["just", "vespa_delete_data"],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,  # Seconds
    )


@pytest.fixture(scope="function")
def vespa_large_app(
    mock_vespa_credentials,
):
    # Connection
    print("\nSetting up Vespa connection...")
    app = Vespa(mock_vespa_credentials["VESPA_INSTANCE_URL"])

    subprocess.run(
        ["just", "vespa_feed_large_data"],
        capture_output=True,
        text=True,
        check=True,
        timeout=600,  # Seconds
    )

    yield app  # This is where the test function will be executed

    # Teardown
    print("\nTearing down Vespa connection...")
    subprocess.run(
        ["just", "vespa_delete_data"],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,  # Seconds
    )


@pytest.fixture
def local_vespa_search_adapter(
    create_vespa_params, mock_vespa_credentials, tmp_path
) -> Generator[VespaSearchAdapter, None, None]:
    """VespaSearchAdapter instantiated from mocked SSM params."""
    instance_url = "http://localhost:8080"
    adapter = VespaSearchAdapter(
        instance_url=instance_url,
    )

    # We can't currently optionally use certs with our search adapter.
    #
    # Instead, overwrite it here.
    adapter.client = Vespa(url=instance_url, cert=None)

    try:
        adapter.client.get_application_status()
    except ConnectionError:
        pytest.fail(
            "Can't connect to a local vespa instance. See guidance here: "
            "`tests/local_vespa/README.md`"
        )

    yield adapter


@pytest_asyncio.fixture
async def mock_async_bucket(
    mock_aws_creds, mock_s3_async_client, test_config
) -> AsyncGenerator[tuple[str, S3Client], None]:
    await mock_s3_async_client.create_bucket(
        Bucket=test_config.cache_bucket,
        CreateBucketConfiguration={"LocationConstraint": "eu-west-1"},
    )
    yield test_config.cache_bucket, mock_s3_async_client

    # Teardown
    try:
        response = await mock_s3_async_client.list_objects_v2(
            Bucket=test_config.cache_bucket
        )
        for obj in response.get("Contents", []):
            try:
                await mock_s3_async_client.delete_object(
                    Bucket=test_config.cache_bucket, Key=obj["Key"]
                )
            except Exception as e:
                print(
                    f"Warning: Failed to delete object {obj['Key']} during teardown: {e}"
                )

        await mock_s3_async_client.delete_bucket(Bucket=test_config.cache_bucket)
    except Exception as e:
        print(
            f"Warning: Failed to clean up bucket {test_config.cache_bucket} during teardown: {e}"
        )


@pytest.fixture
def mock_bucket(
    mock_aws_creds, mock_s3_client, test_config
) -> Generator[str, Any, Any]:
    mock_s3_client.create_bucket(
        Bucket=test_config.cache_bucket,
        CreateBucketConfiguration={"LocationConstraint": "eu-west-1"},
    )
    yield test_config.cache_bucket


@pytest.fixture
def mock_cdn_bucket(
    mock_aws_creds, mock_s3_client, test_wikibase_to_s3_config
) -> Generator[str, Any, Any]:
    mock_s3_client.create_bucket(
        Bucket=test_wikibase_to_s3_config.cdn_bucket_name,
        CreateBucketConfiguration={"LocationConstraint": "eu-west-1"},
    )
    yield test_wikibase_to_s3_config.cdn_bucket_name


@pytest.fixture
def mock_bucket_b(
    mock_aws_creds, mock_s3_client, test_config
) -> Generator[str, Any, Any]:
    bucket = test_config.cache_bucket + "b"
    mock_s3_client.create_bucket(
        Bucket=bucket,
        CreateBucketConfiguration={"LocationConstraint": "eu-west-1"},
    )
    yield bucket


def load_fixture(file_name) -> str:
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
def mock_bucket_documents_b(mock_s3_client, mock_bucket_b):
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
    """Returns the s3 prefix for the concepts."""
    return f"s3://{mock_bucket}"


@pytest.fixture
def s3_prefix_mock_bucket_labelled_passages(
    mock_bucket: str,
    s3_prefix_labelled_passages: str,
) -> str:
    """Returns the s3 prefix for the concepts."""
    return f"s3://{mock_bucket}/{s3_prefix_labelled_passages}"


@pytest.fixture
def s3_prefix_labelled_passages() -> str:
    """Returns the s3 prefix for the concepts."""
    return "labelled_passages/Q788/v4"


@pytest.fixture
def labelled_passage_fixture_ids() -> list[str]:
    """Returns the list of concept fixture files."""

    return [
        "CCLW.executive.10014.4470",
        "CCLW.executive.4934.1571",
    ]


@pytest.fixture
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
def aggregate_inference_results_document_stems() -> list[DocumentStem]:
    """Returns the list of aggregate inference results file stems."""

    return [
        DocumentStem("CCLW.executive.4934.1571"),
        DocumentStem("CCLW.executive.10014.4470_translated_en"),
    ]


@pytest.fixture
def mock_run_output_identifier_str() -> str:
    """Returns the identifier for the run output."""

    return "2025-05-25T07:32-eta85-alchibah"


@pytest.fixture
def s3_prefix_inference_results(mock_run_output_identifier_str: str) -> str:
    """Returns the s3 prefix for the inference results."""

    return f"inference_results/{mock_run_output_identifier_str}/"


@pytest.fixture
def mock_bucket_inference_results(
    mock_s3_client,
    mock_bucket,
    s3_prefix_inference_results: str,
    aggregate_inference_results_document_stems: list[DocumentStem],
) -> dict[str, dict[str, Any]]:
    """A version of the inference results bucket with more files"""

    fixture_root = FIXTURE_DIR / "inference_results"
    fixture_files = [
        fixture_root / f"{document_stem}.json"
        for document_stem in aggregate_inference_results_document_stems
    ]

    inference_results = {}
    for file_path in fixture_files:
        with open(file_path) as f:
            data = f.read()
        body = BytesIO(data.encode("utf-8"))

        key = s3_prefix_inference_results + str(file_path.relative_to(fixture_root))
        inference_results[key] = json.loads(data)

        mock_s3_client.put_object(
            Bucket=mock_bucket, Key=key, Body=body, ContentType="application/json"
        )

    return inference_results


@pytest.fixture
def mock_bucket_labelled_passages_b(
    mock_s3_client,
    mock_bucket_b,
    s3_prefix_labelled_passages,
    labelled_passage_fixture_files,
) -> None:
    """Puts the concept fixture files in the mock bucket."""
    for file_name in labelled_passage_fixture_files:
        data = load_fixture(file_name)
        body = BytesIO(data.encode("utf-8"))
        key = os.path.join(s3_prefix_labelled_passages, file_name)
        mock_s3_client.put_object(
            Bucket=mock_bucket_b, Key=key, Body=body, ContentType="application/json"
        )


@pytest_asyncio.fixture
async def mock_bucket_labelled_passages_large(
    mock_async_bucket,
) -> tuple[list[str], str, S3Client]:
    """A version of the labelled_passage bucket with more files"""
    bucket, mock_s3_async_client = mock_async_bucket
    fixture_root = FIXTURE_DIR / "labelled_passages"
    fixture_files = list(fixture_root.glob("**/*.json"))

    keys = []
    for file_path in fixture_files:
        with open(file_path) as f:
            data = f.read()
        body = BytesIO(data.encode("utf-8"))

        key = "labelled_passages/" + str(file_path.relative_to(fixture_root))
        keys.append(key)

        await mock_s3_async_client.put_object(
            Bucket=bucket, Key=key, Body=body, ContentType="application/json"
        )

    return (keys, bucket, mock_s3_async_client)


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
            parent_concepts=None,
            parent_concept_ids_flat=None,
            model='KeywordClassifier("wood industry")',
            end=100,
            start=0,
            timestamp=datetime.now(),
        ),
        VespaConcept(
            id="1273",
            name="manufacturing sector",
            parent_concepts=[
                {"name": "manufacturing", "id": "Q200"},
                {"name": "processing industry", "id": "Q300"},
            ],
            parent_concept_ids_flat="Q200,Q300",
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
def example_labelled_passages_1_doc(
    labelled_passage_fixture_files,
) -> tuple[str, list[LabelledPassage]]:
    """Returns a list of example labelled passages."""
    file_name = labelled_passage_fixture_files[0]
    data = json.loads(load_fixture(file_name))
    labelled_passages = [LabelledPassage.model_validate(i) for i in data]
    return file_name, labelled_passages


@pytest.fixture
def mock_wandb(mock_s3_client):
    with (
        patch("wandb.init") as mock_init,
        patch("wandb.login"),
    ):
        mock_artifact = Mock()

        class StubbedRun:
            def use_artifact(self, *args, **kwargs):
                return mock_artifact

        mock_run = StubbedRun()
        mock_init.return_value = mock_run
        yield mock_init, mock_run, mock_artifact


@pytest.fixture
def mock_concepts() -> Generator[list[Concept], None, None]:
    yield [
        Concept(
            wikibase_id=WikibaseID("Q10"),
            preferred_label="marine toxins",
        ),
        Concept(
            wikibase_id=WikibaseID("Q20"),
            preferred_label="biome shift",
        ),
        Concept(
            wikibase_id=WikibaseID("Q30"),
            preferred_label="short-lived climate pollutant",
        ),
    ]


@pytest.fixture
def mock_prefect_slack_webhook():
    """Patch the SlackWebhook class to return a mock object."""
    with patch("flows.utils.SlackWebhook") as mock_SlackWebhook:
        mock_prefect_slack_block = MagicMock()
        mock_SlackWebhook.load.return_value = mock_prefect_slack_block
        yield mock_SlackWebhook, mock_prefect_slack_block


@pytest.fixture
def mock_flow():
    """Mock Prefect flow object."""
    mock_flow = MagicMock(spec=Flow)
    mock_flow.name = "TestFlow"
    yield mock_flow


@pytest.fixture
def mock_flow_run():
    """Mock Prefect flow run object."""
    mock_flow_run = MagicMock()
    mock_flow_run.name = "TestFlowRun"
    mock_flow_run.id = "test-flow-run-id"
    mock_flow_run.state = MagicMock(spec=State)
    mock_flow_run.state.name = "Completed"
    mock_flow_run.state.message = "message"
    mock_flow_run.state.timestamp = "2025-01-28T12:00:00+00:00"

    yield mock_flow_run


@pytest.fixture
def mock_concepts_counts_document_keys() -> list[str]:
    """Paths for all concepts_counts fixtures."""
    keys = []
    for path in FIXTURE_DIR.rglob("concepts_counts/**/*.json"):
        keys.append(str(path.relative_to(FIXTURE_DIR)))
    return keys


@pytest.fixture
def mock_bucket_concepts_counts(
    mock_concepts_counts_document_keys,
    mock_s3_client,
    mock_bucket,
) -> None:
    """Puts the concept counts fixture files in the mock bucket."""
    for key in mock_concepts_counts_document_keys:
        data = load_fixture(key)
        body = BytesIO(data.encode("utf-8"))
        mock_s3_client.put_object(
            Bucket=mock_bucket, Key=key, Body=body, ContentType="application/json"
        )


def mock_grouped_text_block_vespa_query_response_json() -> dict:
    """Mock Vespa query response JSON"""

    with open(
        "tests/flows/fixtures/query_responses/grouped_text_block_by_family_document_ref.json"
    ) as f:
        data = json.load(f)

    return data


@pytest.fixture
def mock_vespa_query_response(mock_vespa_credentials: dict) -> VespaQueryResponse:
    """Mock Vespa query response"""

    return VespaQueryResponse(
        json=mock_grouped_text_block_vespa_query_response_json(),
        status_code=200,
        url=mock_vespa_credentials["VESPA_INSTANCE_URL"],
        request_body={},
    )


@pytest.fixture
def mock_vespa_query_response_with_malformed_group(
    mock_vespa_credentials: dict,
) -> VespaQueryResponse:
    """Mock Vespa query response with a malformed group"""

    group_with_malformed_hit = {
        "id": "group:string:986",
        "relevance": 0.0017429193899782135,
        "value": "986",
        "children": [
            {
                "id": "hitlist:hits",
                "relevance": 1.0,
                "label": "hits",
                "children": [
                    {
                        "id": "id:doc_search:document_passage::CCLW.executive.10014.4470.986",
                        "relevance": 0.0017429193899782135,
                        "source": "family-document-passage",
                        "fields_malformed": {},
                    }
                ],
            }
        ],
    }

    response_with_malformed_group_json = (
        mock_grouped_text_block_vespa_query_response_json()
    )

    response_with_malformed_group_json["root"]["children"][0]["children"][0][
        "children"
    ] += [group_with_malformed_hit]

    vespa_query_response_with_malformed_group = VespaQueryResponse(
        json=response_with_malformed_group_json,
        status_code=200,
        url=mock_vespa_credentials["VESPA_INSTANCE_URL"],
        request_body={},
    )

    return vespa_query_response_with_malformed_group


@pytest.fixture
def mock_vespa_query_response_no_continuation_token(
    mock_vespa_credentials: dict,
) -> VespaQueryResponse:
    """Mock Vespa query response with no hits"""

    json_data = mock_grouped_text_block_vespa_query_response_json()
    json_data["root"]["children"][0]["children"][0]["continuation"] = {
        "prev": "BGAAABEBCBC"
    }

    return VespaQueryResponse(
        json=json_data,
        status_code=200,
        url=mock_vespa_credentials["VESPA_INSTANCE_URL"],
        request_body={},
    )


@pytest.fixture
def vespa_lower_max_hit_limit_query_profile_name() -> str:
    """The name of the query profile to use for the lower max hits limit."""
    return "lower_max_hits"


@pytest.fixture
def vespa_lower_max_hit_limit(vespa_lower_max_hit_limit_query_profile_name: str) -> int:
    """Mock Vespa max hit limit"""

    tree = ET.parse(
        "tests/local_vespa/additional_query_profiles/"
        f"{vespa_lower_max_hit_limit_query_profile_name}.xml"
    )
    root = tree.getroot()

    lower_max_hits_limit = None
    for field in root.findall("field"):
        name = field.get("name")
        if name == "maxHits":
            lower_max_hits_limit = field.text
            break

    if not lower_max_hits_limit:
        raise ValueError("Lower max hits limit not found in XML file.")
    return int(lower_max_hits_limit)


@asynccontextmanager
async def s3_block_context():
    """Context manager for creating and cleaning up S3 blocks."""
    bucket_name = S3_BLOCK_RESULTS_CACHE.replace("s3-bucket/", "")

    test_block = S3Bucket(bucket_name=bucket_name)

    try:
        uuid = test_block.save(bucket_name, overwrite=True)
        if asyncio.iscoroutine(uuid):
            uuid = await uuid
        yield uuid
    except Exception as e:
        print(f"Warning: Failed to save S3 block: {e}")

    finally:
        try:
            result = test_block.delete(bucket_name)
            if asyncio.iscoroutine(result):
                await result
        except Exception as e:
            print(f"Warning: Failed to delete S3 block: {e}")


@pytest_asyncio.fixture
async def mock_prefect_s3_block():
    """Create an S3 block against the local prefect server."""
    async with s3_block_context() as block:
        yield block
