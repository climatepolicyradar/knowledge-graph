import json
import os
import subprocess
import traceback
from pathlib import Path
from urllib.parse import parse_qs

import httpx
import pandas as pd
import pytest
from prefect.logging import disable_run_logger
from prefect.testing.utilities import prefect_test_harness

from scripts.config import get_git_root
from src.classifier import Classifier
from src.concept import Concept
from src.wikibase import WikibaseSession


@pytest.fixture(autouse=True, scope="session")
def prefect_test_fixture():
    with prefect_test_harness():
        with disable_run_logger():
            yield


@pytest.fixture(scope="function")
def aws_credentials():
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"


@pytest.fixture
def metrics_df() -> pd.DataFrame:
    fixture_path = Path(
        "tests/fixtures/data/processed/classifiers_performance/Q787.json"
    )
    with open(fixture_path) as f:
        data = json.load(f)

    return pd.DataFrame(data)


@pytest.fixture
def concept_wikibase_id() -> str:
    return "Q787"


@pytest.fixture
def concept() -> Concept:
    fixture_path = Path("tests/fixtures/data/processed/concepts/Q787.json")
    return Concept.load(fixture_path)


@pytest.fixture
def classifier() -> Classifier:
    fixture_path = Path("tests/fixtures/data/processed/classifiers/Q787")
    return Classifier.load(fixture_path)


@pytest.fixture
def run_just_command():
    repo_root = get_git_root()
    if repo_root is None:
        raise ValueError("Could not find the root of the git repository")

    def _run_command(command_name):
        # Split the command into parts to handle arguments correctly
        command_parts = command_name.split()
        result = subprocess.run(
            ["just"] + command_parts, capture_output=True, text=True, cwd=repo_root
        )
        return result

    return _run_command


@pytest.fixture
def mock_wikibase_url():
    # See pyproject.toml for where this is set
    return "https://test.test.test"


@pytest.fixture
def mock_wikibase_token_json():
    return {
        "query": {
            "tokens": {
                "logintoken": "token",
                "csrftoken": "token",
            }
        }
    }


@pytest.fixture
def mock_wikibase_properties_json():
    return {
        "batchcomplete": "",
        "limits": {"allpages": 5000},
        "query": {
            "allpages": [
                {"pageid": 1, "ns": 122, "title": "Property:P1"},
                {"pageid": 1596, "ns": 122, "title": "Property:P10"},
            ]
        },
    }


@pytest.fixture
def mock_wikibase_entities_json():
    return {
        "entities": {
            "Q10": {
                "pageid": 14,
                "ns": 120,
                "title": "Item:Q10",
                "lastrevid": 3785,
                "modified": "2024-06-12T10:09:34Z",
                "type": "item",
                "id": "Q10",
            }
        },
        "success": 1,
    }


@pytest.fixture
def mock_wikibase_items_first_page_json():
    return {
        "batchcomplete": "",
        "continue": {"apcontinue": "Q1003", "continue": "-||"},
        "query": {
            "allpages": [
                {"pageid": 14, "ns": 120, "title": "Item:Q10"},
                {"pageid": 104, "ns": 120, "title": "Item:Q100"},
                {"pageid": 1022, "ns": 120, "title": "Item:Q1000"},
                {"pageid": 1023, "ns": 120, "title": "Item:Q1001"},
                {"pageid": 1024, "ns": 120, "title": "Item:Q1002"},
            ]
        },
    }


@pytest.fixture
def mock_wikibase_items_last_page_json():
    return {
        "batchcomplete": "",
        "query": {
            "allpages": [
                {"pageid": 1031, "ns": 120, "title": "Item:Q1003"},
                {"pageid": 1026, "ns": 120, "title": "Item:Q1004"},
                {"pageid": 1028, "ns": 120, "title": "Item:Q1005"},
                {"pageid": 1029, "ns": 120, "title": "Item:Q1006"},
                {"pageid": 1030, "ns": 120, "title": "Item:Q1007"},
            ]
        },
    }


@pytest.fixture
def mock_wikibase_revisions_json():
    return {
        "continue": {"rvcontinue": "20240612100926|3784", "continue": "||"},
        "query": {
            "pages": {
                "14": {
                    "pageid": 14,
                    "ns": 120,
                    "title": "Item:Q10",
                    "revisions": [
                        {
                            "slots": {
                                "main": {
                                    "contentmodel": "wikibase-item",
                                    "contentformat": "application/json",
                                    "*": '{"type":"item","id":"Q10","labels":{"en":{"language":"en","value":"tropical forests"}},"descriptions":{"en":{"language":"en","value":"generic forest in the tropics"}},"aliases":{"en":[{"language":"en","value":"tropical forests"}]},"claims":{"P2":[{"mainsnak":{"snaktype":"value","property":"P2","hash":"44c11d9ab91c8c3cf203033599eebcb8d1be97ce","datavalue":{"value":{"entity-type":"item","numeric-id":5,"id":"Q5"},"type":"wikibase-entityid"}},"type":"statement","id":"Q10$DB660D9E-8D6C-4ED6-8060-D51161A88382","rank":"normal"}]},"sitelinks":[]}',
                                }
                            }
                        }
                    ],
                }
            }
        },
    }


class MockedWikibaseException(Exception):
    """
    Exception raised by the mocked Wikibase API.

    Means that `MockedWikibaseSession` is not configured properly for what is
    being tested. There might be some misconfiguration or a missing scenario.
    """

    pass


@pytest.fixture
def MockedWikibaseSession(
    mock_wikibase_url,
    mock_wikibase_token_json,
    mock_wikibase_properties_json,
    mock_wikibase_items_first_page_json,
    mock_wikibase_items_last_page_json,
    mock_wikibase_revisions_json,
    mock_wikibase_entities_json,
):
    def mock_request_handler(request):
        if not httpx.URL(mock_wikibase_url).host == request.url.host:
            raise MockedWikibaseException(f"Non-test endpoint used: {request.url}")

        if not request.url.path == "/w/api.php":
            raise MockedWikibaseException(f"Expected Api path not used: {request.url}")

        # Define request scenarios to catch here:
        if request.url.params:
            action = request.url.params.get("action")
        else:
            action = parse_qs(request.content.decode())["action"][0]
        match action:
            case "query":
                # _login
                if request.url.params.get("meta") == "tokens":
                    return httpx.Response(200, json=mock_wikibase_token_json)
                # get_concept
                if request.url.params.get("prop") == "revisions":
                    return httpx.Response(200, json=mock_wikibase_revisions_json)

                if request.url.params.get("list") == "allpages":
                    apnamespace = request.url.params.get("apnamespace")
                    # get_properties
                    if apnamespace == "122":
                        return httpx.Response(200, json=mock_wikibase_properties_json)
                    # get_concepts
                    if apnamespace == "120":
                        # Requests get continue details from the response,
                        # so the first request won't have continue params
                        if "continue" not in request.url.params:
                            return httpx.Response(
                                200, json=mock_wikibase_items_first_page_json
                            )
                        else:
                            return httpx.Response(
                                200, json=mock_wikibase_items_last_page_json
                            )
            #  _login
            case "login":
                return httpx.Response(200)
            # get_concept
            case "wbgetentities":
                return httpx.Response(200, json=mock_wikibase_entities_json)
            # get_statements
            case "wbgetclaims":
                return httpx.Response(200, json={"claims": {}})
            # add_statement
            case "wbcreateclaim":
                return httpx.Response(200, json={})

        raise MockedWikibaseException(f"Unhandled test endpoint: {request.url}")

    mock_transport = httpx.MockTransport(mock_request_handler)
    with httpx.Client(transport=mock_transport) as client:
        WikibaseSession.session = client
        try:
            yield WikibaseSession
        except MockedWikibaseException:
            exc_info = traceback.format_exc()
            pytest.fail(f"Wikibase test failed because of an exception:\n {exc_info}")
