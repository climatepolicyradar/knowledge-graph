import asyncio
import json
import os
import subprocess
import traceback
import uuid
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch
from urllib.parse import parse_qs

import boto3
import httpx
import pandas as pd
import pytest
from moto import mock_aws

from knowledge_graph.classifier.classifier import Classifier
from knowledge_graph.concept import Concept
from knowledge_graph.config import get_git_root
from knowledge_graph.wikibase import WikibaseSession


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
    def get_mock_response(wikibase_id: str = "Q10") -> dict:
        return {
            "entities": {
                wikibase_id: {
                    "pageid": 14,
                    "ns": 120,
                    "title": f"Item:{wikibase_id}",
                    "lastrevid": 3785,
                    "modified": "2024-06-12T10:09:34Z",
                    "type": "item",
                    "id": wikibase_id,
                }
            },
            "success": 1,
        }

    return get_mock_response


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
    def generate_concept_data(wikibase_id: str) -> dict:
        return {
            "type": "item",
            "id": wikibase_id,
            "labels": {
                "en": {
                    "language": "en",
                    "value": f"concept {wikibase_id}",
                }
            },
            "descriptions": {
                "en": {
                    "language": "en",
                    "value": f"description for {wikibase_id}",
                }
            },
            "aliases": {
                "en": [
                    {
                        "language": "en",
                        "value": f"alias for {wikibase_id}",
                    }
                ]
            },
            "claims": {
                "P1": [
                    {
                        "mainsnak": {
                            "snaktype": "value",
                            "property": "P1",
                            "datavalue": {
                                "value": {
                                    "entity-type": "item",
                                    "id": "Q20",
                                },
                                "type": "wikibase-entityid",
                            },
                        }
                    }
                ],
                "P2": [
                    {
                        "mainsnak": {
                            "snaktype": "value",
                            "property": "P2",
                            "datavalue": {
                                "value": {
                                    "entity-type": "item",
                                    "id": "Q5",
                                },
                                "type": "wikibase-entityid",
                            },
                        }
                    }
                ],
                "P7": [
                    {
                        "mainsnak": {
                            "snaktype": "value",
                            "property": "P7",
                            "datavalue": {
                                "value": f"Definition for {wikibase_id}",
                                "type": "string",
                            },
                        }
                    }
                ],
            },
        }

    def get_mock_response(wikibase_id: str) -> dict:
        return {
            "continue": {"rvcontinue": "20240612100926|3784", "continue": "||"},
            "query": {
                "pages": {
                    "14": {
                        "pageid": 14,
                        "ns": 120,
                        "title": f"Item:{wikibase_id}",
                        "revisions": [
                            {
                                "revid": 12345,
                                "slots": {
                                    "main": {
                                        "contentmodel": "wikibase-item",
                                        "contentformat": "application/json",
                                        "*": json.dumps(
                                            generate_concept_data(wikibase_id)
                                        ),
                                    }
                                },
                            }
                        ],
                    }
                }
            },
        }

    return get_mock_response


@pytest.fixture
def mock_wikibase_redirects_json():
    return {
        "batchcomplete": "",
        "query": {
            "allpages": [
                {"pageid": 1234, "ns": 120, "title": "Item:Q15", "redirect": True}
            ]
        },
    }


@pytest.fixture
def mock_wikibase_redirect_target_json():
    return {
        "entities": {
            "Q15": {
                "pageid": 1234,
                "ns": 120,
                "title": "Item:Q15",
                "redirects": {"from": "Q15", "to": "Q10"},
            }
        }
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
    mock_wikibase_redirects_json,
    mock_wikibase_redirect_target_json,
):
    def mock_request_handler(request):
        if not httpx.URL(mock_wikibase_url).host == request.url.host:
            raise MockedWikibaseException(f"Non-test endpoint used: {request.url}")

        if request.url.path == "/query/sparql":
            query = request.url.params.get("query", "")
            if "select ?entity where" in query.lower():
                sparql_response = {
                    "head": {"vars": ["entity"]},
                    "results": {"bindings": []},
                }
                return httpx.Response(200, json=sparql_response)
            else:
                raise MockedWikibaseException(
                    f"Expected SPARQL query format not used: {query}"
                )

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
                    wikibase_id = request.url.params.get("ids")
                    return httpx.Response(
                        200, json=mock_wikibase_revisions_json(wikibase_id)
                    )

                if request.url.params.get("list") == "allpages":
                    apnamespace = request.url.params.get("apnamespace")
                    apfilterredir = request.url.params.get("apfilterredir")
                    # get_properties
                    if apnamespace == "122":
                        return httpx.Response(200, json=mock_wikibase_properties_json)
                    # get_redirects
                    if apfilterredir == "redirects":
                        return httpx.Response(200, json=mock_wikibase_redirects_json)
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
                # Handle redirects
                if request.url.params.get("ids") == "Q15":
                    return httpx.Response(200, json=mock_wikibase_redirect_target_json)

                ids_param = request.url.params.get("ids")
                if "|" in ids_param:
                    # Handle batch requests - split the IDs and create response for each
                    ids = ids_param.split("|")
                    entities_response = {"entities": {}, "success": 1}
                    for wikibase_id in ids:
                        single_response = mock_wikibase_entities_json(wikibase_id)
                        entities_response["entities"].update(
                            single_response["entities"]
                        )
                    return httpx.Response(200, json=entities_response)
                else:
                    # Handle single ID request
                    wikibase_id = ids_param
                    return httpx.Response(
                        200, json=mock_wikibase_entities_json(wikibase_id)
                    )
            # get_statements
            case "wbgetclaims":
                return httpx.Response(200, json={"claims": {}})
            # add_statement
            case "wbcreateclaim":
                return httpx.Response(200, json={})

        raise MockedWikibaseException(f"Unhandled test endpoint: {request.url}")

    mock_transport = httpx.MockTransport(mock_request_handler)

    # Patch WikibaseSession to use the mocked transport
    original_get_client = WikibaseSession._get_client

    async def mocked_get_client(self):
        if self._client is None:
            self._client = httpx.AsyncClient(transport=mock_transport, timeout=30)
            # Skip login and redirects for tests - set them directly
            self._csrf_token = "test_csrf_token"
            self._redirects = {}

        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.MAX_CONCURRENT_REQUESTS)

        return self._client

    WikibaseSession._get_client = mocked_get_client

    try:
        yield WikibaseSession
    except MockedWikibaseException:
        exc_info = traceback.format_exc()
        pytest.fail(f"Wikibase test failed because of an exception:\n {exc_info}")
    finally:
        # Restore original method
        WikibaseSession._get_client = original_get_client


@pytest.fixture
def mock_argilla_client():
    """
    Mock an Argilla client with a patched Argilla class.

    Returns a tuple of (mock_client, mock_argilla_class) for cases where you need to
    verify the class was called correctly.
    """
    with patch("knowledge_graph.labelling.Argilla") as mock_argilla_class:
        mock_client = MagicMock()
        mock_argilla_class.return_value = mock_client
        yield mock_client, mock_argilla_class


@pytest.fixture
def mock_workspace():
    """Factory fixture for creating mock Argilla workspaces"""

    def _create_workspace(name="test-workspace", datasets=None):
        mock_ws = MagicMock()
        mock_ws.name = name
        mock_ws.datasets = datasets or []
        return mock_ws

    return _create_workspace


@pytest.fixture
def mock_dataset():
    """Factory fixture for creating mock Argilla datasets"""

    def _create_dataset(name="Q123", records=None):
        mock_ds = MagicMock()
        mock_ds.name = name
        if records is not None:
            mock_ds.records.return_value = records
        return mock_ds

    return _create_dataset


@pytest.fixture
def mock_user():
    """Factory fixture for creating mock Argilla users"""

    def _create_user(username="test-user", user_id=None):
        mock_u = MagicMock()
        mock_u.username = username
        mock_u.id = user_id or uuid.uuid4()
        return mock_u

    return _create_user


@pytest.fixture
def concept_without_a_wikibase_id():
    """Test concept without wikibase_id for error testing"""
    return Concept(
        preferred_label="Test concept",
        description="A concept without a wikibase ID",
    )
