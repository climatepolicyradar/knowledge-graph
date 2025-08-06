import textwrap
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import pytest

from flows.classifier_specs.spec_interface import ClassifierSpec
from scripts.cloud import AwsEnv
from scripts.update_classifier_spec import (
    get_all_available_classifiers,
    sort_specs,
)
from src.identifiers import WikibaseID


@pytest.fixture
def mock_wandb_api():
    with patch("wandb.Api") as mock_api:
        # Create a mock for the API instance
        api_instance = Mock()
        mock_api.return_value = api_instance

        # Create mock classifier artifacts
        mock_artifacts = []
        for model_data in [
            {"name": "Q111:v1", "env": "sandbox", "id": "abcd2345"},
            {"name": "Q444:v2", "env": "labs", "id": "efgh6789"},
            {"name": "Q222:v1", "env": "sandbox", "id": "2345abcd"},
        ]:
            mock_artifact = Mock()
            mock_artifact.name = model_data["name"]
            mock_artifact.metadata = {"aws_env": model_data["env"]}
            mock_artifact.json_encode.return_value = {"sequenceName": model_data["id"]}
            mock_artifacts.append(mock_artifact)

        # api.registries().collections().versions() = artifacts
        mock_registries = Mock()
        api_instance.registries.return_value = mock_registries

        mock_collections = Mock()
        mock_registries.collections.return_value = mock_collections

        mock_collections.versions.return_value = mock_artifacts

        yield mock_api


def test_get_all_available_classifiers(mock_wandb_api):
    with TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        with patch("flows.classifier_specs.spec_interface.SPEC_DIR", temp_dir):
            get_all_available_classifiers(aws_envs=[AwsEnv.sandbox])
            output_path = temp_dir / "sandbox.yaml"

            with open(output_path, "r") as file:
                results = file.read()

            expected = textwrap.dedent("""
                ---
                - classifier_id: abcd2345
                  wandb_registry_version: 1
                  wikibase_id: Q111
                - classifier_id: 2345abcd
                  wandb_registry_version: 1
                  wikibase_id: Q222
                """).lstrip()

            assert results == expected


def test_sort_specs():
    unsorted_specs = [
        ClassifierSpec(
            wikibase_id=WikibaseID("Q444"),
            classifier_id="abcd2345",
            wandb_registry_version="v1",
        ),
        ClassifierSpec(
            wikibase_id=WikibaseID("Q111"),
            classifier_id="abcd2345",
            wandb_registry_version="v1",
        ),
        ClassifierSpec(
            wikibase_id=WikibaseID("Q555"),
            classifier_id="efgh2345",
            wandb_registry_version="v1",
        ),
        ClassifierSpec(
            wikibase_id=WikibaseID("Q333"),
            classifier_id="abcd2345",
            wandb_registry_version="v1",
        ),
        ClassifierSpec(
            wikibase_id=WikibaseID("Q222"),
            classifier_id="abcd2345",
            wandb_registry_version="v1",
        ),
        ClassifierSpec(
            wikibase_id=WikibaseID("Q555"),
            classifier_id="abcd2345",
            wandb_registry_version="v1",
        ),
    ]

    assert sort_specs(unsorted_specs) == [
        ClassifierSpec(
            wikibase_id=WikibaseID("Q111"),
            classifier_id="abcd2345",
            wandb_registry_version="v1",
        ),
        ClassifierSpec(
            wikibase_id=WikibaseID("Q222"),
            classifier_id="abcd2345",
            wandb_registry_version="v1",
        ),
        ClassifierSpec(
            wikibase_id=WikibaseID("Q333"),
            classifier_id="abcd2345",
            wandb_registry_version="v1",
        ),
        ClassifierSpec(
            wikibase_id=WikibaseID("Q444"),
            classifier_id="abcd2345",
            wandb_registry_version="v1",
        ),
        ClassifierSpec(
            wikibase_id=WikibaseID("Q555"),
            classifier_id="abcd2345",
            wandb_registry_version="v1",
        ),
        ClassifierSpec(
            wikibase_id=WikibaseID("Q555"),
            classifier_id="efgh2345",
            wandb_registry_version="v1",
        ),
    ]
