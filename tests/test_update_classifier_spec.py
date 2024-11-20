from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import pytest

from scripts.update_classifier_spec import (
    get_all_available_classifiers,
    is_concept_model,
    read_spec_file,
)
from wandb.apis.public import ArtifactType
from wandb.apis.public.artifacts import ArtifactCollection


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
            mock_artifact.name = f"{model_data['name']}:v1"
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


def test_is_concept_model():
    # Make the artifact collection mockable
    def mocked_artifact_init(self, name):
        self.name = name

    ArtifactCollection.__init__ = mocked_artifact_init

    # Some other model
    not_concept_model = ArtifactCollection(name="other_model")
    assert not is_concept_model(not_concept_model)

    # Concept model
    concept_model = ArtifactCollection(name="Q123456")
    assert is_concept_model(concept_model)


def test_get_all_available_classifiers(mock_wandb_api):
    with TemporaryDirectory() as temp_dir:
        with patch("scripts.update_classifier_spec.SPEC_DIR", Path(temp_dir)):
            get_all_available_classifiers(aws_envs=["sandbox"])
            specs = read_spec_file("sandbox")
            assert specs == ["Q111:v1", "Q222:v1"]
