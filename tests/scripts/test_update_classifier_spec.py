from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import pytest

from scripts.cloud import AwsEnv, ClassifierSpec
from scripts.update_classifier_spec import (
    get_all_available_classifiers,
    is_concept_model,
    is_latest_model_in_env,
    parse_spec_file,
    read_spec_file,
    sort_specs,
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
            {"name": "Q444", "env": "labs"},
            {"name": "some_other_model", "env": "sandbox"},
            {"name": "Q222", "env": "sandbox"},
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


def test_is_latest_model_in_env():
    classifier_specs = [
        ClassifierSpec(name="Q22", alias="v1"),
        ClassifierSpec(name="Q11", alias="v2"),
    ]
    assert not is_latest_model_in_env(classifier_specs, model_name="Q11")
    assert is_latest_model_in_env(classifier_specs, model_name="Q33")


def test_get_all_available_classifiers(mock_wandb_api):
    with TemporaryDirectory() as temp_dir:
        with patch("scripts.update_classifier_spec.SPEC_DIR", Path(temp_dir)):
            get_all_available_classifiers(
                aws_envs=[AwsEnv.sandbox],
                api_key="test_wandb_api_key",
            )
            specs = read_spec_file(AwsEnv.sandbox)
            assert specs == ["Q111:v1", "Q222:v1"]


@pytest.mark.parametrize(
    "spec_contents,expected_specs",
    [
        # Test valid single entry
        (["Q123:v1"], [ClassifierSpec(name="Q123", alias="v1")]),
        # Test valid multiple entries
        (
            ["Q123:v1", "Q456:v2"],
            [
                ClassifierSpec(name="Q123", alias="v1"),
                ClassifierSpec(name="Q456", alias="v2"),
            ],
        ),
        # Test empty list
        ([], []),
    ],
)
def test_parse_spec_file(spec_contents, expected_specs, tmp_path):
    # Create a temporary spec file
    test_env = AwsEnv.sandbox
    spec_dir = tmp_path / "classifier_specs"
    spec_dir.mkdir(parents=True)
    spec_file = spec_dir / f"{test_env}.yaml"

    # Write test contents
    import yaml

    with open(spec_file, "w") as f:
        yaml.dump(spec_contents, f)

    # Patch the SPEC_DIR to use our temporary directory
    with patch("scripts.update_classifier_spec.SPEC_DIR", spec_dir):
        result = parse_spec_file(test_env)
        assert result == expected_specs


@pytest.mark.parametrize(
    "invalid_contents",
    [
        ["invalid_format"],  # Missing colon separator
        ["Q123:v1:extra"],  # Too many separators
        ["Q123"],  # No version
        [":v1"],  # No name
        ["Q123:"],  # No version after separator
    ],
)
def test_parse_spec_file_invalid_format(invalid_contents, tmp_path):
    # Create a temporary spec file
    test_env = AwsEnv.sandbox
    spec_dir = tmp_path / "classifier_specs"
    spec_dir.mkdir(parents=True)
    spec_file = spec_dir / f"{test_env}.yaml"

    # Write test contents
    import yaml

    with open(spec_file, "w") as f:
        yaml.dump(invalid_contents, f)

    # Patch the SPEC_DIR to use our temporary directory
    with patch("scripts.update_classifier_spec.SPEC_DIR", spec_dir):
        with pytest.raises(ValueError):
            parse_spec_file(test_env)


def test_sort_specs():
    unsorted_specs = [
        ClassifierSpec(name="Q123", alias="v4"),
        ClassifierSpec(name="Q789", alias="v1"),
        ClassifierSpec(name="Q456", alias="v30"),
        ClassifierSpec(name="Q999", alias="v3"),
        ClassifierSpec(name="Q111", alias="v3"),
    ]

    assert sort_specs(unsorted_specs) == [
        ClassifierSpec(name="Q111", alias="v3"),
        ClassifierSpec(name="Q123", alias="v4"),
        ClassifierSpec(name="Q456", alias="v30"),
        ClassifierSpec(name="Q789", alias="v1"),
        ClassifierSpec(name="Q999", alias="v3"),
    ]
