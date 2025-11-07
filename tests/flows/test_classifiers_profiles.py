from unittest.mock import AsyncMock, Mock, patch

import pytest

from flows.classifier_specs.spec_interface import ClassifierSpec
from flows.classifiers_profiles import (
    compare_classifiers_profiles,
    convert_dict_to_classifier_spec,
    convert_dict_to_classifiers_profile_mapping,
    demote_classifier_profile,
    get_classifiers_profiles,
    promote_classifier_profile,
    update_classifier_profile,
    validate_artifact_metadata_rules,
    wandb_validation,
)
from flows.result import Err, Error, Ok
from knowledge_graph.classifiers_profiles import (
    ClassifiersProfileMapping,
    Profile,
)
from knowledge_graph.cloud import AwsEnv
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import ClassifierID, WikibaseID
from knowledge_graph.version import Version
from knowledge_graph.wikibase import StatementRank, WikibaseSession


@pytest.fixture
def mock_specs():
    return ClassifierSpec(
        wikibase_id="Q123",
        classifier_id="abcd3456",
        classifiers_profile="primary",
        wandb_registry_version="v20",
    )


@pytest.fixture
def mock_profile_mapping():
    return ClassifiersProfileMapping(
        wikibase_id="Q123",
        classifier_id="abcd3456",
        classifiers_profile="retired",
    )


@pytest.mark.asyncio
async def test_get_classifiers_profiles():
    # mock concepts
    list_concepts = [
        Concept(wikibase_id="Q123", preferred_label="Concept 123"),
        Concept(wikibase_id="Q100", preferred_label="Concept 100"),
        Concept(wikibase_id="Q999", preferred_label="Concept 999"),
        Concept(wikibase_id="Q200", preferred_label="Concept 200"),
    ]

    # mock response from wikibase.get_classifier_ids_async
    mock_wikibase = AsyncMock(spec=WikibaseSession)

    # 1 success, 3 failures
    mock_wikibase.get_classifier_ids_async.side_effect = [
        # Q123: success
        [
            (StatementRank.PREFERRED, ClassifierID("aaaa2222")),
            (StatementRank.DEPRECATED, ClassifierID("yyyy9999")),
        ],
        # Q100: failure
        Exception("Failed to fetch classifier profiles for Q100"),
        # Q999: fail validation - 2 primary profiles
        [
            (StatementRank.PREFERRED, ClassifierID("bbbb3333")),
            (StatementRank.PREFERRED, ClassifierID("cccc4444")),
        ],
        # Q200: fail validation - same classifier ID in 2 profiles
        [
            (StatementRank.PREFERRED, ClassifierID("xyzz2345")),
            (StatementRank.DEPRECATED, ClassifierID("xyzz2345")),
        ],
    ]

    # Call the function under test
    results = await get_classifiers_profiles(
        wikibase=mock_wikibase, concepts=list_concepts
    )

    # assert successful profiles
    classifier_profiles = [r._value for r in results if isinstance(r, Ok)]
    assert len(classifier_profiles) == 2
    assert classifier_profiles[0] == ClassifiersProfileMapping(
        wikibase_id=WikibaseID("Q123"),
        classifier_id=ClassifierID("aaaa2222"),
        classifiers_profile=Profile.PRIMARY,
    )
    assert classifier_profiles[1] == ClassifiersProfileMapping(
        wikibase_id=WikibaseID("Q123"),
        classifier_id=ClassifierID("yyyy9999"),
        classifiers_profile=Profile.RETIRED,
    )

    # assert validation errors
    failures = [r._error for r in results if isinstance(r, Err)]
    assert len(failures) == 3

    assert failures[0].metadata.get("wikibase_id") == "Q100"
    assert (
        failures[0].msg
        == "Error getting classifier ID from wikibase: Failed to fetch classifier profiles for Q100"
    )

    # check mocked method called
    assert mock_wikibase.get_classifier_ids_async.call_count == 4


def test_compare_classifiers_profiles():
    # Mock classifier specs (left dataframe)
    classifier_specs = [
        ClassifierSpec(
            wikibase_id="Q123",
            classifier_id="aaaa2222",
            classifiers_profile="primary",
            concept_id="aaaa2222",
            wandb_registry_version="v1",
        ),
        ClassifierSpec(
            wikibase_id="Q100",
            classifier_id="nnnn5555",
            classifiers_profile="experimental",
            concept_id="nnnn5555",
            wandb_registry_version="v1",
        ),
        ClassifierSpec(
            wikibase_id="Q1",
            classifier_id="abcd2345",
            classifiers_profile="primary",
            concept_id="abcd2345",
            wandb_registry_version="v1",
        ),
    ]

    # Mock classifiers profiles (right dataframe)
    classifiers_profiles = [
        ClassifiersProfileMapping(
            wikibase_id=WikibaseID("Q123"),
            classifier_id=ClassifierID("aaaa2222"),
            classifiers_profile="experimental",
        ),
        ClassifiersProfileMapping(
            wikibase_id=WikibaseID("Q100"),
            classifier_id=ClassifierID("nnnn5555"),
            classifiers_profile="experimental",
        ),
        ClassifiersProfileMapping(
            wikibase_id=WikibaseID("Q222"),
            classifier_id=ClassifierID("abab4444"),
            classifiers_profile="primary",
        ),
    ]

    results = compare_classifiers_profiles(classifier_specs, classifiers_profiles)

    assert (
        len([d for d in results if d.get("status") == "ignore"]) == 0
    )  # ignores are not returned
    assert len([d for d in results if d.get("status") == "add"]) == 1
    assert len([d for d in results if d.get("status") == "remove"]) == 1
    assert len([d for d in results if d.get("status") == "update"]) == 1


def test_promote_classifiers_profiles(mock_profile_mapping):
    """Test promoting classifiers profiles for successful validation."""

    # mock response to wandb_validation
    with (
        patch("flows.classifiers_profiles.wandb_validation") as mock_wandb_validation,
        patch("scripts.promote.main") as mock_promote,
    ):
        result = promote_classifier_profile(
            current_specs=None,
            new_specs=mock_profile_mapping,
            aws_env=AwsEnv.staging,
        )

        # Ensure wandb_validation was called with the correct arguments
        mock_wandb_validation.assert_called_once_with(
            wikibase_id=mock_profile_mapping.wikibase_id,
            classifier_id=mock_profile_mapping.classifier_id,
            wandb_registry_version=None,
            aws_env=AwsEnv.staging,
        )

        # Ensure scripts.promote.main was called with the correct arguments
        # TODO: Uncomment when promote script is added
        # mock_promote.assert_called_once_with(
        #     wikibase_id=mock_profile_mapping.wikibase_id,
        #     classifier_id=mock_profile_mapping.classifier_id,
        #     add_classifiers_profiles=[mock_profile_mapping.classifiers_profile],
        #     aws_env=AwsEnv.staging,
        # )
        mock_promote.assert_not_called()  # Remove when uncommented above

        assert isinstance(result, Ok)
        assert result._value.get("wikibase_id") == mock_profile_mapping.wikibase_id
        assert result._value.get("classifier_id") == mock_profile_mapping.classifier_id
        assert result._value.get("classifiers_profile") == [
            str(mock_profile_mapping.classifiers_profile)
        ]


def test_demote_classifiers_profiles(mock_specs):
    """Test demoting classifiers profiles for successful validation."""

    # mock response to wandb_validation
    with (
        patch("flows.classifiers_profiles.wandb_validation") as mock_wandb_validation,
        patch("scripts.demote.main") as mock_demote,
    ):
        result = demote_classifier_profile(
            current_specs=mock_specs,
            new_specs=None,
            aws_env=AwsEnv.staging,
        )

        # Ensure wandb_validation was called with the correct arguments
        mock_wandb_validation.assert_called_once_with(
            wikibase_id=mock_specs.wikibase_id,
            classifier_id=mock_specs.classifier_id,
            wandb_registry_version=mock_specs.wandb_registry_version,
            aws_env=AwsEnv.staging,
        )

        # Ensure scripts.demote.main was called with the correct arguments
        # TODO: Uncomment when demote script is added
        # mock_demote.assert_called_once_with(
        #     wikibase_id=mock_specs.wikibase_id,
        #     wandb_registry_version=mock_specs.wandb_registry_version,
        #     aws_env=AwsEnv.staging,
        # )
        mock_demote.assert_not_called()  # Remove when uncommented above

        assert isinstance(result, Ok)
        assert result._value.get("wikibase_id") == mock_specs.wikibase_id
        assert result._value.get("classifier_id") == mock_specs.classifier_id
        assert (
            result._value.get("wandb_registry_version")
            == mock_specs.wandb_registry_version
        )


def test_update_classifier_profile(mock_specs, mock_profile_mapping):
    """Test updating a classifier profile for successful validation."""

    # mock response to wandb_validation
    with (
        patch("flows.classifiers_profiles.wandb_validation") as mock_wandb_validation,
        patch("scripts.classifier_metadata.update") as mock_update,
    ):
        result = update_classifier_profile(
            current_specs=mock_specs,
            new_specs=mock_profile_mapping,
            aws_env=AwsEnv.staging,
        )

        # Ensure wandb_validation was called with the correct arguments
        mock_wandb_validation.assert_called_once_with(
            wikibase_id=mock_specs.wikibase_id,
            classifier_id=mock_specs.classifier_id,
            wandb_registry_version=None,
            aws_env=AwsEnv.staging,
        )

        # Ensure scripts.classifier_metadata.update was called with the correct arguments
        # TODO: Uncomment when update script is added
        # mock_update.assert_called_once_with(
        #     wikibase_id=mock_specs.wikibase_id,
        #     classifier_id=mock_specs.classifier_id,
        #     remove_classifiers_profiles=[mock_specs.classifiers_profile],
        #     add_classifiers_profiles=[mock_profile_mapping.classifiers_profile],
        #     aws_env=AwsEnv.staging,
        #     update_specs=False,
        # )
        mock_update.assert_not_called()  # Remove when uncommented above

        assert isinstance(result, Ok)
        assert result._value.get("wikibase_id") == mock_specs.wikibase_id
        assert result._value.get("classifier_id") == mock_specs.classifier_id
        assert result._value.get("remove_classifiers_profile") == [
            str(mock_specs.classifiers_profile)
        ]
        assert result._value.get("add_classifiers_profile") == [
            str(mock_profile_mapping.classifiers_profile)
        ]


def test_update_classifier_profile__failed_validation(mock_specs, mock_profile_mapping):
    """Test updating a classifier profile for failed validation."""

    # mock response to wandb_validation
    with (
        patch("flows.classifiers_profiles.wandb_validation") as mock_wandb_validation,
        patch("scripts.classifier_metadata.update") as mock_update,
    ):
        mock_wandb_validation.return_value = Err(
            Error(msg="Validation failed", metadata={})
        )

        result = update_classifier_profile(
            current_specs=mock_specs,
            new_specs=mock_profile_mapping,
            aws_env=AwsEnv.staging,
        )

        mock_update.assert_not_called()
        assert isinstance(result, Err)
        assert result._error.msg == "Validation failed"


@patch("wandb.Api")
def test_validate_artifact_exists__success_classifier_id(mock_api):
    """Test wandb_validation function for successful validation with classifier ID param."""
    mock_metadata = {"aws_env": "staging", "classifier_name": "ValidClassifier"}
    mock_artifacts = [
        Mock(version="v1", metadata=mock_metadata),
        Mock(version="v2", metadata=mock_metadata),
    ]

    mock_api.return_value.artifacts.return_value = mock_artifacts

    mock_artifact = Mock()
    mock_artifact.metadata = mock_metadata
    mock_artifact.logged_by.return_value.config = {}

    mock_api.return_value.artifact.return_value = mock_artifact

    result = wandb_validation(
        wikibase_id=WikibaseID("Q1"),
        classifier_id=ClassifierID("aaaa2222"),
        aws_env=AwsEnv.staging,
    )

    mock_api.return_value.artifact.assert_called_once()
    mock_api.return_value.artifacts.assert_called_once()

    assert isinstance(result, Ok)
    assert result._value == WikibaseID("Q1")


@patch("wandb.Api")
def test_wandb_validation__success_wandb_registry_version(mock_api):
    """Test wandb_validation function for successful validation with wandb registry version."""
    mock_metadata = {"aws_env": "staging", "classifier_name": "ValidClassifier"}
    mock_artifact = Mock()
    mock_artifact.metadata = mock_metadata
    mock_artifact.logged_by.return_value.config = {}

    mock_api.return_value.artifact.return_value = mock_artifact

    result = wandb_validation(
        wikibase_id=WikibaseID("Q1"),
        wandb_registry_version=Version("v2"),
        aws_env=AwsEnv.staging,
    )

    mock_api.return_value.artifact.assert_called_once()

    assert isinstance(result, Ok)
    assert result._value == WikibaseID("Q1")


@patch("wandb.Api")
def test_wandb_validation__failure_artifact_not_found(mock_api):
    """Test wandb_validation function for failure when artifact not found."""
    with (
        patch(
            "flows.classifiers_profiles.validate_artifact_metadata_rules"
        ) as mock_validate_metadata,
    ):
        # Mock the artifacts method to raise an exception
        mock_api.return_value.artifacts.side_effect = Exception("Artifact not found")

        result = wandb_validation(
            wikibase_id=WikibaseID("Q1"),
            classifier_id=ClassifierID("aaaa2222"),
            aws_env=AwsEnv.staging,
        )

        mock_api.return_value.artifact.assert_not_called()
        mock_api.return_value.artifacts.assert_called_once()
        mock_validate_metadata.assert_not_called()

        assert isinstance(result, Err)
        assert "Error retrieving artifact" in result._error.msg


def test_wandb_validation__failure_restricted_classifier():
    """Test wandb_validation when the classifier name is restricted"""

    mock_metadata = {
        "aws_env": "staging",
        "classifier_name": "LLMClassifier",
    }  # Restricted name

    mock_artifact = Mock()
    mock_artifact.metadata = mock_metadata
    mock_artifact.logged_by.return_value.config = {}

    result = validate_artifact_metadata_rules(artifact=mock_artifact)

    assert isinstance(result, Err)
    assert "artifact validation failed for classifier type" in result._error.msg


def test_wandb_validation__failure_restricted_run_config():
    """Test wandb_validation when the classifier name is restricted"""

    mock_metadata = {"aws_env": "staging", "classifier_name": "ValidClassifier"}

    mock_artifact = Mock()
    mock_artifact.metadata = mock_metadata
    mock_artifact.logged_by.return_value.config = {
        "experimental_concept": True
    }  # Restricted config

    result = validate_artifact_metadata_rules(artifact=mock_artifact)

    assert isinstance(result, Err)
    assert "artifact validation failed for run config" in result._error.msg


def test_convert_dict_to_classifier_spec(mock_specs):
    """Test converting a dictionary to a ClassifierSpec object."""
    # Test with required fields plus classifiers_profile
    input_data = {
        "wikibase_id": "Q123",
        "classifier_id": "abcd3456",
        "classifiers_profile": "primary",
        "wandb_registry_version": "v20",
    }

    result = convert_dict_to_classifier_spec(input_data)

    assert result.wikibase_id == mock_specs.wikibase_id
    assert result.classifier_id == mock_specs.classifier_id
    assert result.classifiers_profile == mock_specs.classifiers_profile
    assert result.wandb_registry_version == mock_specs.wandb_registry_version


def test_convert_dict_to_classifiers_profile_mapping():
    """Test converting a dictionary to a ClassifiersProfileMapping object."""
    # Test will more than required fields
    input_data = {
        "wikibase_id": "Q123",
        "classifier_id": "abcd3456",
        "classifiers_profile": "primary",
    }

    expected_mapping = ClassifiersProfileMapping(
        wikibase_id="Q123",
        classifier_id="abcd3456",
        classifiers_profile="primary",
    )

    result = convert_dict_to_classifiers_profile_mapping(input_data)

    assert result.wikibase_id == expected_mapping.wikibase_id
    assert result.classifier_id == expected_mapping.classifier_id
    assert result.classifiers_profile == expected_mapping.classifiers_profile
