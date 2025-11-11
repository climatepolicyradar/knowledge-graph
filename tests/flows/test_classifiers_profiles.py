from unittest.mock import AsyncMock, Mock, patch

import pytest

from flows.classifier_specs.spec_interface import ClassifierSpec
from flows.classifiers_profiles import (
    compare_classifiers_profiles,
    demote_classifier_profile,
    get_classifiers_profiles,
    handle_classifier_profile_action,
    promote_classifier_profile,
    update_classifier_profile,
    validate_artifact_metadata_rules,
    wandb_validation,
)
from flows.result import Err, Error, Ok, unwrap_err, unwrap_ok
from knowledge_graph.classifiers_profiles import (
    ClassifiersProfileMapping,
    Profile,
)
from knowledge_graph.cloud import AwsEnv
from knowledge_graph.compare_result_operation import Demote, Ignore, Promote, Update
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import ClassifierID, WikibaseID
from knowledge_graph.version import Version
from knowledge_graph.wikibase import StatementRank


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

    with patch("flows.classifiers_profiles.WikibaseSession") as mock_wikibase_session:
        # mock response from wikibase.get_classifier_ids_async
        mock_wikibase_auth = Mock()
        mock_wikibase = mock_wikibase_session.return_value

        # 1 success, 3 failures
        mock_wikibase.get_classifier_ids_async = AsyncMock(
            side_effect=[
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
        )

        # Call the function under test
        results = await get_classifiers_profiles(
            wikibase_auth=mock_wikibase_auth, concepts=list_concepts
        )

    assert mock_wikibase.get_classifier_ids_async.call_count == 4

    # assert successful profiles
    classifier_profiles = [unwrap_ok(r) for r in results if isinstance(r, Ok)]
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
    failures = [unwrap_err(r) for r in results if isinstance(r, Err)]
    assert len(failures) == 3

    assert failures[0].metadata.get("wikibase_id") == "Q100"
    assert (
        failures[0].msg
        == "Error getting classifier ID from wikibase: Failed to fetch classifier profiles for Q100"
    )

    # check mocked method called
    assert mock_wikibase.get_classifier_ids_async.call_count == 4


def test_compare_classifiers_profiles():
    # Mock classifier specs
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
    ]

    # add spec to be identified to remove
    mock_spec_remove = ClassifierSpec(
        wikibase_id="Q1",
        classifier_id="abcd2345",
        classifiers_profile="primary",
        concept_id="abcd2345",
        wandb_registry_version="v1",
    )
    classifier_specs.append(mock_spec_remove)

    # Mock classifiers profiles mappings
    classifiers_profile_mapping_ignore = [
        ClassifiersProfileMapping(
            wikibase_id=WikibaseID("Q123"),
            classifier_id=ClassifierID("aaaa2222"),
            classifiers_profile=Profile.PRIMARY,
        )
    ]

    classifiers_profile_mapping_add = [
        ClassifiersProfileMapping(
            wikibase_id=WikibaseID("Q222"),
            classifier_id=ClassifierID("abab4444"),
            classifiers_profile=Profile.PRIMARY,
        )
    ]

    classifiers_profile_mapping_update = [
        ClassifiersProfileMapping(
            wikibase_id=WikibaseID("Q100"),
            classifier_id=ClassifierID("nnnn5555"),
            classifiers_profile=Profile.RETIRED,
        )
    ]

    classifiers_profile_mappings = (
        classifiers_profile_mapping_ignore
        + classifiers_profile_mapping_add
        + classifiers_profile_mapping_update
    )

    results = compare_classifiers_profiles(
        classifier_specs, classifiers_profile_mappings
    )

    assert (
        len([d for d in results if isinstance(d, Ignore)]) == 0
    )  # ignores are not returned, otherwise should be 1
    assert len([d for d in results if isinstance(d, Promote)]) == 1
    assert len([d for d in results if isinstance(d, Demote)]) == 1
    assert len([d for d in results if isinstance(d, Update)]) == 1

    # check Promote
    assert [d for d in results if isinstance(d, Promote)][
        0
    ].classifiers_profile_mapping == classifiers_profile_mapping_add[0]
    # check Demote
    assert [d for d in results if isinstance(d, Demote)][
        0
    ].classifier_spec == mock_spec_remove
    # check Update
    assert [d for d in results if isinstance(d, Update)][
        0
    ].classifiers_profile_mapping == classifiers_profile_mapping_update[0]


def test_handle_classifier_profile_action(mock_profile_mapping):
    """Test handling different classifier profile actions from action_function"""

    with patch("flows.classifiers_profiles.wandb_validation") as mock_wandb_validation:
        mock_wandb_validation.return_value = Ok(mock_profile_mapping.wikibase_id)

        def mock_action_function(wikibase_id, aws_env, **kwargs):
            print(
                f"Action Function Called: wikibase_id={wikibase_id}, aws_env={aws_env}, kwargs={kwargs}"
            )

        result = handle_classifier_profile_action(
            action="test_action",
            wikibase_id=mock_profile_mapping.wikibase_id,
            aws_env=AwsEnv.staging,
            action_function=mock_action_function,
            classifier_id=mock_profile_mapping.classifier_id,
            classifiers_profile=mock_profile_mapping.classifiers_profile,
        )

        # Ensure wandb_validation was called with the correct arguments
        mock_wandb_validation.assert_called_once_with(
            wikibase_id=mock_profile_mapping.wikibase_id,
            classifier_id=mock_profile_mapping.classifier_id,
            wandb_registry_version=None,
            aws_env=AwsEnv.staging,
        )

        assert isinstance(result, Ok)
        assert result._value.get("wikibase_id") == mock_profile_mapping.wikibase_id
        assert result._value.get("classifier_id") == mock_profile_mapping.classifier_id
        assert (
            result._value.get("classifiers_profile")
            == mock_profile_mapping.classifiers_profile
        )


def test_promote_classifiers_profiles(mock_profile_mapping):
    """Test promoting classifiers profiles for successful validation."""

    # mock response to wandb_validation
    with (
        patch("scripts.promote.main") as mock_promote,
    ):
        promote_classifier_profile(
            wikibase_id=mock_profile_mapping.wikibase_id,
            classifier_id=mock_profile_mapping.classifier_id,
            classifiers_profile=mock_profile_mapping.classifiers_profile,
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


def test_demote_classifiers_profiles(mock_specs):
    """Test demoting classifiers profiles for successful validation."""

    # mock response to wandb_validation
    with (
        patch("scripts.demote.main") as mock_demote,
    ):
        demote_classifier_profile(
            wikibase_id=mock_specs.wikibase_id,
            wandb_registry_version=mock_specs.wandb_registry_version,
            aws_env=AwsEnv.staging,
            classifier_id=mock_specs.classifier_id,
        )

        # Ensure scripts.demote.main was called with the correct arguments
        # TODO: Uncomment when demote script is added
        # mock_demote.assert_called_once_with(
        #     wikibase_id=mock_specs.wikibase_id,
        #     wandb_registry_version=mock_specs.wandb_registry_version,
        #     aws_env=AwsEnv.staging,
        # )
        mock_demote.assert_not_called()  # Remove when uncommented above


def test_update_classifier_profile(mock_specs, mock_profile_mapping):
    """Test updating a classifier profile for successful validation."""

    # mock response to wandb_validation
    with (
        patch("scripts.classifier_metadata.update") as mock_update,
    ):
        update_classifier_profile(
            wikibase_id=mock_specs.wikibase_id,
            classifier_id=mock_specs.classifier_id,
            aws_env=AwsEnv.staging,
            add_classifiers_profiles=[mock_profile_mapping.classifiers_profile],
            remove_classifiers_profiles=[mock_specs.classifiers_profile],
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


def test_handle_classifier_profile_action__failed_validation(mock_specs):
    """Test updating a classifier profile for failed validation."""

    # mock response to wandb_validation
    with (
        patch("flows.classifiers_profiles.wandb_validation") as mock_wandb_validation,
    ):
        mock_wandb_validation.return_value = Err(
            Error(msg="Validation failed", metadata={})
        )

        mock_action_function = Mock()

        result = handle_classifier_profile_action(
            action="test_action",
            wikibase_id=mock_specs.wikibase_id,
            aws_env=AwsEnv.staging,
            action_function=mock_action_function,
            classifier_id=mock_specs.classifier_id,
        )

        mock_action_function.assert_not_called()
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
