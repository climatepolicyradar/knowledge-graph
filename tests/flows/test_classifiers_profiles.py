from unittest.mock import AsyncMock, Mock, patch

import pytest

# from cpr_sdk.models.search import ClassifiersProfiles as VespaClassifiersProfiles
from cpr_sdk.models.search import ClassifiersProfile as VespaClassifiersProfile
from cpr_sdk.models.search import WikibaseId as VespaWikibaseId

from flows.classifier_specs.spec_interface import ClassifierSpec
from flows.classifiers_profiles import (
    compare_classifiers_profiles,
    create_vespa_classifiers_profile,
    create_vespa_profile_mappings,
    demote_classifier_profile,
    emit_finished,
    get_classifiers_profiles,
    handle_classifier_profile_action,
    promote_classifier_profile,
    update_classifier_profile,
    update_vespa_with_classifiers_profiles,
    validate_artifact_metadata_rules,
    wandb_validation,
)
from flows.result import Err, Error, Ok, is_ok, unwrap_err, unwrap_ok
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


@pytest.fixture
def mock_specs_2profiles():
    return [
        ClassifierSpec(
            wikibase_id="Q123",
            classifier_id="aaaa2222",
            classifiers_profile="primary",
            wandb_registry_version="v12",
            concept_id="abcd2345",
        ),
        ClassifierSpec(
            wikibase_id="Q100",
            classifier_id="nnnn5555",
            classifiers_profile="experimental",
            wandb_registry_version="v2",
            concept_id="efgh5678",
        ),
    ]


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
            upload_to_wandb=False,
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

    # mock response to wandb scripts
    with (
        patch("scripts.promote.main") as mock_promote,
    ):
        promote_classifier_profile(
            wikibase_id=mock_profile_mapping.wikibase_id,
            classifier_id=mock_profile_mapping.classifier_id,
            classifiers_profile=mock_profile_mapping.classifiers_profile,
            aws_env=AwsEnv.staging,
            upload_to_wandb=True,
        )

        # Ensure scripts.promote.main was called with the correct arguments
        mock_promote.assert_called_once_with(
            wikibase_id=mock_profile_mapping.wikibase_id,
            classifier_id=mock_profile_mapping.classifier_id,
            aws_env=AwsEnv.staging,
            add_classifiers_profiles=[mock_profile_mapping.classifiers_profile.value],
        )


def test_promote_classifiers_profiles__dry_run(mock_profile_mapping):
    """Test promoting classifiers profiles for successful validation."""

    # mock response to wandb scripts
    with (
        patch("scripts.promote.main") as mock_promote,
    ):
        promote_classifier_profile(
            wikibase_id=mock_profile_mapping.wikibase_id,
            classifier_id=mock_profile_mapping.classifier_id,
            classifiers_profile=mock_profile_mapping.classifiers_profile,
            aws_env=AwsEnv.staging,
            upload_to_wandb=False,
        )

        # Ensure scripts.demote.main was not called when dry run
        mock_promote.assert_not_called()


def test_demote_classifiers_profiles(mock_specs):
    """Test demoting classifiers profiles for successful validation."""

    # mock response to wandb scripts
    with (
        patch("scripts.demote.main") as mock_demote,
    ):
        demote_classifier_profile(
            wikibase_id=mock_specs.wikibase_id,
            wandb_registry_version=mock_specs.wandb_registry_version,
            aws_env=AwsEnv.staging,
            classifier_id=mock_specs.classifier_id,
            upload_to_wandb=True,
        )

        # Ensure scripts.demote.main was called with the correct arguments
        mock_demote.assert_called_once_with(
            wikibase_id=mock_specs.wikibase_id,
            wandb_registry_version=mock_specs.wandb_registry_version,
            aws_env=AwsEnv.staging,
        )


def test_demote_classifiers_profiles__dry_run(mock_specs):
    """Test demoting classifiers profiles for successful validation."""

    # mock response to wandb scripts
    with (
        patch("scripts.demote.main") as mock_demote,
    ):
        demote_classifier_profile(
            wikibase_id=mock_specs.wikibase_id,
            wandb_registry_version=mock_specs.wandb_registry_version,
            aws_env=AwsEnv.staging,
            classifier_id=mock_specs.classifier_id,
            upload_to_wandb=False,
        )

        # Ensure scripts.demote.main was not called when dry run
        mock_demote.assert_not_called()


def test_update_classifier_profile(mock_specs, mock_profile_mapping):
    """Test updating a classifier profile for successful validation."""

    # mock response to wandb scripts
    with (
        patch("scripts.classifier_metadata.update") as mock_update,
    ):
        update_classifier_profile(
            wikibase_id=mock_specs.wikibase_id,
            classifier_id=mock_specs.classifier_id,
            aws_env=AwsEnv.staging,
            add_classifiers_profiles=[mock_profile_mapping.classifiers_profile],
            remove_classifiers_profiles=[mock_specs.classifiers_profile],
            upload_to_wandb=True,
        )

        # Ensure scripts.classifier_metadata.update was called with the correct arguments
        mock_update.assert_called_once_with(
            wikibase_id=mock_specs.wikibase_id,
            classifier_id=mock_specs.classifier_id,
            remove_classifiers_profiles=[mock_specs.classifiers_profile],
            add_classifiers_profiles=[mock_profile_mapping.classifiers_profile.value],
            aws_env=AwsEnv.staging,
            update_specs=False,
        )


def test_update_classifier_profile__dry_run(mock_specs, mock_profile_mapping):
    """Test updating a classifier profile for successful validation."""

    # mock response to wandb scripts
    with (
        patch("scripts.classifier_metadata.update") as mock_update,
    ):
        update_classifier_profile(
            wikibase_id=mock_specs.wikibase_id,
            classifier_id=mock_specs.classifier_id,
            aws_env=AwsEnv.staging,
            add_classifiers_profiles=[mock_profile_mapping.classifiers_profile],
            remove_classifiers_profiles=[mock_specs.classifiers_profile],
            upload_to_wandb=False,
        )

        # Ensure scripts.demote.main was not called when dry run
        mock_update.assert_not_called()


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
            upload_to_wandb=False,
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
    assert (
        f"artifact validation failed: classifier name {mock_artifact.metadata.get('classifier_name')} violates classifier name rules"
        in result._error.msg
    )


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
    assert "artifact validation failed: run config" in result._error.msg


def test_create_vespa_classifiers_profile():
    mappings = [
        VespaClassifiersProfile.Mapping(
            concept_id="aaaa2345",
            concept_wikibase_id=VespaWikibaseId("Q1"),
            classifier_id="bbbb3456",
        )
    ]
    profile = create_vespa_classifiers_profile(Profile.PRIMARY, mappings)
    assert profile.name == "primary"
    assert len(profile.mappings) == 1
    assert isinstance(profile, VespaClassifiersProfile)
    assert profile.mappings[0] == mappings[0]
    assert profile.multi is False
    assert profile.response_raw == {}


def test_create_vespa_profile_mappings(mock_specs_2profiles):
    profile_mappings = create_vespa_profile_mappings(mock_specs_2profiles)

    assert len(profile_mappings) == len(mock_specs_2profiles)
    assert all(isinstance(m, VespaClassifiersProfile.Mapping) for m in profile_mappings)

    for m, spec in zip(profile_mappings, mock_specs_2profiles):
        assert m.concept_id == spec.concept_id
        assert m.concept_wikibase_id == VespaWikibaseId(spec.wikibase_id)
        assert m.classifier_id == spec.classifier_id


@pytest.mark.asyncio
async def test_update_vespa_with_classifiers_profiles__success(mock_specs_2profiles):
    """Test successful updates to Vespa with valid classifiers specs"""

    with (
        patch("flows.classifiers_profiles.VespaAsync") as mock_vespa_connection_pool,
    ):
        # Mock Vespa connection pool
        mock_vespa_connection_pool.update_data = AsyncMock(
            return_value=Mock(is_successful=lambda: True)
        )

        # Call the function
        results = await update_vespa_with_classifiers_profiles(
            classifier_specs=mock_specs_2profiles,
            vespa_connection_pool=mock_vespa_connection_pool,
        )

        # Assertions
        assert len(results) == 1  # No errors
        assert all(is_ok(r) for r in results)
        assert (
            mock_vespa_connection_pool.update_data.call_count
            == len(mock_specs_2profiles) + 1
        )  # 2 profiles + 1 classifiers_profiles


@pytest.mark.asyncio
async def test_update_vespa_with_classifiers__profiles_mapping_failure(
    mock_specs_2profiles,
):
    """Test Vespa update with mapping creation failure."""
    mock_specs = [
        Mock(
            classifiers_profile="primary",
            concept_id="abcdeg98",
            classifier_id="xyz23456",
            wikibase_id="abc123",  # invalid wikibase id
        )
    ]
    with (
        patch("flows.classifiers_profiles.VespaAsync") as mock_vespa_connection_pool,
    ):
        # Call the function
        results = await update_vespa_with_classifiers_profiles(
            classifier_specs=mock_specs,
            vespa_connection_pool=mock_vespa_connection_pool,
            upload_to_vespa=False,
        )

        # Assertions
        assert len(results) == 1  # One error
        assert isinstance(results[0], Err)
        assert (
            f"Failed to create mapping for {mock_specs[0].wikibase_id}"
            in results[0]._error.msg
        )
        mock_vespa_connection_pool.assert_not_called()


@pytest.mark.asyncio
async def test_update_vespa_with_classifiers_profiles__profile_creation_failure(
    mock_specs_2profiles,
):
    """Test Vespa update with profile creation failure."""
    with (
        patch(
            "flows.classifiers_profiles.create_vespa_classifiers_profile"
        ) as mock_create_profile,
        patch("flows.classifiers_profiles.VespaAsync") as mock_vespa_connection_pool,
    ):
        # Mock Vespa profile creation to raise an exception
        mock_create_profile.side_effect = ValueError("Profile creation failed")

        # Call the function
        results = await update_vespa_with_classifiers_profiles(
            classifier_specs=mock_specs_2profiles,
            vespa_connection_pool=mock_vespa_connection_pool,
        )

        # Assertions
        assert len(results) == 1  # One error
        assert isinstance(results[0], Err)
        assert "Profile creation failed" in results[0]._error.msg
        mock_vespa_connection_pool.assert_not_called()


@pytest.mark.asyncio
async def test_update_vespa_with_classifiers_profiles__vespa_sync_failure(
    mock_specs_2profiles,
):
    """Test Vespa update with sync failure."""
    with (
        patch("flows.classifiers_profiles.VespaAsync") as mock_vespa_connection_pool,
    ):
        # Mock Vespa connection pool to simulate a sync failure
        mock_vespa_connection_pool.update_data = AsyncMock(
            return_value=Mock(is_successful=lambda: False)
        )

        # Call the function
        results = await update_vespa_with_classifiers_profiles(
            classifier_specs=mock_specs_2profiles,
            vespa_connection_pool=mock_vespa_connection_pool,
        )

        # Assertions
        assert len(results) == 1  # One error
        assert isinstance(results[0], Err)
        assert "Error syncing VespaClassifiersProfile" in results[0]._error.msg
        assert (
            mock_vespa_connection_pool.update_data.call_count == 1
        )  # fails and returns


@pytest.mark.vespa
@pytest.mark.asyncio
async def test_update_vespa_with_classifiers_profiles__real_vespa(
    mock_specs_2profiles, local_vespa_search_adapter
):
    """Test update_vespa_with_classifiers_profiles with real Vespa and mock classifier specs."""
    async with local_vespa_search_adapter.client.asyncio(
        connections=1
    ) as vespa_connection_pool:
        results = await update_vespa_with_classifiers_profiles(
            mock_specs_2profiles, vespa_connection_pool
        )

        assert len(results) == 1  # One for the classifiers_profiles update
        assert all(is_ok(r) for r in results), f"All results should be Ok, {results}"


@pytest.mark.asyncio
async def test_update_vespa_with_classifiers_profiles__vespa_upload_false(
    mock_specs_2profiles,
):
    """Test Vespa update with flag upload_to_vespa set to False."""
    with (
        patch("flows.classifiers_profiles.VespaAsync") as mock_vespa_connection_pool,
    ):
        # Call the function with mock specs
        results = await update_vespa_with_classifiers_profiles(
            classifier_specs=mock_specs_2profiles,
            vespa_connection_pool=mock_vespa_connection_pool,
            upload_to_vespa=False,
        )

        # Assertions
        assert len(results) == 1  # Ok for no upload
        assert all(is_ok(r) for r in results), f"All results should be Ok, {results}"

        mock_vespa_connection_pool.update_data.assert_not_called()
        assert unwrap_ok(results[0]) is None


def test_emit_finished__success():
    """Test emit_finished with promotions succeeds."""
    promotions = [
        Promote(
            classifiers_profile_mapping=ClassifiersProfileMapping(
                wikibase_id=WikibaseID("Q123"),
                classifier_id=ClassifierID("aaaa2222"),
                classifiers_profile=Profile.PRIMARY,
            )
        )
    ]

    with patch("flows.classifiers_profiles.emit_event") as mock_emit_event:
        mock_event = Mock()
        mock_emit_event.return_value = mock_event

        result = emit_finished(promotions=promotions, aws_env=AwsEnv.staging)

        mock_emit_event.assert_called_once()
        call_args = mock_emit_event.call_args
        assert call_args.kwargs["event"] == "sync-classifiers_profiles.finished"
        assert (
            call_args.kwargs["resource"]["prefect.resource.id"]
            == "sync-classifiers-profiles"
        )
        assert call_args.kwargs["resource"]["awsenv"] == AwsEnv.staging
        assert len(call_args.kwargs["payload"]["promotions"]) == 1

        assert result == Ok(mock_event)


def test_emit_finished__no_promotions():
    """Test emit_finished with no promotions returns."""
    result = emit_finished(promotions=[], aws_env=AwsEnv.staging)

    assert result == Ok(None)


def test_emit_finished__emit_event_returns_none():
    """Test emit_finished when emit_event returns."""
    promotions = [
        Promote(
            classifiers_profile_mapping=ClassifiersProfileMapping(
                wikibase_id=WikibaseID("Q123"),
                classifier_id=ClassifierID("aaaa2222"),
                classifiers_profile=Profile.PRIMARY,
            )
        )
    ]

    with patch("flows.classifiers_profiles.emit_event") as mock_emit_event:
        mock_emit_event.return_value = None

        result = emit_finished(promotions=promotions, aws_env=AwsEnv.staging)

        assert result == Err(
            Error(
                msg="emitting event returned `None`, indicating it wasn't sent",
                metadata={
                    "event": None,
                    "resource": {
                        "prefect.resource.id": "sync-classifiers-profiles",
                        "awsenv": "staging",
                    },
                    "payload": {
                        "promotions": [
                            {
                                "classifiers_profile_mapping": {
                                    "wikibase_id": "Q123",
                                    "classifier_id": "aaaa2222",
                                    "classifiers_profile": "primary",
                                }
                            }
                        ]
                    },
                },
            )
        )


def test_emit_finished__emit_event_raises_exception():
    """Test emit_finished when emit_event raises an exception."""
    promotions = [
        Promote(
            classifiers_profile_mapping=ClassifiersProfileMapping(
                wikibase_id=WikibaseID("Q123"),
                classifier_id=ClassifierID("aaaa2222"),
                classifiers_profile=Profile.PRIMARY,
            )
        )
    ]

    with patch("flows.classifiers_profiles.emit_event") as mock_emit_event:
        mock_emit_event.side_effect = Exception("Failed to emit event")

        result = emit_finished(promotions=promotions, aws_env=AwsEnv.staging)

        assert result == Err(
            Error(
                msg="failed to emit event",
                metadata={
                    "event": "sync-classifiers_profiles.finished",
                    "resource": {
                        "prefect.resource.id": "sync-classifiers-profiles",
                        "awsenv": "staging",
                    },
                    "payload": {
                        "promotions": [
                            {
                                "classifiers_profile_mapping": {
                                    "wikibase_id": "Q123",
                                    "classifier_id": "aaaa2222",
                                    "classifiers_profile": "primary",
                                }
                            }
                        ]
                    },
                    "exception": "Failed to emit event",
                },
            )
        )
