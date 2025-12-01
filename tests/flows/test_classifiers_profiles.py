from unittest.mock import ANY, AsyncMock, Mock, patch

import pytest
from cpr_sdk.models.search import ClassifiersProfile as VespaClassifiersProfile
from cpr_sdk.models.search import WikibaseId as VespaWikibaseId
from cpr_sdk.search_adaptors import VespaSearchAdapter
from pydantic import SecretStr

from flows.classifier_specs.spec_interface import ClassifierSpec
from flows.classifiers_profiles import (
    _post_errors_main,
    _post_errors_thread,
    compare_classifiers_profiles,
    concept_present_in_vespa,
    create_classifiers_profiles_artifact,
    create_vespa_classifiers_profile,
    create_vespa_profile_mappings,
    demote_classifier_profile,
    emit_finished,
    get_classifiers_profiles,
    handle_classifier_profile_action,
    maybe_allow_retiring,
    promote_classifier_profile,
    read_concepts,
    send_classifiers_profile_slack_alert,
    sync_classifiers_profiles,
    update_classifier_profile,
    update_vespa_with_classifiers_profiles,
    validate_artifact_metadata_rules,
    wandb_validation,
)
from flows.result import Err, Error, Ok, Result, is_err, is_ok, unwrap_err, unwrap_ok
from knowledge_graph.classifiers_profiles import (
    ClassifiersProfileMapping,
    Profile,
)
from knowledge_graph.cloud import AwsEnv
from knowledge_graph.compare_result_operation import Demote, Ignore, Promote, Update
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import ClassifierID, WikibaseID
from knowledge_graph.version import Version
from knowledge_graph.wikibase import StatementRank, WikibaseAuth


@pytest.fixture
def mock_specs():
    return ClassifierSpec(
        wikibase_id="Q123",
        classifier_id="abcd3456",
        classifiers_profile="primary",
        wandb_registry_version="v20",
        concept_id="mmmm5555",
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


@pytest.fixture
def mock_concepts():
    return [
        Concept(wikibase_id="Q123", preferred_label="Concept 123"),
        Concept(wikibase_id="Q100", preferred_label="Concept 100"),
        Concept(wikibase_id="Q999", preferred_label="Concept 999"),
        Concept(wikibase_id="Q200", preferred_label="Concept 200"),
        Concept(wikibase_id="Q201", preferred_label="Concept 201"),
    ]


@pytest.fixture
def mock_classifier_ids():
    return [
        # Q123: success
        [
            (StatementRank.PREFERRED, ClassifierID("aaaa2222")),
            (StatementRank.DEPRECATED, ClassifierID("abcd3456")),
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
        # Q201: debug False - success (no profiles)
        [],
    ]


@pytest.fixture
def mock_wikibase_auth():
    return WikibaseAuth(
        username="test",
        password="password",
        url="https://example.com",
    )


@pytest.mark.asyncio
async def test_get_classifiers_profiles(mock_concepts, mock_classifier_ids):
    with patch("flows.classifiers_profiles.WikibaseSession") as mock_wikibase_session:
        # mock response from wikibase.get_classifier_ids_async
        mock_wikibase_auth = Mock()
        mock_wikibase = mock_wikibase_session.return_value

        # 1 success, 3 failures
        mock_wikibase.get_classifier_ids_async = AsyncMock(
            side_effect=mock_classifier_ids
        )

        # Call the function under test
        results = await get_classifiers_profiles(
            wikibase_auth=mock_wikibase_auth, concepts=mock_concepts, debug=False
        )

        assert mock_wikibase.get_classifier_ids_async.call_count == 5  # one per concept

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
        classifier_id=ClassifierID("abcd3456"),
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


@pytest.mark.asyncio
async def test_get_classifiers_profiles__debug(mock_concepts, mock_classifier_ids):
    with patch("flows.classifiers_profiles.WikibaseSession") as mock_wikibase_session:
        # mock response from wikibase.get_classifier_ids_async
        mock_wikibase_auth = Mock()
        mock_wikibase = mock_wikibase_session.return_value

        # 1 success, 3 failures
        mock_wikibase.get_classifier_ids_async = AsyncMock(
            side_effect=mock_classifier_ids,
        )
        # Call the function with debug mode on - results in extra failure
        results_debug = await get_classifiers_profiles(
            wikibase_auth=mock_wikibase_auth, concepts=mock_concepts, debug=True
        )

    classifier_profiles = [unwrap_ok(r) for r in results_debug if isinstance(r, Ok)]
    assert len(classifier_profiles) == 2

    failures = [unwrap_err(r) for r in results_debug if isinstance(r, Err)]
    assert len(failures) == 4


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


def test_concept_present_in_vespa__has_results():
    """Test concept_present_in_vespa when concept has results in Vespa."""
    mock_search_adapter = Mock()
    mock_search_adapter.search.return_value = Mock(results=[{"doc1": "data"}])

    result = concept_present_in_vespa(
        wikibase_id=WikibaseID("Q123"),
        classifier_id=ClassifierID("aaaa2222"),
        vespa_search_adapter=mock_search_adapter,
    )

    assert result == Ok(True)

    # Verify search was called with correct parameters
    call_args = mock_search_adapter.search.call_args
    search_params = call_args[0][0]
    assert len(search_params.concept_v2_document_filters) == 1
    assert search_params.concept_v2_document_filters[0].concept_wikibase_id == "Q123"
    assert search_params.concept_v2_document_filters[0].classifier_id == "aaaa2222"
    assert search_params.documents_only is True
    assert search_params.limit == 1


def test_concept_present_in_vespa__no_results():
    """Test concept_present_in_vespa when concept has no results in Vespa."""
    mock_search_adapter = Mock()
    mock_search_adapter.search.return_value = Mock(results=[])

    result = concept_present_in_vespa(
        wikibase_id=WikibaseID("Q123"),
        classifier_id=ClassifierID("aaaa2222"),
        vespa_search_adapter=mock_search_adapter,
    )

    assert result == Ok(False)


def test_concept_present_in_vespa__search_error():
    """Test concept_present_in_vespa when Vespa search raises an exception."""
    mock_search_adapter = Mock()
    mock_search_adapter.search.side_effect = Exception("Vespa connection failed")

    result = concept_present_in_vespa(
        wikibase_id=WikibaseID("Q123"),
        classifier_id=ClassifierID("aaaa2222"),
        vespa_search_adapter=mock_search_adapter,
    )

    assert result == Err(
        _error=Error(
            msg="failed to search Vespa for results",
            metadata={
                "concept_wikibase_id": "Q123",
                "classifier_id": "aaaa2222",
                "exception": "Vespa connection failed",
            },
        )
    )


def test_maybe_allow_retiring__retiring_profile_with_results():
    """Test maybe_allow_retiring allows retiring when concept has results in Vespa."""
    promote_op = Promote(
        classifiers_profile_mapping=ClassifiersProfileMapping(
            wikibase_id=WikibaseID("Q123"),
            classifier_id=ClassifierID("aaaa2222"),
            classifiers_profile=Profile.RETIRED,
        )
    )

    mock_search_adapter = Mock()
    mock_search_adapter.search.return_value = Mock(results=[{"doc1": "data"}])

    wandb_results = []

    allow, updated_results = maybe_allow_retiring(
        promote_op, mock_search_adapter, wandb_results
    )

    assert allow
    assert updated_results == []


def test_maybe_allow_retiring__retiring_profile_without_results():
    """Test maybe_allow_retiring blocks retiring when concept has no results in Vespa."""
    promote_op = Promote(
        classifiers_profile_mapping=ClassifiersProfileMapping(
            wikibase_id=WikibaseID("Q123"),
            classifier_id=ClassifierID("aaaa2222"),
            classifiers_profile=Profile.RETIRED,
        )
    )

    mock_search_adapter = Mock()
    mock_search_adapter.search.return_value = Mock(results=[])

    wandb_results = []

    allow, updated_results = maybe_allow_retiring(
        promote_op, mock_search_adapter, wandb_results
    )

    assert not allow
    assert updated_results == [
        Err(
            _error=Error(
                msg="no results found in Vespa, so can't retire",
                metadata={
                    "wikibase_id": "Q123",
                    "classifier_id": "aaaa2222",
                    "classifiers_profile": "retired",
                },
            )
        )
    ]


def test_maybe_allow_retiring__retiring_profile_vespa_error():
    """Test maybe_allow_retiring blocks retiring when Vespa check fails."""
    update_op = Update(
        classifier_spec=ClassifierSpec(
            wikibase_id="Q123",
            classifier_id="aaaa2222",
            classifiers_profile="primary",
            wandb_registry_version="v1",
        ),
        classifiers_profile_mapping=ClassifiersProfileMapping(
            wikibase_id=WikibaseID("Q123"),
            classifier_id=ClassifierID("aaaa2222"),
            classifiers_profile=Profile.RETIRED,
        ),
    )

    mock_search_adapter = Mock()
    mock_search_adapter.search.side_effect = Exception("Vespa connection failed")

    wandb_results = []

    allow, updated_results = maybe_allow_retiring(
        update_op, mock_search_adapter, wandb_results
    )

    assert not allow
    assert updated_results == [
        Err(
            _error=Error(
                msg="failed to search Vespa for results. Failed to check for results in Vespa, so can't retire",
                metadata={
                    "concept_wikibase_id": "Q123",
                    "classifier_id": "aaaa2222",
                    "exception": "Vespa connection failed",
                },
            )
        )
    ]


def test_maybe_allow_retiring__non_retiring_profile():
    """Test maybe_allow_retiring allows non-retiring operations without checks."""
    promote_op = Promote(
        classifiers_profile_mapping=ClassifiersProfileMapping(
            wikibase_id=WikibaseID("Q123"),
            classifier_id=ClassifierID("aaaa2222"),
            classifiers_profile=Profile.PRIMARY,
        )
    )

    mock_search_adapter = Mock()
    wandb_results = []

    allow, updated_results = maybe_allow_retiring(
        promote_op, mock_search_adapter, wandb_results
    )

    assert allow
    assert updated_results == []
    # Verify Vespa search was not called
    mock_search_adapter.search.assert_not_called()


def test_maybe_allow_retiring__update_to_retired():
    """Test maybe_allow_retiring checks Vespa when updating to retired profile."""
    update_op = Update(
        classifier_spec=ClassifierSpec(
            wikibase_id="Q123",
            classifier_id="aaaa2222",
            classifiers_profile="experimental",
            wandb_registry_version="v1",
        ),
        classifiers_profile_mapping=ClassifiersProfileMapping(
            wikibase_id=WikibaseID("Q123"),
            classifier_id=ClassifierID("aaaa2222"),
            classifiers_profile=Profile.RETIRED,
        ),
    )

    mock_search_adapter = Mock()
    mock_search_adapter.search.return_value = Mock(results=[{"doc1": "data"}])

    wandb_results = []

    allow, updated_results = maybe_allow_retiring(
        update_op, mock_search_adapter, wandb_results
    )

    assert allow
    assert updated_results == []
    # Verify Vespa search was called
    mock_search_adapter.search.assert_called_once()


@pytest.mark.asyncio
async def test_read_concepts(mock_wikibase_auth, mock_concepts):
    mock_wikibase_session = AsyncMock()
    mock_wikibase_session.get_concepts_async.return_value = mock_concepts

    with patch(
        "flows.classifiers_profiles.WikibaseSession", return_value=mock_wikibase_session
    ):
        concepts = await read_concepts(
            wikibase_auth=mock_wikibase_auth,
            wikibase_cache_path=None,
            wikibase_cache_save_if_missing=False,
        )

    assert concepts == mock_concepts
    mock_wikibase_session.get_concepts_async.assert_called_once()


@pytest.mark.asyncio
async def test_send_classifiers_profile_slack_alert_success():
    """Test send_classifiers_profile_slack_alert with successful Slack messages."""
    mock_slack_client = AsyncMock()
    mock_slack_client.chat_postMessage.return_value = {"ok": True, "ts": "12345"}

    validation_errors = [
        Error(msg="Validation error", metadata={"wikibase_id": "Q1"}),
        Error(msg="Validation error", metadata={"wikibase_id": "Q2"}),
    ]
    wandb_errors = [
        Error(msg="WandB error", metadata={"wikibase_id": "Q3"}),
    ]
    vespa_errors = [
        Error(msg="Vespa error", metadata={"wikibase_id": "Q4"}),
    ]
    successes = [
        {"wikibase_id": "Q5", "classifier_id": "abcd2345"},
        {"wikibase_id": "Q6", "classifier_id": "yyyy8888"},
    ]
    pr_results: Result[int | None, Error] = Err(Error(msg="PR error", metadata={}))

    with (
        patch(
            "flows.classifiers_profiles.get_slack_client",
            return_value=mock_slack_client,
        ),
        patch(
            "flows.classifiers_profiles._post_errors_main", wraps=_post_errors_main
        ) as spy_post_errors_main,
        patch(
            "flows.classifiers_profiles._post_errors_thread", wraps=_post_errors_thread
        ) as spy_post_errors_thread,
    ):
        await send_classifiers_profile_slack_alert(
            validation_errors=validation_errors,
            wandb_errors=wandb_errors,
            vespa_errors=vespa_errors,
            successes=successes,
            aws_env=AwsEnv.staging,
            upload_to_wandb=True,
            upload_to_vespa=True,
            event=Mock(),
            cs_pr_results=pr_results,
        )

        mock_slack_client.chat_postMessage.assert_any_call(
            channel="alerts-platform-staging",
            text="Classifiers Profile Sync Summary: uploading to wandb uploading to vespa",
            attachments=ANY,  # ignore content of attachments
        )

        mock_slack_client.chat_postMessage.assert_any_call(
            channel="alerts-platform-staging",
            thread_ts="12345",
            text=f"Data Quality Issues: {len(validation_errors)} issues found",
            blocks=ANY,
        )
        mock_slack_client.chat_postMessage.assert_any_call(
            channel="alerts-platform-staging",
            thread_ts="12345",
            text=f"WandB Errors: {len(wandb_errors)} issues found",
            blocks=ANY,
        )
        mock_slack_client.chat_postMessage.assert_any_call(
            channel="alerts-platform-staging",
            thread_ts="12345",
            text=f"Vespa Errors: {len(vespa_errors)} issues found",
            blocks=ANY,
        )
        mock_slack_client.chat_postMessage.assert_any_call(
            channel="alerts-platform-staging",
            thread_ts="12345",
            text="PR Errors: 1 issues found",
            blocks=ANY,
        )
        assert spy_post_errors_main.call_count == 2
        assert spy_post_errors_thread.call_count == 4


@pytest.mark.asyncio
async def test_send_classifiers_profile_slack_alert__slack_failure():
    """Test send_classifiers_profile_slack_alert when Slack API fails."""
    mock_slack_client = AsyncMock()
    mock_slack_client.chat_postMessage.return_value = {
        "ok": False,
        "error": "Slack API failure",
    }
    with (
        patch(
            "flows.classifiers_profiles.get_slack_client",
            return_value=mock_slack_client,
        ),
        patch(
            "flows.classifiers_profiles._post_errors_main", wraps=_post_errors_main
        ) as spy_post_errors_main,
        patch(
            "flows.classifiers_profiles._post_errors_thread", wraps=_post_errors_thread
        ) as spy_post_errors_thread,
    ):
        await send_classifiers_profile_slack_alert(
            validation_errors=[
                Error(msg="Validation error: test", metadata={"wikibase_id": "Q1"}),
            ],
            wandb_errors=[],
            vespa_errors=[],
            successes=[],
            aws_env=AwsEnv.staging,
            upload_to_wandb=True,
            upload_to_vespa=True,
            event=Mock(),
            cs_pr_results=Mock(),
        )

        mock_slack_client.chat_postMessage.assert_called_once_with(
            channel="alerts-platform-staging",
            text="Classifiers Profile Sync Summary: uploading to wandb uploading to vespa",
            attachments=ANY,
        )

        assert spy_post_errors_main.call_count == 1
        spy_post_errors_thread.assert_not_called()


@pytest.mark.asyncio
async def test_sync_classifiers_profiles(
    mock_wikibase_auth, mock_concepts, mock_classifier_ids, mock_specs
):
    """Test full sync_classifiers_profiles with success."""

    # mock wikibase session and return concepts and classifier ids from calls
    mock_wikibase_session = AsyncMock()
    mock_wikibase_session.get_concepts_async.return_value = mock_concepts
    mock_wikibase_session.get_classifier_ids_async = AsyncMock(
        side_effect=mock_classifier_ids
    )

    # mock wandb api key
    mock_wandb_api_key = Mock(SecretStr("mock-wandb-api-key"))
    mock_wandb_api_key.get_secret_value.return_value = "mock-wandb-api-key"

    # mock vespa search adaptor for calls to concept_present_in_vespa
    mock_vespa_search_adapter = Mock(VespaSearchAdapter(instance_url="test-url"))
    mock_vespa_search_adapter.search.return_value = Mock(results=[{"doc1": "data"}])
    mock_vespa_search_adapter.client.asyncio.return_value = AsyncMock()

    # mock create_and_merge_pr results as async function
    pr_number = 123
    mock_pr_results = Ok(pr_number)
    mock_create_and_merge_pr = AsyncMock(return_value=mock_pr_results)

    # wandb validation mocks
    mock_metadata = {"aws_env": "sandbox", "classifier_name": "ValidClassifier"}
    mock_artifacts = [
        Mock(version="v1", metadata=mock_metadata),
        Mock(version="v2", metadata=mock_metadata),
    ]
    mock_api = Mock()
    mock_api.return_value.artifacts.return_value = mock_artifacts

    mock_artifact = Mock()
    mock_artifact.metadata = mock_metadata
    mock_artifact.logged_by.return_value.config = {}

    mock_api.return_value.artifact.return_value = mock_artifact

    mock_slack_client = AsyncMock()
    mock_slack_client.chat_postMessage.return_value = {"ok": True, "ts": "12345"}

    mock_updated_specs = [
        ClassifierSpec(
            wikibase_id="Q123",
            classifier_id="aaaa2222",
            classifiers_profile="primary",
            wandb_registry_version="v2",
            concept_id="mmmm5566",
        ),
        ClassifierSpec(
            wikibase_id="Q123",
            classifier_id="abcd3456",
            classifiers_profile="retired",
            wandb_registry_version="v20",
            concept_id="mmmm5555",
        ),
    ]

    with (
        patch(
            "flows.classifiers_profiles.WikibaseSession",
            return_value=mock_wikibase_session,
        ),
        patch(
            "flows.classifiers_profiles.load_classifier_specs",
            side_effect=[[mock_specs], mock_updated_specs],
        ),
        patch("wandb.login") as mock_wandb_login,
        patch("wandb.Api", return_value=mock_api.return_value),
        patch("flows.classifiers_profiles.refresh_all_available_classifiers"),
        patch(
            "flows.classifiers_profiles.create_classifiers_specs_pr.create_and_merge_pr",
            mock_create_and_merge_pr,
        ),
        patch(
            "flows.classifiers_profiles.get_slack_client",
            return_value=mock_slack_client,
        ),
        patch(
            "flows.classifiers_profiles.handle_classifier_profile_action",
            wraps=handle_classifier_profile_action,
        ) as spy_handle_classifier_profile_action,
        patch(
            "flows.classifiers_profiles.acreate_table_artifact", new_callable=AsyncMock
        ) as mock_acreate_table_artifact,
        patch(
            "flows.classifiers_profiles.update_vespa_with_classifiers_profiles",
            wraps=update_vespa_with_classifiers_profiles,
        ) as spy_update_vespa_with_classifiers_profiles,
    ):
        await sync_classifiers_profiles(
            wandb_api_key=mock_wandb_api_key,
            wikibase_auth=mock_wikibase_auth,
            vespa_search_adapter=mock_vespa_search_adapter,
            github_token=Mock(SecretStr("mock-github-token")),
            upload_to_wandb=False,
            upload_to_vespa=False,
            automerge_classifier_specs_pr=False,
            auto_train=False,
        )

        # check wandb login called with api key
        mock_wandb_login.assert_called_once_with(key="mock-wandb-api-key")

        # check handle_classifier_profile_action called twice: 1 update, 1 promote
        assert spy_handle_classifier_profile_action.call_count == 2
        action_calls = [
            call.kwargs["action"]
            for call in spy_handle_classifier_profile_action.call_args_list
        ]
        assert "updating" in action_calls
        assert "promoting" in action_calls

        # check vespa search called once (1 update with retired profile)
        mock_vespa_search_adapter.search.assert_called_once()
        # check create and merge pr called once
        mock_create_and_merge_pr.assert_called_once()
        # check vespa called
        spy_update_vespa_with_classifiers_profiles.assert_called_once()

        # check slack messages sent, only validation errors so: 1 main, 1 thread
        assert mock_slack_client.chat_postMessage.call_count == 2
        mock_slack_client.chat_postMessage.assert_any_call(
            channel="alerts-platform-sandbox",
            text="Classifiers Profile Sync Summary: (dry run, not uploading to wandb) (dry run, not uploading to vespa)",
            attachments=ANY,
        )

        # use artifact call args to check final results
        mock_acreate_table_artifact.assert_called_once()
        artifact_call_args = mock_acreate_table_artifact.call_args.kwargs
        assert artifact_call_args["key"] == "classifiers-profiles-validation-sandbox"
        assert (
            len(artifact_call_args["table"]) == 2 + 3
        )  # number rows: 2 successes + 3 validation errors
        assert (
            f"**Classifiers Specs PR**: [#{pr_number}]"
            in artifact_call_args["description"]
        )


@pytest.mark.asyncio
async def test_sync_classifiers_profiles__failure_creating_pr(
    mock_wikibase_auth, mock_concepts, mock_classifier_ids, mock_specs
):
    """Test full sync_classifiers_profiles when create/merge PR fails"""

    # mock wikibase session and return concepts and classifier ids from calls
    mock_wikibase_session = AsyncMock()
    mock_wikibase_session.get_concepts_async.return_value = mock_concepts
    mock_wikibase_session.get_classifier_ids_async = AsyncMock(
        side_effect=mock_classifier_ids
    )

    # mock wandb api key
    mock_wandb_api_key = Mock(SecretStr("mock-wandb-api-key"))
    mock_wandb_api_key.get_secret_value.return_value = "mock-wandb-api-key"

    # mock vespa search adaptor for calls to concept_present_in_vespa
    mock_vespa_search_adapter = Mock(VespaSearchAdapter(instance_url="test-url"))
    mock_vespa_search_adapter.search.return_value = Mock(results=[{"doc1": "data"}])
    mock_vespa_search_adapter.client.asyncio.return_value = AsyncMock()

    # mock create_and_merge_pr results as async function with error
    mock_pr_results = Err(Error(msg="Error creating PR", metadata={}))
    mock_create_and_merge_pr = AsyncMock(return_value=mock_pr_results)

    # wandb validation mocks
    mock_metadata = {"aws_env": "sandbox", "classifier_name": "ValidClassifier"}
    mock_artifacts = [
        Mock(version="v1", metadata=mock_metadata),
        Mock(version="v2", metadata=mock_metadata),
    ]
    mock_api = Mock()
    mock_api.return_value.artifacts.return_value = mock_artifacts

    mock_artifact = Mock()
    mock_artifact.metadata = mock_metadata
    mock_artifact.logged_by.return_value.config = {}

    mock_api.return_value.artifact.return_value = mock_artifact

    mock_slack_client = AsyncMock()
    mock_slack_client.chat_postMessage.return_value = {"ok": True, "ts": "12345"}

    mock_updated_specs = [
        ClassifierSpec(
            wikibase_id="Q123",
            classifier_id="aaaa2222",
            classifiers_profile="primary",
            wandb_registry_version="v2",
            concept_id="mmmm5566",
        ),
        ClassifierSpec(
            wikibase_id="Q123",
            classifier_id="abcd3456",
            classifiers_profile="retired",
            wandb_registry_version="v20",
            concept_id="mmmm5555",
        ),
    ]

    with (
        patch(
            "flows.classifiers_profiles.WikibaseSession",
            return_value=mock_wikibase_session,
        ),
        patch(
            "flows.classifiers_profiles.load_classifier_specs",
            side_effect=[[mock_specs], mock_updated_specs],
        ),
        patch("wandb.login"),
        patch("wandb.Api", return_value=mock_api.return_value),
        patch("flows.classifiers_profiles.refresh_all_available_classifiers"),
        patch(
            "flows.classifiers_profiles.create_classifiers_specs_pr.create_and_merge_pr",
            mock_create_and_merge_pr,
        ),
        patch(
            "flows.classifiers_profiles.get_slack_client",
            return_value=mock_slack_client,
        ),
        patch(
            "flows.classifiers_profiles.acreate_table_artifact", new_callable=AsyncMock
        ) as mock_acreate_table_artifact,
        patch(
            "flows.classifiers_profiles.update_vespa_with_classifiers_profiles",
            wraps=update_vespa_with_classifiers_profiles,
        ) as spy_update_vespa_with_classifiers_profiles,
    ):
        with pytest.raises(
            Exception, match="Errors occurred while creating classifiers specs PR"
        ):
            await sync_classifiers_profiles(
                wandb_api_key=mock_wandb_api_key,
                wikibase_auth=mock_wikibase_auth,
                vespa_search_adapter=mock_vespa_search_adapter,
                github_token=Mock(SecretStr("mock-github-token")),
                upload_to_wandb=False,
                upload_to_vespa=False,
                automerge_classifier_specs_pr=False,
                auto_train=False,
            )

        # vespa should not be called when create PR fails
        spy_update_vespa_with_classifiers_profiles.assert_not_called()
        # check create and merge pr called once
        mock_create_and_merge_pr.assert_called_once()

        # use artifact call args to check final results
        mock_acreate_table_artifact.assert_called_once()
        artifact_call_args = mock_acreate_table_artifact.call_args.kwargs
        assert artifact_call_args["key"] == "classifiers-profiles-validation-sandbox"
        assert (
            len(artifact_call_args["table"]) == 2 + 3
        )  # number rows: 2 successes + 3 validation errors
        assert (
            "**Classifiers Specs PR**: Error creating or merging PR"
            in artifact_call_args["description"]
        )


@pytest.mark.asyncio
async def test_sync_classifiers_profiles__failure_updating_vespa(
    mock_wikibase_auth, mock_concepts, mock_classifier_ids, mock_specs
):
    """
    Test full sync_classifiers_profiles when no changes to classifier specs

    Create and merge PR does not run and vespa fails
    """

    # mock wikibase session and return concepts and classifier ids from calls
    mock_wikibase_session = AsyncMock()
    mock_wikibase_session.get_concepts_async.return_value = mock_concepts
    mock_wikibase_session.get_classifier_ids_async = AsyncMock(
        side_effect=mock_classifier_ids
    )

    # mock wandb api key
    mock_wandb_api_key = Mock(SecretStr("mock-wandb-api-key"))
    mock_wandb_api_key.get_secret_value.return_value = "mock-wandb-api-key"

    # mock vespa search adaptor for calls to concept_present_in_vespa
    mock_vespa_search_adapter = Mock(VespaSearchAdapter(instance_url="test-url"))
    mock_vespa_search_adapter.search.return_value = Mock(results=[{"doc1": "data"}])
    mock_vespa_search_adapter.client.asyncio.return_value = AsyncMock()

    # mock create_and_merge_pr results
    mock_create_and_merge_pr = AsyncMock(return_value=Ok(123))

    # wandb validation mocks
    mock_metadata = {"aws_env": "sandbox", "classifier_name": "ValidClassifier"}
    mock_artifacts = [
        Mock(version="v1", metadata=mock_metadata),
        Mock(version="v2", metadata=mock_metadata),
    ]
    mock_api = Mock()
    mock_api.return_value.artifacts.return_value = mock_artifacts

    mock_artifact = Mock()
    mock_artifact.metadata = mock_metadata
    mock_artifact.logged_by.return_value.config = {}

    mock_api.return_value.artifact.return_value = mock_artifact

    mock_slack_client = AsyncMock()
    mock_slack_client.chat_postMessage.return_value = {"ok": True, "ts": "12345"}

    mock_vespa_results = [Err(Error(msg="Error creating Vespa Objects", metadata={}))]
    mock_update_vespa = AsyncMock(return_value=mock_vespa_results)

    with (
        patch(
            "flows.classifiers_profiles.WikibaseSession",
            return_value=mock_wikibase_session,
        ),
        patch(
            "flows.classifiers_profiles.load_classifier_specs",
            return_value=[mock_specs],
        ),
        patch("wandb.login"),
        patch("wandb.Api", return_value=mock_api.return_value),
        patch("flows.classifiers_profiles.refresh_all_available_classifiers"),
        patch(
            "flows.classifiers_profiles.update_vespa_with_classifiers_profiles",
            mock_update_vespa,
        ),
        patch(
            "flows.classifiers_profiles.create_classifiers_specs_pr.create_and_merge_pr",
            mock_create_and_merge_pr,
        ),
        patch(
            "flows.classifiers_profiles.get_slack_client",
            return_value=mock_slack_client,
        ),
        patch(
            "flows.classifiers_profiles.acreate_table_artifact", new_callable=AsyncMock
        ) as mock_acreate_table_artifact,
    ):
        with pytest.raises(
            Exception,
            match="Errors occurred while updating Vespa with classifiers profiles",
        ):
            await sync_classifiers_profiles(
                wandb_api_key=mock_wandb_api_key,
                wikibase_auth=mock_wikibase_auth,
                vespa_search_adapter=mock_vespa_search_adapter,
                github_token=Mock(SecretStr("mock-github-token")),
                upload_to_wandb=False,
                upload_to_vespa=False,
                automerge_classifier_specs_pr=False,
                auto_train=False,
            )

        # vespa should be called once and return exception
        mock_update_vespa.assert_called_once()
        # check create and merge pr should not be called as no change to classifiers specs
        mock_create_and_merge_pr.assert_not_called()

        # use artifact call args to check final results
        mock_acreate_table_artifact.assert_called_once()
        artifact_call_args = mock_acreate_table_artifact.call_args.kwargs
        assert artifact_call_args["key"] == "classifiers-profiles-validation-sandbox"
        assert (
            len(artifact_call_args["table"]) == 2 + 3 + 1
        )  # number rows: 2 successes + 3 validation errors + 1 vespa error
        assert (
            "**Classifiers Specs PR**: No PR created"
            in artifact_call_args["description"]
        )


@pytest.mark.asyncio
async def test_create_classifiers_profiles_artifact():
    validation_errors = [
        Error(msg="Validation error 1", metadata={"wikibase_id": "Q1"}),
        Error(msg="Validation error 2", metadata={"wikibase_id": "Q2"}),
    ]
    wandb_errors = [
        Error(msg="WandB error 1", metadata={"wikibase_id": "Q3"}),
    ]
    vespa_errors = [
        Error(msg="Vespa error 1", metadata={"wikibase_id": "Q4"}),
    ]
    successes = [
        {"wikibase_id": "Q5", "classifier_id": "abcd2345"},
        {"wikibase_id": "Q6", "classifier_id": "yyyy8888"},
    ]
    aws_env = AwsEnv.staging
    cs_pr_results = Ok(123)

    with patch(
        "flows.classifiers_profiles.acreate_table_artifact", new_callable=AsyncMock
    ) as mock_acreate_table_artifact:
        await create_classifiers_profiles_artifact(
            validation_errors=validation_errors,
            wandb_errors=wandb_errors,
            vespa_errors=vespa_errors,
            successes=successes,
            aws_env=aws_env,
            cs_pr_results=cs_pr_results,
        )

        mock_acreate_table_artifact.assert_called_once()

        call_args = mock_acreate_table_artifact.call_args
        key = call_args.kwargs["key"]
        table = call_args.kwargs["table"]
        description = call_args.kwargs["description"]

        # Assert the key is correct
        assert key == f"classifiers-profiles-validation-{aws_env.value}"

        # Assert the table contains the correct number of rows
        assert len(table) == len(successes) + len(validation_errors) + len(
            wandb_errors
        ) + len(vespa_errors)  # pr errors not added to table

        # Assert the description contains the PR number
        assert f"**Classifiers Specs PR**: [#{unwrap_ok(cs_pr_results)}]" in description


@pytest.mark.asyncio
async def test_create_classifiers_profiles_artifact__pr_error():
    validation_errors = []
    wandb_errors = []
    vespa_errors = []
    successes = [
        {"wikibase_id": "Q5", "classifier_id": "abcd2345"},
        {"wikibase_id": "Q6", "classifier_id": "yyyy8888"},
    ]
    aws_env = AwsEnv.staging
    cs_pr_results = Err(Error(msg="Failed to create PR", metadata={}))

    pr_errors = [unwrap_err(cs_pr_results)] if is_err(cs_pr_results) else []
    with patch(
        "flows.classifiers_profiles.acreate_table_artifact", new_callable=AsyncMock
    ) as mock_acreate_table_artifact:
        await create_classifiers_profiles_artifact(
            validation_errors=validation_errors,
            wandb_errors=wandb_errors,
            vespa_errors=vespa_errors,
            successes=successes,
            aws_env=aws_env,
            cs_pr_results=cs_pr_results,
        )

        mock_acreate_table_artifact.assert_called_once()

        call_args = mock_acreate_table_artifact.call_args
        key = call_args.kwargs["key"]
        table = call_args.kwargs["table"]
        description = call_args.kwargs["description"]

        # Assert the key is correct
        assert key == f"classifiers-profiles-validation-{aws_env.value}"

        # Assert the table contains the correct number of rows
        assert len(table) == len(successes) + len(validation_errors) + len(
            wandb_errors
        ) + len(vespa_errors)  # pr errors not added to table

        # Assert the description contains the PR number
        assert (
            f"**Classifiers Specs PR**: Error creating or merging PR {pr_errors[0]}"
            in description
        )
