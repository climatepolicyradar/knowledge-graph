import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from argilla import Dataset, ResponseStatus
from hypothesis import given
from hypothesis import strategies as st

from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.labelling import (
    ArgillaSession,
    ResourceAlreadyExistsError,
    ResourceDoesNotExistError,
)
from knowledge_graph.span import Span


def test_whether_argilla_session_initialisation_uses_environment_variables(
    monkeypatch, mock_argilla_client
):
    monkeypatch.setenv("ARGILLA_API_KEY", "test-api-key")
    monkeypatch.setenv("ARGILLA_API_URL", "http://test.argilla.url")

    _, mock_argilla_class = mock_argilla_client
    session = ArgillaSession()

    assert session.default_workspace == "knowledge-graph"
    mock_argilla_class.assert_called_once_with(
        api_key="test-api-key",
        api_url="http://test.argilla.url",
    )


def test_whether_argilla_session_initialisation_uses_explicit_parameters(
    mock_argilla_client,
):
    _, mock_argilla_class = mock_argilla_client
    session = ArgillaSession(
        api_key="custom-key",
        api_url="http://custom.url",
        workspace="custom-workspace",
    )

    assert session.default_workspace == "custom-workspace"
    mock_argilla_class.assert_called_once_with(
        api_key="custom-key",
        api_url="http://custom.url",
    )


def test_whether_argilla_session_string_representation_uses_default_workspace(
    mock_argilla_client,
):
    _ = mock_argilla_client
    session = ArgillaSession(workspace="test-workspace")
    assert repr(session) == "<ArgillaSession: workspace=test-workspace>"


def test_whether_get_workspace_finds_an_existing_workspace(
    mock_argilla_client, mock_workspace
):
    mock_client, _ = mock_argilla_client
    workspace = mock_workspace(name="test-workspace")
    mock_client.workspaces.return_value = workspace

    session = ArgillaSession()
    result = session.get_workspace("test-workspace")

    assert result == workspace
    mock_client.workspaces.assert_called_once_with(name="test-workspace")


def test_whether_get_workspace_uses_default_workspace(
    mock_argilla_client, mock_workspace
):
    mock_client, _ = mock_argilla_client
    workspace = mock_workspace(name="knowledge-graph")
    mock_client.workspaces.return_value = workspace

    session = ArgillaSession()
    result = session.get_workspace()

    assert result == workspace
    mock_client.workspaces.assert_called_once_with(name="knowledge-graph")


def test_whether_get_workspace_raises_error_if_workspace_not_found(
    mock_argilla_client,
):
    mock_client, _ = mock_argilla_client
    mock_client.workspaces.return_value = None

    session = ArgillaSession()

    with pytest.raises(
        ResourceDoesNotExistError, match="Workspace 'nonexistent' doesn't exist"
    ):
        session.get_workspace("nonexistent")


def test_whether_create_workspace_creates_a_new_workspace(mock_argilla_client):
    mock_client, _ = mock_argilla_client
    # Mock that the workspace doesn't exist
    mock_client.workspaces.return_value = None

    with patch("knowledge_graph.labelling.Workspace") as mock_workspace_class:
        mock_workspace = MagicMock()
        mock_workspace.name = "new-workspace"
        mock_workspace_class.return_value = mock_workspace
        mock_workspace.create.return_value = mock_workspace

        session = ArgillaSession()
        result = session.create_workspace("new-workspace")

        assert result == mock_workspace
        mock_workspace_class.assert_called_once_with(name="new-workspace")
        mock_workspace.create.assert_called_once()


def test_whether_create_workspace_raises_error_if_workspace_already_exists(
    mock_argilla_client, mock_workspace
):
    mock_client, _ = mock_argilla_client
    existing_workspace = mock_workspace(name="existing-workspace")
    mock_client.workspaces.return_value = existing_workspace

    session = ArgillaSession()

    with pytest.raises(
        ResourceAlreadyExistsError,
        match="Workspace 'existing-workspace' already exists",
    ):
        session.create_workspace("existing-workspace")

    mock_client.workspaces.assert_called_once_with(name="existing-workspace")


def test_whether_get_dataset_finds_an_existing_dataset(
    mock_argilla_client, mock_workspace, mock_dataset
):
    mock_client, _ = mock_argilla_client
    workspace = mock_workspace(name="test-workspace")
    dataset = mock_dataset(name="Q123")

    mock_client.workspaces.return_value = workspace
    mock_client.datasets.return_value = dataset

    session = ArgillaSession()
    result = session.get_dataset("Q123")

    assert result == dataset
    mock_client.datasets.assert_called_once_with(name="Q123", workspace=workspace)


def test_whether_get_dataset_raises_error_if_dataset_not_found(
    mock_argilla_client, mock_workspace
):
    mock_client, _ = mock_argilla_client
    workspace = mock_workspace(name="test-workspace")

    mock_client.workspaces.return_value = workspace
    mock_client.datasets.return_value = None

    session = ArgillaSession()

    with pytest.raises(
        ResourceDoesNotExistError,
        match="Dataset 'Q999' doesn't exist",
    ):
        session.get_dataset("Q999")


def test_whether_get_all_datasets_returns_all_datasets_in_a_workspace(
    mock_argilla_client, mock_workspace, mock_dataset
):
    """Refactored to use factory fixtures (cleaner for multiple objects)"""
    mock_client, _ = mock_argilla_client

    dataset_1 = mock_dataset(name="Q123")
    dataset_2 = mock_dataset(name="Q456")
    workspace = mock_workspace(name="test-workspace", datasets=[dataset_1, dataset_2])

    mock_client.workspaces.return_value = workspace

    session = ArgillaSession()
    result = session.get_all_datasets()

    assert len(result) == 2
    assert {dataset.name for dataset in result} == {"Q123", "Q456"}


def test_whether_create_dataset_creates_a_new_dataset(concept):
    with patch("knowledge_graph.labelling.Argilla") as mock_argilla_class:
        mock_client = MagicMock()
        mock_argilla_class.return_value = mock_client

        mock_workspace = MagicMock()
        mock_workspace.name = "test-workspace"
        mock_client.workspaces.return_value = mock_workspace
        # Mock that dataset doesn't exist initially
        mock_client.datasets.return_value = None

        with (
            patch("knowledge_graph.labelling.Settings"),
            patch("knowledge_graph.labelling.TextField"),
            patch("knowledge_graph.labelling.SpanQuestion"),
            patch("knowledge_graph.labelling.TaskDistribution"),
        ):
            with patch.object(Dataset, "create") as mock_create:
                mock_dataset = MagicMock()
                mock_dataset.name = "Q787"
                mock_create.return_value = mock_dataset

                session = ArgillaSession()
                result = session.create_dataset(concept)

                assert result == mock_dataset
                mock_create.assert_called_once()


def test_whether_create_dataset_raises_error_if_dataset_already_exists(
    mock_argilla_client, mock_workspace, mock_dataset, concept
):
    mock_client, _ = mock_argilla_client
    workspace = mock_workspace(name="test-workspace")
    existing_dataset = mock_dataset(name="Q787")

    mock_client.workspaces.return_value = workspace
    mock_client.datasets.return_value = existing_dataset

    session = ArgillaSession()

    with pytest.raises(ResourceAlreadyExistsError, match="'Q787' already exists"):
        session.create_dataset(concept)


def test_whether_create_dataset_raises_value_error_if_concept_has_no_wikibase_id(
    mock_argilla_client, mock_workspace, concept_without_a_wikibase_id
):
    mock_client, _ = mock_argilla_client
    workspace = mock_workspace()
    mock_client.workspaces.return_value = workspace

    session = ArgillaSession()

    with pytest.raises(ValueError, match="must have a Wikibase ID to create a dataset"):
        session.create_dataset(concept_without_a_wikibase_id)


def test_whether_get_user_finds_a_user_by_username(mock_argilla_client, mock_user):
    mock_client, _ = mock_argilla_client
    user = mock_user(username="alice")
    mock_client.users.return_value = user

    session = ArgillaSession()
    result = session.get_user(username="alice")

    assert result == user
    mock_client.users.assert_called_once_with(username="alice")


def test_whether_get_user_finds_a_user_by_id(mock_argilla_client, mock_user):
    mock_client, _ = mock_argilla_client
    user_id = uuid.uuid4()
    user = mock_user(username="alice", user_id=user_id)
    mock_client.users.return_value = user

    session = ArgillaSession()
    result = session.get_user(user_id=user_id)

    assert result == user
    mock_client.users.assert_called_once_with(id=user_id)


def test_whether_get_user_raises_value_error_if_neither_username_nor_id_is_provided(
    mock_argilla_client,
):
    _ = mock_argilla_client
    session = ArgillaSession()

    with pytest.raises(
        ValueError, match="One of 'username' or 'user_id' must be provided"
    ):
        session.get_user()


def test_whether_get_user_raises_value_error_if_both_username_and_id_are_provided(
    mock_argilla_client,
):
    _ = mock_argilla_client
    session = ArgillaSession()

    with pytest.raises(
        ValueError, match="Only one of 'username' or 'user_id' must be provided"
    ):
        session.get_user(username="alice", user_id=uuid.uuid4())


def test_whether_get_user_raises_error_if_user_not_found(mock_argilla_client):
    mock_client, _ = mock_argilla_client
    mock_client.users.return_value = None

    session = ArgillaSession()

    with pytest.raises(
        ResourceDoesNotExistError, match="User 'nonexistent' doesn't exist"
    ):
        session.get_user(username="nonexistent")


def test_whether_create_user_creates_a_new_user(mock_argilla_client):
    mock_client, _ = mock_argilla_client
    # Mock that the user doesn't exist
    mock_client.users.return_value = None

    with patch("knowledge_graph.labelling.User") as mock_user_class:
        mock_user = MagicMock()
        mock_user.username = "bob"
        mock_user_class.return_value = mock_user
        mock_user.create.return_value = mock_user

        session = ArgillaSession()
        result = session.create_user(
            username="bob",
            password="secure123",
            first_name="Bob",
            last_name="Smith",
        )

        assert result == mock_user
        mock_user.create.assert_called_once()


def test_whether_create_user_raises_error_if_user_already_exists(
    mock_argilla_client, mock_user
):
    mock_client, _ = mock_argilla_client

    existing_user = mock_user(username="bob")
    mock_client.users.return_value = existing_user

    session = ArgillaSession()

    with pytest.raises(ResourceAlreadyExistsError, match="'bob' already exists"):
        session.create_user(username="bob")

    mock_client.users.assert_called_once_with(username="bob")


def test_whether_add_user_to_workspace_adds_a_user_to_a_workspace(
    mock_argilla_client, mock_workspace, mock_user
):
    mock_client, _ = mock_argilla_client
    workspace = mock_workspace(name="test-workspace")
    user = mock_user(username="alice")

    mock_client.workspaces.return_value = workspace
    mock_client.users.return_value = user

    session = ArgillaSession()
    session.add_user_to_workspace("alice", "test-workspace")

    user.add_to_workspace.assert_called_once_with(workspace)


def test_whether_remove_user_from_workspace_removes_a_user_from_a_workspace(
    mock_argilla_client, mock_workspace, mock_user
):
    mock_client, _ = mock_argilla_client
    workspace = mock_workspace(name="test-workspace")
    user = mock_user(username="alice")

    mock_client.workspaces.return_value = workspace
    mock_client.users.return_value = user

    session = ArgillaSession()
    session.remove_user_from_workspace("alice", "test-workspace")

    user.remove_from_workspace.assert_called_once_with(workspace)


def test_whether_add_labelled_passages_adds_labelled_passages_to_a_dataset(
    mock_argilla_client, mock_workspace, mock_dataset, mock_user
):
    text = "Renewable energy is key to climate mitigation efforts"
    span = Span(
        text=text,
        start_index=27,
        end_index=46,
        concept_id=WikibaseID("Q123"),
        labellers=["alice"],
        timestamps=[datetime.now()],
    )
    test_passage = LabelledPassage(text=text, spans=[span])

    mock_client, _ = mock_argilla_client
    workspace = mock_workspace()
    dataset = mock_dataset(name="Q123")
    dataset.records = MagicMock()

    # Mock dataset settings with SpanQuestion
    mock_span_question = MagicMock()
    mock_span_question.labels = {"Q123": "test label"}
    dataset.settings = MagicMock()
    dataset.settings.questions = [mock_span_question]

    # Mock user lookup
    user = mock_user(username="alice")
    mock_client.users.return_value = user

    mock_client.workspaces.return_value = workspace
    mock_client.datasets.return_value = dataset

    session = ArgillaSession()
    result = session.add_labelled_passages(
        labelled_passages=[test_passage],
        wikibase_id="Q123",
    )

    assert result == dataset

    # Verify dataset.records.log was called
    dataset.records.log.assert_called_once()

    # Verify the record is a Record object with fields and responses
    logged_records = dataset.records.log.call_args[0][0]
    assert len(logged_records) == 1
    record = logged_records[0]

    # Check it's a Record object
    from argilla import Record

    assert isinstance(record, Record)

    # Check fields
    assert record.fields["text"] == text

    # Check responses
    responses_list = list(record.responses)
    assert len(responses_list) == 1
    response = responses_list[0]
    assert response.question_name == "entities"
    assert response.user_id == user.id
    assert response.status == ResponseStatus.submitted
    assert len(response.value) == 1
    assert response.value[0] == {"start": 27, "end": 46, "label": "Q123"}


def test_add_labelled_passages_with_spans_no_labellers_with_user(
    mock_argilla_client, mock_workspace, mock_dataset, mock_user
):
    """Test that spans without labellers use the user parameter"""
    text = "Climate change mitigation"
    span = Span(
        text=text,
        start_index=0,
        end_index=14,
        concept_id=WikibaseID("Q123"),
        labellers=[],  # No labellers
        timestamps=[],
    )
    test_passage = LabelledPassage(text=text, spans=[span])

    mock_client, _ = mock_argilla_client
    workspace = mock_workspace()
    dataset = mock_dataset(name="Q123")
    dataset.records = MagicMock()

    # Mock dataset settings
    mock_span_question = MagicMock()
    mock_span_question.labels = {"Q123": "test label"}
    dataset.settings = MagicMock()
    dataset.settings.questions = [mock_span_question]

    # Mock user lookup
    user = mock_user(username="bob")
    mock_client.users.return_value = user

    mock_client.workspaces.return_value = workspace
    mock_client.datasets.return_value = dataset

    session = ArgillaSession()
    result = session.add_labelled_passages(
        labelled_passages=[test_passage],
        wikibase_id="Q123",
        users=["bob"],  # Provide default users
    )

    assert result == dataset

    # Verify response was created with default user
    logged_records = dataset.records.log.call_args[0][0]
    record = logged_records[0]
    responses_list = list(record.responses)
    assert len(responses_list) == 1
    assert responses_list[0].user_id == user.id


def test_add_labelled_passages_with_spans_no_labellers_no_user(
    mock_argilla_client, mock_workspace, mock_dataset
):
    """Test that spans without labellers and no user parameter are skipped"""
    text = "Climate change mitigation"
    span = Span(
        text=text,
        start_index=0,
        end_index=14,
        concept_id=WikibaseID("Q123"),
        labellers=[],  # No labellers
        timestamps=[],
    )
    test_passage = LabelledPassage(text=text, spans=[span])

    mock_client, _ = mock_argilla_client
    workspace = mock_workspace()
    dataset = mock_dataset(name="Q123")
    dataset.records = MagicMock()

    # Mock dataset settings
    mock_span_question = MagicMock()
    mock_span_question.labels = {"Q123": "test label"}
    dataset.settings = MagicMock()
    dataset.settings.questions = [mock_span_question]

    mock_client.workspaces.return_value = workspace
    mock_client.datasets.return_value = dataset

    session = ArgillaSession()
    result = session.add_labelled_passages(
        labelled_passages=[test_passage],
        wikibase_id="Q123",
        # No user parameter provided
    )

    assert result == dataset

    # Verify no responses were created
    logged_records = dataset.records.log.call_args[0][0]
    record = logged_records[0]
    assert len(list(record.responses)) == 0


def test_add_labelled_passages_multiple_labellers_per_span(
    mock_argilla_client, mock_workspace, mock_dataset, mock_user
):
    """Test that spans with multiple labellers create separate responses"""
    text = "Climate change mitigation"
    span = Span(
        text=text,
        start_index=0,
        end_index=14,
        concept_id=WikibaseID("Q123"),
        labellers=["alice", "bob"],  # Multiple labellers
        timestamps=[datetime.now(), datetime.now()],
    )
    test_passage = LabelledPassage(text=text, spans=[span])

    mock_client, _ = mock_argilla_client
    workspace = mock_workspace()
    dataset = mock_dataset(name="Q123")
    dataset.records = MagicMock()

    # Mock dataset settings
    mock_span_question = MagicMock()
    mock_span_question.labels = {"Q123": "test label"}
    dataset.settings = MagicMock()
    dataset.settings.questions = [mock_span_question]

    # Mock user lookup - return different users
    alice_user = mock_user(username="alice", user_id=uuid.uuid4())
    bob_user = mock_user(username="bob", user_id=uuid.uuid4())

    def get_user_side_effect(username):
        if username == "alice":
            return alice_user
        elif username == "bob":
            return bob_user
        raise ResourceDoesNotExistError("User", username)

    mock_client.users.side_effect = get_user_side_effect

    mock_client.workspaces.return_value = workspace
    mock_client.datasets.return_value = dataset

    session = ArgillaSession()
    result = session.add_labelled_passages(
        labelled_passages=[test_passage],
        wikibase_id="Q123",
    )

    assert result == dataset

    # Verify two separate responses were created
    logged_records = dataset.records.log.call_args[0][0]
    record = logged_records[0]

    # Verify that the record has 2 responses
    responses_list = list(record.responses)
    assert len(responses_list) == 2

    # Verify both users are represented
    user_ids = {resp.user_id for resp in responses_list}
    assert alice_user.id in user_ids
    assert bob_user.id in user_ids


def test_add_labelled_passages_with_invalid_concept_ids(
    mock_argilla_client, mock_workspace, mock_dataset, mock_user
):
    """Test that spans with invalid concept IDs are skipped but response is still created as negative example"""
    text = "Climate change mitigation"
    span = Span(
        text=text,
        start_index=0,
        end_index=14,
        concept_id=WikibaseID("Q999"),  # Not in dataset labels
        labellers=["alice"],
        timestamps=[datetime.now()],
    )
    test_passage = LabelledPassage(text=text, spans=[span])

    mock_client, _ = mock_argilla_client
    workspace = mock_workspace()
    dataset = mock_dataset(name="Q123")
    dataset.records = MagicMock()

    # Mock dataset settings with only Q123
    mock_span_question = MagicMock()
    mock_span_question.labels = {"Q123": "test label"}  # Q999 is not valid
    dataset.settings = MagicMock()
    dataset.settings.questions = [mock_span_question]

    # Mock user lookup
    user = mock_user(username="alice")
    mock_client.users.return_value = user

    mock_client.workspaces.return_value = workspace
    mock_client.datasets.return_value = dataset

    session = ArgillaSession()
    result = session.add_labelled_passages(
        labelled_passages=[test_passage],
        wikibase_id="Q123",
    )

    assert result == dataset

    # Verify a response was created with empty spans (negative example)
    logged_records = dataset.records.log.call_args[0][0]
    record = logged_records[0]
    responses_list = list(record.responses)
    assert len(responses_list) == 1
    response = responses_list[0]
    assert response.user_id == user.id
    assert len(response.value) == 0  # No spans (negative example)


def test_add_labelled_passages_with_missing_user(
    mock_argilla_client, mock_workspace, mock_dataset
):
    """Test that spans with non-existent users are skipped"""
    text = "Climate change mitigation"
    span = Span(
        text=text,
        start_index=0,
        end_index=14,
        concept_id=WikibaseID("Q123"),
        labellers=["nonexistent"],
        timestamps=[datetime.now()],
    )
    test_passage = LabelledPassage(text=text, spans=[span])

    mock_client, _ = mock_argilla_client
    workspace = mock_workspace()
    dataset = mock_dataset(name="Q123")
    dataset.records = MagicMock()

    # Mock dataset settings
    mock_span_question = MagicMock()
    mock_span_question.labels = {"Q123": "test label"}
    dataset.settings = MagicMock()
    dataset.settings.questions = [mock_span_question]

    # Mock user lookup to raise error
    mock_client.users.side_effect = ResourceDoesNotExistError("User", "nonexistent")

    mock_client.workspaces.return_value = workspace
    mock_client.datasets.return_value = dataset

    session = ArgillaSession()
    result = session.add_labelled_passages(
        labelled_passages=[test_passage],
        wikibase_id="Q123",
    )

    assert result == dataset

    # Verify no responses were created (user not found)
    logged_records = dataset.records.log.call_args[0][0]
    record = logged_records[0]
    assert len(list(record.responses)) == 0


def test_add_labelled_passages_backward_compatibility(
    mock_argilla_client, mock_workspace, mock_dataset
):
    """Test that passages without spans continue to work (backward compatibility)"""
    text = "Climate change mitigation"
    test_passage = LabelledPassage(text=text, spans=[])  # No spans

    mock_client, _ = mock_argilla_client
    workspace = mock_workspace()
    dataset = mock_dataset(name="Q123")
    dataset.records = MagicMock()

    # Mock dataset settings
    mock_span_question = MagicMock()
    mock_span_question.labels = {"Q123": "test label"}
    dataset.settings = MagicMock()
    dataset.settings.questions = [mock_span_question]

    mock_client.workspaces.return_value = workspace
    mock_client.datasets.return_value = dataset

    session = ArgillaSession()
    result = session.add_labelled_passages(
        labelled_passages=[test_passage],
        wikibase_id="Q123",
    )

    assert result == dataset

    # Verify record was created without responses
    logged_records = dataset.records.log.call_args[0][0]
    assert len(logged_records) == 1
    record = logged_records[0]
    assert record.fields["text"] == text
    assert len(list(record.responses)) == 0


def test_whether_get_labelled_passages_returns_labelled_passages_from_a_dataset(
    mock_argilla_client, mock_workspace, mock_user
):
    mock_client, _ = mock_argilla_client
    workspace = mock_workspace()
    user_id = uuid.uuid4()
    user = mock_user(username="alice", user_id=user_id)

    mock_client.workspaces.return_value = workspace
    mock_client.users.return_value = user

    # Mock a record with one response
    mock_record = MagicMock()
    mock_record.fields = {"text": "This is test text with climate mitigation"}
    mock_record.metadata = {"document-id": "doc123"}
    mock_record.updated_at = datetime.now()
    mock_record.inserted_at = datetime.now()

    mock_response = MagicMock()
    mock_response.status = ResponseStatus.submitted
    mock_response.user_id = user_id
    mock_response.value = [{"start": 24, "end": 41, "label": "Q123"}]
    mock_record.responses = [mock_response]

    mock_dataset = MagicMock()
    mock_dataset.name = "Q123"
    mock_dataset.records.return_value = [mock_record]
    mock_client.datasets.return_value = mock_dataset

    session = ArgillaSession()
    result = session.get_labelled_passages("Q123")

    assert len(result) == 1
    passage = result[0]
    assert passage.text == "This is test text with climate mitigation"
    assert len(passage.spans) == 1
    assert passage.spans[0].start_index == 24
    assert passage.spans[0].end_index == 41
    assert passage.spans[0].concept_id == "Q123"
    assert passage.spans[0].labellers == ["alice"]


def test_whether_get_labelled_passages_returns_labelled_passages_from_a_dataset_with_multiple_responses_per_record(
    mock_argilla_client, mock_workspace
):
    mock_client, _ = mock_argilla_client
    workspace = mock_workspace()
    mock_client.workspaces.return_value = workspace

    # Mock users
    alice_id = uuid.uuid4()
    bob_id = uuid.uuid4()

    def mock_get_user(**kwargs):
        user_id = kwargs.get("id")
        if user_id == alice_id:
            user = MagicMock()
            user.username = "alice"
            return user
        elif user_id == bob_id:
            user = MagicMock()
            user.username = "bob"
            return user
        return None

    # Mock a record with one response from each labeller
    mock_record = MagicMock()
    mock_record.fields = {"text": "Climate mitigation is important"}
    mock_record.metadata = {}
    mock_record.updated_at = datetime.now()
    mock_record.inserted_at = datetime.now()

    mock_response_a = MagicMock()
    mock_response_a.status = ResponseStatus.submitted
    mock_response_a.user_id = alice_id
    mock_response_a.value = [{"start": 0, "end": 18, "label": "Q123"}]

    mock_response_b = MagicMock()
    mock_response_b.status = ResponseStatus.submitted
    mock_response_b.user_id = bob_id
    mock_response_b.value = [{"start": 0, "end": 18, "label": "Q123"}]

    mock_record.responses = [mock_response_a, mock_response_b]

    mock_dataset = MagicMock()
    mock_dataset.name = "Q123"
    mock_dataset.records.return_value = [mock_record]
    mock_client.datasets.return_value = mock_dataset
    mock_client.users.side_effect = mock_get_user

    session = ArgillaSession()
    result = session.get_labelled_passages("Q123", merge_responses_by_text=False)

    # Should create two labelled passages, one per response
    assert len(result) == 2
    assert result[0].spans[0].labellers == ["alice"]
    assert result[1].spans[0].labellers == ["bob"]


def test_whether_get_labelled_passages_respects_status_filters(
    mock_argilla_client, mock_workspace, mock_user
):
    mock_client, _ = mock_argilla_client
    workspace = mock_workspace()
    user = mock_user(username="alice")

    mock_client.workspaces.return_value = workspace
    mock_client.users.return_value = user

    # mock a record with one response from each labeller
    mock_record = MagicMock()
    mock_record.fields = {"text": "Test text"}
    mock_record.metadata = {}
    mock_record.updated_at = datetime.now()
    mock_record.inserted_at = datetime.now()

    # a submitted response
    mock_response_a = MagicMock()
    mock_response_a.status = ResponseStatus.submitted
    mock_response_a.user_id = uuid.uuid4()
    mock_response_a.value = [{"start": 0, "end": 4, "label": "Q123"}]

    # a draft response, which should be filtered out
    mock_response_b = MagicMock()
    mock_response_b.status = ResponseStatus.draft
    mock_response_b.user_id = uuid.uuid4()
    mock_response_b.value = [{"start": 5, "end": 9, "label": "Q123"}]

    mock_record.responses = [mock_response_a, mock_response_b]

    mock_dataset = MagicMock()
    mock_dataset.name = "Q123"
    mock_dataset.records.return_value = [mock_record]
    mock_client.datasets.return_value = mock_dataset

    session = ArgillaSession()
    result = session.get_labelled_passages("Q123")

    # by default we should only get the submitted response
    assert len(result) == 1

    # if we specify the statuses, we should get the responses with those statuses
    result = session.get_labelled_passages(
        "Q123", include_statuses=[ResponseStatus.submitted]
    )
    assert len(result) == 1

    result = session.get_labelled_passages(
        "Q123",
        include_statuses=[ResponseStatus.submitted, ResponseStatus.draft],
        merge_responses_by_text=False,
    )
    assert len(result) == 2


def test_whether_get_labelled_passages_respects_limit_parameter(
    mock_argilla_client, mock_workspace
):
    mock_client, _ = mock_argilla_client
    workspace = mock_workspace()
    mock_client.workspaces.return_value = workspace

    mock_dataset = MagicMock()
    mock_dataset.name = "Q123"
    mock_client.datasets.return_value = mock_dataset

    session = ArgillaSession()
    session.get_labelled_passages("Q123", limit=10)

    # Verify limit was passed to dataset.records()
    mock_dataset.records.assert_called_once_with(with_responses=True, limit=10)


# Hypothesis strategies for testing the metadata formatting function
metadata_key_strategy = st.text(
    alphabet=st.characters(
        whitelist_categories=(
            # https://en.wikipedia.org/wiki/Unicode_character_property#General_Category
            "Lu",  # Letter, uppercase
            "Ll",  # Letter, lowercase
        ),
        whitelist_characters="._- ",
    ),
    min_size=1,
    max_size=50,
).filter(lambda x: x and not x.isspace())

metadata_value_strategy = st.one_of(
    st.text(min_size=0, max_size=100),
    st.integers(),
    st.floats(allow_nan=False, allow_infinity=False),
    st.booleans(),
)

metadata_dict_strategy = st.dictionaries(
    keys=metadata_key_strategy,
    values=metadata_value_strategy,
    min_size=0,
    max_size=10,
)


@given(metadata_dict_strategy)
def test_whether_format_metadata_lowercases_keys(metadata):
    with patch("knowledge_graph.labelling.Argilla"):
        session = ArgillaSession()
        result = session._format_metadata_keys_for_argilla(metadata)
        for key in result.keys():
            assert key == key.lower(), f"Key '{key}' is not lowercase"


@given(metadata_dict_strategy)
def test_whether_format_metadata_preserves_values(metadata):
    with patch("knowledge_graph.labelling.Argilla"):
        session = ArgillaSession()
        result = session._format_metadata_keys_for_argilla(metadata)
        original_values = sorted(str(v) for v in metadata.values())
        result_values = sorted(str(v) for v in result.values())
        assert original_values == result_values


@given(metadata_dict_strategy)
def test_whether_format_metadata_property_idempotent(metadata):
    with patch("knowledge_graph.labelling.Argilla"):
        session = ArgillaSession()
        result_once = session._format_metadata_keys_for_argilla(metadata)
        result_twice = session._format_metadata_keys_for_argilla(result_once)
        assert result_once == result_twice


@given(metadata_dict_strategy)
def test_whether_format_metadata_replaces_dots_with_hyphens(metadata):
    with patch("knowledge_graph.labelling.Argilla"):
        session = ArgillaSession()
        result = session._format_metadata_keys_for_argilla(metadata)
        for key in result.keys():
            assert "." not in key, f"Key '{key}' contains a dot"
