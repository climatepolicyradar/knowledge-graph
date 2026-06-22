from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from knowledge_graph.labelling import (
    ResourceAlreadyExistsError,
    ResourceDoesNotExistError,
)
from scripts.argilla.users import WORKSPACE_NAME, app

runner = CliRunner()


@pytest.fixture
def mock_session():
    session = MagicMock()
    return session


@pytest.fixture
def mock_connect(mock_session):
    with patch("scripts.argilla.users._connect", return_value=mock_session):
        yield mock_session


# ---------------------------------------------------------------------------
# create
# ---------------------------------------------------------------------------


def test_create_annotator_creates_user_and_adds_to_workspace(
    mock_connect, mock_workspace, mock_user
):
    session = mock_connect
    workspace = mock_workspace(name=WORKSPACE_NAME)
    user = mock_user(username="alice")
    session.get_workspace.return_value = workspace
    session.create_user.return_value = user
    session.client.users.list.return_value = []

    with patch(
        "scripts.argilla.users.Prompt.ask",
        side_effect=["alice", "annotator", "", "", ""],
    ):
        result = runner.invoke(app, ["create"])

    assert result.exit_code == 0
    session.create_user.assert_called_once()
    user.add_to_workspace.assert_called_once_with(workspace)


def test_create_with_blank_password_auto_generates_and_shows_it(
    mock_connect, mock_workspace, mock_user
):
    session = mock_connect
    workspace = mock_workspace(name=WORKSPACE_NAME)
    user = mock_user(username="carol")
    session.get_workspace.return_value = workspace
    session.create_user.return_value = user
    session.client.users.list.return_value = []

    with patch(
        "scripts.argilla.users.Prompt.ask",
        side_effect=["carol", "annotator", "", "", ""],
    ):
        result = runner.invoke(app, ["create"])

    assert result.exit_code == 0
    assert "auto-generated" in result.output


def test_create_rejects_mismatched_passwords(mock_connect):
    with patch(
        "scripts.argilla.users.Prompt.ask",
        side_effect=["eve", "annotator", "", "", "pass1", "pass2"],
    ):
        result = runner.invoke(app, ["create"])

    assert result.exit_code != 0


def test_create_rejects_empty_username(mock_connect):
    with patch("scripts.argilla.users.Prompt.ask", side_effect=[""]):
        result = runner.invoke(app, ["create"])

    assert result.exit_code != 0


def test_create_when_user_already_exists_ensures_workspace_membership(
    mock_connect, mock_workspace, mock_user
):
    """Running the CLI should still add an existing user to the relevant workspaces."""
    session = mock_connect
    workspace = mock_workspace(name=WORKSPACE_NAME)
    existing = mock_user(username="frank")
    session.get_workspace.return_value = workspace
    session.create_user.side_effect = ResourceAlreadyExistsError("User", "frank")
    session.get_user.return_value = existing
    session.client.users.list.return_value = []  # not yet a member

    with patch(
        "scripts.argilla.users.Prompt.ask",
        side_effect=["frank", "annotator", "", "", ""],
    ):
        result = runner.invoke(app, ["create"])

    assert result.exit_code == 0
    session.get_user.assert_called_once_with(username="frank")
    existing.add_to_workspace.assert_called_once_with(workspace)


def test_create_annotator_creates_workspace_when_it_does_not_exist(
    mock_connect, mock_workspace, mock_user
):
    session = mock_connect
    new_workspace = mock_workspace(name=WORKSPACE_NAME)
    user = mock_user(username="grace")
    session.get_workspace.side_effect = ResourceDoesNotExistError(
        "Workspace", WORKSPACE_NAME
    )
    session.create_workspace.return_value = new_workspace
    session.create_user.return_value = user
    session.client.users.list.return_value = []

    with patch(
        "scripts.argilla.users.Prompt.ask",
        side_effect=["grace", "annotator", "", "", ""],
    ):
        result = runner.invoke(app, ["create"])

    assert result.exit_code == 0
    session.create_workspace.assert_called_once_with(WORKSPACE_NAME)
    user.add_to_workspace.assert_called_once_with(new_workspace)


def test_create_skips_workspace_add_when_user_already_a_member(
    mock_connect, mock_workspace, mock_user
):
    session = mock_connect
    workspace = mock_workspace(name=WORKSPACE_NAME)
    user = mock_user(username="heidi")
    existing_member = MagicMock()
    existing_member.username = "heidi"
    session.get_workspace.return_value = workspace
    session.create_user.return_value = user
    session.client.users.list.return_value = [existing_member]

    with patch(
        "scripts.argilla.users.Prompt.ask",
        side_effect=["heidi", "annotator", "", "", ""],
    ):
        result = runner.invoke(app, ["create"])

    assert result.exit_code == 0
    user.add_to_workspace.assert_not_called()


# ---------------------------------------------------------------------------
# list
# ---------------------------------------------------------------------------


def test_list_shows_all_users(mock_connect):
    session = mock_connect
    session.client.workspaces = []

    user_a = MagicMock()
    user_a.username = "alice"
    user_a.first_name = "Alice"
    user_a.last_name = "Smith"
    user_a.role = "annotator"

    user_b = MagicMock()
    user_b.username = "bob"
    user_b.first_name = "Bob"
    user_b.last_name = "Jones"
    user_b.role = "owner"

    session.client.users.list.return_value = [user_a, user_b]

    result = runner.invoke(app, ["list"])

    assert result.exit_code == 0
    assert "alice" in result.output
    assert "bob" in result.output


def test_list_shows_workspace_memberships(mock_connect, mock_workspace):
    session = mock_connect
    ws = mock_workspace(name="knowledge-graph")
    ws.name = "knowledge-graph"
    session.client.workspaces = [ws]

    member = MagicMock()
    member.username = "ivan"
    member.first_name = "Ivan"
    member.last_name = ""
    member.role = "annotator"

    def list_side_effect(*args, workspace=None):
        if workspace is ws:
            return [member]
        return [member]  # full user list

    session.client.users.list.side_effect = list_side_effect

    result = runner.invoke(app, ["list"])

    assert result.exit_code == 0
    assert "knowledge-graph" in result.output
