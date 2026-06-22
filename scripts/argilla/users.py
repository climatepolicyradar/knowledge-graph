"""
Manage Argilla users.

Two commands:

- `create`: interactively create a single user as an annotator or an owner.
  Annotators are also assigned to the knowledge-graph workspace.
- `list`: print every Argilla user with their name, role and workspaces.

Argilla owner credentials are pulled from SSM, so you must be authenticated to
AWS (the default credentials / profile is used) when running this script.
"""

import os
import re
import secrets

import typer
from argilla._models import Role
from cpr_sdk.ssm import get_aws_ssm_param
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from knowledge_graph.labelling import (
    ArgillaSession,
    ResourceAlreadyExistsError,
    ResourceDoesNotExistError,
)

app = typer.Typer()
console = Console()

WORKSPACE_NAME = "knowledge-graph"

# Roles a user can be created with via this script.
ROLE_CHOICES = [Role.annotator.value, Role.owner.value]

# Argilla only accepts usernames of letters, digits, hyphens and underscores
# that don't start with a hyphen or underscore.
USERNAME_PATTERN = re.compile(r"^(?!-|_)[A-Za-z0-9-_]+$")


def _connect() -> ArgillaSession:
    """
    Connect to Argilla using owner credentials pulled from SSM.

    Sets the ARGILLA_API_URL / ARGILLA_API_KEY environment variables from SSM
    (so they're picked up by ArgillaSession) and returns a connected session.
    Requires AWS authentication for SSM access.
    """
    with console.status("Fetching Argilla credentials from SSM..."):
        os.environ["ARGILLA_API_URL"] = get_aws_ssm_param("/Argilla/APIURL")
        os.environ["ARGILLA_API_KEY"] = get_aws_ssm_param("/Argilla/Owner/APIKey")

    with console.status("Connecting to Argilla..."):
        session = ArgillaSession()
    console.log("✅ Connected to Argilla")
    return session


@app.command("create")
def create():
    """Interactively create a single Argilla user (annotator or owner)."""
    argilla = _connect()

    username = Prompt.ask("Username").strip()
    if not username:
        raise typer.BadParameter("Username cannot be empty")
    if not USERNAME_PATTERN.match(username):
        raise typer.BadParameter(
            f"Invalid username '{username}'. Usernames may only contain letters, "
            "digits, hyphens and underscores (no dots or spaces), and cannot start "
            "with a hyphen or underscore. E.g. use 'silke-vanbeselaere'."
        )

    role_name = Prompt.ask(
        "Role",
        choices=ROLE_CHOICES,
        default=Role.annotator.value,
    )
    role = Role(role_name)

    first_name = Prompt.ask("First name (optional)", default="").strip() or None
    last_name = Prompt.ask("Last name (optional)", default="").strip() or None

    password = Prompt.ask(
        "Password (leave blank to auto-generate)",
        password=True,
        default="",
    ).strip()

    if password_was_generated := not password:
        password = secrets.token_urlsafe(16)
    else:
        confirm = Prompt.ask("Confirm password", password=True).strip()
        if confirm != password:
            raise typer.BadParameter("Passwords do not match")

    # Make sure the knowledge-graph workspace exists before assigning (annotators only).
    workspace = None
    if role == Role.annotator:
        try:
            workspace = argilla.get_workspace(WORKSPACE_NAME)
        except ResourceDoesNotExistError:
            console.log(f"Workspace '{WORKSPACE_NAME}' not found, creating it")
            workspace = argilla.create_workspace(WORKSPACE_NAME)

    try:
        # Use the returned User object directly for the workspace assignment: the
        # session caches username lookups, so a fresh get_user() after creation
        # would return a stale "not found" result.
        user = argilla.create_user(
            username=username,
            password=password,
            first_name=first_name,
            last_name=last_name,
            role=role,
        )
        console.log(f"✅ Created user '{username}' with role '{role_name}'")
    except ResourceAlreadyExistsError:
        console.print(
            f"[yellow]User '{username}' already exists — "
            "ensuring workspace membership only.[/yellow]"
        )
        user = argilla.get_user(username=username)
        password_was_generated = False

    if role == Role.annotator and workspace is not None:
        members = argilla.client.users.list(workspace=workspace)
        if any(member.username == username for member in members):
            console.log(f"'{username}' is already in workspace '{WORKSPACE_NAME}'")
        else:
            user.add_to_workspace(workspace)
            console.log(f"✅ Added '{username}' to workspace '{WORKSPACE_NAME}'")

    console.print()
    console.print(f"[bold green]User:[/bold green] {username}")
    console.print(f"  Role:      {role_name}")
    if role == Role.annotator:
        console.print(f"  Workspace: {WORKSPACE_NAME}")
    if password_was_generated:
        console.print(
            f"  Password:  [bold]{password}[/bold] "
            "(auto-generated — copy it now, it won't be shown again)"
        )


@app.command("list")
def list_users():
    """List all Argilla users with their name, role and workspaces."""
    argilla = _connect()

    with console.status("Loading users and workspaces..."):
        # Build a username -> [workspace names] map from workspace membership.
        workspaces_by_user: dict[str, list[str]] = {}
        for workspace in argilla.client.workspaces:
            if not workspace.name:
                continue
            for member in argilla.client.users.list(workspace=workspace):
                workspaces_by_user.setdefault(member.username, []).append(
                    workspace.name
                )

        users = argilla.client.users.list()

    table = Table(title="Argilla users")
    table.add_column("Username", style="bold")
    table.add_column("Name")
    table.add_column("Role")
    table.add_column("Workspaces")

    for user in sorted(users, key=lambda u: u.username):
        name = " ".join(filter(None, [user.first_name, user.last_name])).strip()
        workspaces = ", ".join(sorted(workspaces_by_user.get(user.username, [])))
        table.add_row(user.username, name, str(user.role), workspaces)

    console.print(table)
    console.print(f"{len(users)} user(s) total")


if __name__ == "__main__":
    app()
