import os
import subprocess
import tempfile
from datetime import timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from pydantic import SecretStr

from flows.create_classifiers_specs_pr import (
    GitCliOps,
    GitPyOps,
    _run_subprocess_with_error_logging,
    commit_and_create_pr,
    create_and_merge_pr,
    enable_auto_merge,
    extract_pr_details,
    wait_for_pr_merge,
)
from flows.result import Err, Error, Ok, is_err, is_ok, unwrap_err
from knowledge_graph.cloud import AwsEnv


def test_run_subprocess_with_error_logging__success():
    """Test _run_subprocess_with_error_logging with a successful command."""
    with patch("subprocess.run") as mock_run:
        mock_result = Mock(returncode=0, stdout="output", stderr="")
        mock_run.return_value = mock_result

        result = _run_subprocess_with_error_logging(
            cmd=["echo", "test"], cwd=Path("/tmp"), check=True
        )

        assert result == mock_result
        mock_run.assert_called_once_with(
            ["echo", "test"],
            cwd=Path("/tmp"),
            capture_output=True,
            text=True,
            check=True,
        )


def test_run_subprocess_with_error_logging__failure_with_stdout_and_stderr():
    """Test _run_subprocess_with_error_logging logs both stdout and stderr on failure."""
    with (
        patch("subprocess.run") as mock_run,
        patch("flows.create_classifiers_specs_pr.get_logger") as mock_get_logger,
    ):
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        error = subprocess.CalledProcessError(
            returncode=1,
            cmd=["git", "status"],
            output="some output",
            stderr="some error",
        )
        error.stdout = "some output"
        error.stderr = "some error"
        mock_run.side_effect = error

        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            _run_subprocess_with_error_logging(
                cmd=["git", "status"], cwd=Path("/tmp"), check=True
            )

        assert exc_info.value == error
        mock_logger.error.assert_any_call(
            "Command `git status` failed with exit code 1"
        )
        mock_logger.error.assert_any_call("STDOUT: some output")
        mock_logger.error.assert_any_call("STDERR: some error")
        assert mock_logger.error.call_count == 3


def test_run_subprocess_with_error_logging__failure_with_only_stderr():
    """Test _run_subprocess_with_error_logging logs only stderr when stdout is empty."""
    with (
        patch("subprocess.run") as mock_run,
        patch("flows.create_classifiers_specs_pr.get_logger") as mock_get_logger,
    ):
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        error = subprocess.CalledProcessError(
            returncode=1,
            cmd=["git", "push"],
            output="",
            stderr="Permission denied",
        )
        error.stdout = ""
        error.stderr = "Permission denied"
        mock_run.side_effect = error

        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            _run_subprocess_with_error_logging(
                cmd=["git", "push"], cwd=Path("/tmp"), check=True
            )

        assert exc_info.value == error
        mock_logger.error.assert_any_call("Command `git push` failed with exit code 1")
        mock_logger.error.assert_any_call("STDERR: Permission denied")
        # Should only log stderr, not stdout
        assert mock_logger.error.call_count == 2


def test_run_subprocess_with_error_logging__failure_with_only_stdout():
    """Test _run_subprocess_with_error_logging logs only stdout when stderr is empty."""
    with (
        patch("subprocess.run") as mock_run,
        patch("flows.create_classifiers_specs_pr.get_logger") as mock_get_logger,
    ):
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        error = subprocess.CalledProcessError(
            returncode=2,
            cmd=["make", "build"],
            output="Build output here",
            stderr="",
        )
        error.stdout = "Build output here"
        error.stderr = ""
        mock_run.side_effect = error

        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            _run_subprocess_with_error_logging(
                cmd=["make", "build"], cwd=Path("/tmp"), check=True
            )

        assert exc_info.value == error
        mock_logger.error.assert_any_call(
            "Command `make build` failed with exit code 2"
        )
        mock_logger.error.assert_any_call("STDOUT: Build output here")
        # Should only log stdout, not stderr
        assert mock_logger.error.call_count == 2


def test_run_subprocess_with_error_logging__failure_with_no_output():
    """Test _run_subprocess_with_error_logging logs only command failure when no output."""
    with (
        patch("subprocess.run") as mock_run,
        patch("flows.create_classifiers_specs_pr.get_logger") as mock_get_logger,
    ):
        mock_logger = Mock()
        mock_get_logger.return_value = mock_logger

        error = subprocess.CalledProcessError(
            returncode=127,
            cmd=["nonexistent", "command"],
            output="",
            stderr="",
        )
        error.stdout = ""
        error.stderr = ""
        mock_run.side_effect = error

        with pytest.raises(subprocess.CalledProcessError) as exc_info:
            _run_subprocess_with_error_logging(
                cmd=["nonexistent", "command"], cwd=Path("/tmp"), check=True
            )

        assert exc_info.value == error
        mock_logger.error.assert_called_once_with(
            "Command `nonexistent command` failed with exit code 127"
        )


def test_run_subprocess_with_error_logging__check_false():
    """Test _run_subprocess_with_error_logging with check=False doesn't raise."""
    with patch("subprocess.run") as mock_run:
        mock_result = Mock(returncode=1, stdout="", stderr="error")
        mock_run.return_value = mock_result

        result = _run_subprocess_with_error_logging(
            cmd=["exit", "1"], cwd=Path("/tmp"), check=False
        )

        assert result == mock_result
        mock_run.assert_called_once_with(
            ["exit", "1"],
            cwd=Path("/tmp"),
            capture_output=True,
            text=True,
            check=False,
        )


@pytest.mark.asyncio
async def test_commit_and_create_pr__with_changes():
    """Test commit_and_create_pr when there are changes to commit."""
    with patch("subprocess.run") as mock_run:
        # Create mock result objects with proper stdout attributes
        git_success = Mock()
        git_success.stdout = ""
        git_success.returncode = 0

        gh_pr_result = Mock()
        gh_pr_result.stdout = (
            "https://github.com/climatepolicyradar/knowledge-graph/pull/123"
        )
        gh_pr_result.returncode = 0

        # Mock subprocess.run for gh commands only
        mock_run.side_effect = [
            git_success,  # gh auth setup-git
            git_success,  # git remote set-url
            gh_pr_result,  # gh pr create
        ]

        repo_path_mock = Mock()
        mock_git = Mock()
        mock_git.status_porcelain.return_value = "M testfile\n"

        pr_number = await commit_and_create_pr(
            file_path="testfile",
            commit_message="Update testfile",
            pr_title="Update testfile",
            pr_body="Automated update",
            git=mock_git,
            repo="climatepolicyradar/knowledge-graph",
            base_branch="main",
            repo_path=repo_path_mock,
        )

        assert pr_number == 123
        # Verify git operations were called
        mock_git.status_porcelain.assert_called_once_with("testfile")
        mock_git.config.assert_any_call("user.email", "tech@climatepolicyradar.org")
        mock_git.config.assert_any_call("user.name", "cpr-tech-admin")
        mock_git.checkout_new_branch.assert_called_once()
        mock_git.add.assert_called_once_with("testfile")
        mock_git.commit.assert_called_once_with("Update testfile")
        mock_git.push.assert_called_once()


@pytest.mark.asyncio
async def test_commit_and_create_pr__no_changes():
    """Test commit_and_create_pr when there are no changes."""
    repo_path_mock = Mock()
    mock_git = Mock()
    mock_git.status_porcelain.return_value = ""

    pr_number = await commit_and_create_pr(
        file_path="testfile",
        commit_message="Update testfile",
        pr_title="Update testfile",
        pr_body="Automated update",
        git=mock_git,
        repo="climatepolicyradar/knowledge-graph",
        base_branch="main",
        repo_path=repo_path_mock,
    )

    assert pr_number is None
    # Verify status was checked but no other git operations happened
    mock_git.status_porcelain.assert_called_once_with("testfile")
    mock_git.config.assert_not_called()
    mock_git.checkout_new_branch.assert_not_called()
    mock_git.add.assert_not_called()
    mock_git.commit.assert_not_called()
    mock_git.push.assert_not_called()


@pytest.mark.asyncio
async def test_enable_auto_merge():
    """Test enabling auto-merge on a PR."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = Mock()
        pr_number = 123
        result = await enable_auto_merge(
            pr_number=pr_number,
            repo="climatepolicyradar/knowledge-graph",
            merge_method="squash",
        )

        from pathlib import Path

        mock_run.assert_called_once_with(
            [
                "gh",
                "pr",
                "merge",
                "123",
                "--squash",
                "--auto",
                "--repo",
                "climatepolicyradar/knowledge-graph",
            ],
            cwd=Path("./"),
            capture_output=True,
            text=True,
            check=True,
        )
        assert result == Ok(pr_number)


@pytest.mark.asyncio
async def test_enable_auto_merge__exception():
    """Test enabling auto-merge on a PR where exception is raised."""
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = Exception("GitHub CLI error")
        pr_number = 123
        result = await enable_auto_merge(
            pr_number=pr_number,
            repo="climatepolicyradar/knowledge-graph",
            merge_method="squash",
        )

        from pathlib import Path

        mock_run.assert_called_once_with(
            [
                "gh",
                "pr",
                "merge",
                "123",
                "--squash",
                "--auto",
                "--repo",
                "climatepolicyradar/knowledge-graph",
            ],
            cwd=Path("./"),
            capture_output=True,
            text=True,
            check=True,
        )
        assert is_err(result)
        error = unwrap_err(result)
        assert error.msg == "Failed to enable auto-merge for PR."
        assert error.metadata.get("pr_number") == pr_number


@pytest.mark.asyncio
async def test_wait_for_pr_merge():
    """Test waiting for a PR to merge with success."""
    with patch("subprocess.run") as mock_run:
        # Mock subprocess.run for gh pr view
        mock_run.side_effect = [
            Mock(
                stdout='{"state": "OPEN", "mergedAt": null}', returncode=0
            ),  # First poll
            Mock(
                stdout='{"state": "MERGED", "mergedAt": "2023-01-01T00:00:00Z"}',
                returncode=0,
            ),  # Second poll
        ]

        pr_number = 123
        result = await wait_for_pr_merge(
            pr_number=pr_number,
            repo="climatepolicyradar/knowledge-graph",
            timeout=timedelta(seconds=1),
            poll_interval=timedelta(milliseconds=100),
        )

        assert mock_run.call_count == 2  # merged by 2nd call
        mock_run.assert_any_call(
            [
                "gh",
                "pr",
                "view",
                "123",
                "--json",
                "state,mergedAt",
                "--repo",
                "climatepolicyradar/knowledge-graph",
            ],
            capture_output=True,
            text=True,
        )
        assert result == Ok(pr_number)


@pytest.mark.asyncio
async def test_wait_for_pr_merge__timeout():
    """Test waiting for a PR to merge with error from timeout."""
    with patch("subprocess.run") as mock_run:
        # Mock subprocess.run for gh pr view
        mock_run.return_value = Mock(
            stdout='{"state": "OPEN", "mergedAt": null}', returncode=0
        )

        result = await wait_for_pr_merge(
            pr_number=123,
            repo="climatepolicyradar/knowledge-graph",
            timeout=timedelta(seconds=0),
            poll_interval=timedelta(milliseconds=100),
        )

        assert is_err(result)
        assert (
            "TimeoutError: PR did not merge within the timeout period."
            in unwrap_err(result).msg
        )
        assert unwrap_err(result).metadata


@pytest.mark.asyncio
async def test_wait_for_pr_merge__closed():
    """Test waiting for a PR to merge with error from closing without merging."""
    with patch("subprocess.run") as mock_run:
        # Mock subprocess.run for gh pr view
        mock_run.return_value = Mock(
            stdout='{"state": "CLOSED", "mergedAt": null}', returncode=0
        )

        pr_number = 123
        result = await wait_for_pr_merge(
            pr_number=pr_number,
            repo="climatepolicyradar/knowledge-graph",
            timeout=timedelta(minutes=1),
            poll_interval=timedelta(milliseconds=100),
        )

        assert is_err(result)
        assert "RuntimeError: PR was closed without merging." in unwrap_err(result).msg
        assert unwrap_err(result).metadata.get("pr_number") == pr_number
        assert unwrap_err(result).metadata.get("pr_state") == "CLOSED"


@pytest.mark.asyncio
async def test_wait_for_pr_merge__failed_to_get_pr_timeout():
    """Test waiting for a PR to merge with timeout error from when non-zero returncode."""
    with patch("subprocess.run") as mock_run:
        # Mock subprocess.run for gh pr view
        # Returncode 1 simulates failure to get PR info until timeout
        pr_number = 123
        mock_run.return_value = Mock(
            stdout='{"state": "OPEN", "mergedAt": null}', returncode=1
        )

        result = await wait_for_pr_merge(
            pr_number=pr_number,
            repo="climatepolicyradar/knowledge-graph",
            timeout=timedelta(milliseconds=100),
            poll_interval=timedelta(milliseconds=200),
        )

        assert is_err(result)
        assert (
            "TimeoutError: PR did not merge within the timeout period."
            in unwrap_err(result).msg
        )
        assert unwrap_err(result).metadata.get("pr_number") == pr_number


@pytest.mark.asyncio
async def test_create_and_merge_pr():
    """Test the workflow with success."""
    with (
        patch("flows.create_classifiers_specs_pr.commit_and_create_pr") as mock_commit,
        patch(
            "flows.create_classifiers_specs_pr.enable_auto_merge"
        ) as mock_enable_merge,
        patch("flows.create_classifiers_specs_pr.wait_for_pr_merge") as mock_wait_merge,
    ):
        # Mock commit_and_create_pr
        mock_commit.return_value = 123

        # Mock enable_auto_merge
        mock_enable_merge.return_value = Ok(None)

        # Mock wait_for_pr_merge
        mock_wait_merge.return_value = Ok(None)
        # Call the main function
        results = await create_and_merge_pr(
            spec_file="testfile",
            aws_env=AwsEnv.staging,
            flow_run_name="Test Run",
            flow_run_url="http://example.com",
            github_token=SecretStr("mock-token"),
            auto_merge=True,
        )

        mock_commit.assert_called_once()
        mock_enable_merge.assert_called_once_with(
            pr_number=123,
            repo="climatepolicyradar/knowledge-graph",
            merge_method="REBASE",
        )
        mock_wait_merge.assert_called_once_with(
            pr_number=123,
            repo="climatepolicyradar/knowledge-graph",
            timeout=timedelta(minutes=30),
            poll_interval=timedelta(seconds=30),
        )
        assert os.environ["GITHUB_TOKEN"] == "mock-token"
        assert is_ok(results)


@pytest.mark.asyncio
async def test_create_and_merge_pr__no_automerge():
    """Test the workflow with success when automerge set to false."""
    with (
        patch("flows.create_classifiers_specs_pr.commit_and_create_pr") as mock_commit,
        patch(
            "flows.create_classifiers_specs_pr.enable_auto_merge"
        ) as mock_enable_merge,
        patch("flows.create_classifiers_specs_pr.wait_for_pr_merge") as mock_wait_merge,
    ):
        # Mock commit_and_create_pr
        mock_commit.return_value = 123

        # Call the main function
        results = await create_and_merge_pr(
            spec_file="testfile",
            aws_env=AwsEnv.staging,
            flow_run_name="Test Run",
            flow_run_url="http://example.com",
            github_token=SecretStr("mock-token"),
            auto_merge=False,
        )

        mock_commit.assert_called_once()
        mock_enable_merge.assert_not_called()
        mock_wait_merge.assert_not_called()

        assert is_ok(results)


@pytest.mark.asyncio
async def test_create_and_merge_pr__automerge_failure():
    """Test the workflow when automerge fails."""
    with (
        patch("flows.create_classifiers_specs_pr.commit_and_create_pr") as mock_commit,
        patch(
            "flows.create_classifiers_specs_pr.enable_auto_merge"
        ) as mock_enable_merge,
        patch("flows.create_classifiers_specs_pr.wait_for_pr_merge") as mock_wait_merge,
    ):
        # Mock commit_and_create_pr
        mock_commit.return_value = 123

        # Mock enable_auto_merge failure
        mock_enable_merge.return_value = Err(Error(msg="Test error", metadata={}))

        # Call the main function
        results = await create_and_merge_pr(
            spec_file="testfile",
            aws_env=AwsEnv.staging,
            flow_run_name="Test Run",
            flow_run_url="http://example.com",
            github_token=SecretStr("mock-token"),
            auto_merge=True,
        )

        mock_commit.assert_called_once()
        mock_enable_merge.assert_called_once_with(
            pr_number=123,
            repo="climatepolicyradar/knowledge-graph",
            merge_method="REBASE",
        )
        mock_wait_merge.assert_not_called()

        errors = unwrap_err(results)
        assert is_err(results)
        assert "Test error" in errors.msg


def test_extract_pr_details_valid_url():
    """Test extracting PR details from a valid URL."""
    result_str = "https://github.com/climatepolicyradar/knowledge-graph/pull/123"
    pr_number, pr_url = extract_pr_details(result_str)

    assert pr_number == 123
    assert pr_url == result_str


def test_extract_pr_details_invalid_url():
    """Test extracting PR details from an invalid URL."""
    result_str = "https://github.com/climatepolicyradar/knowledge-graph/pull/abc"
    import re

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Failed to extract PR details: invalid literal for int() with base 10: 'abc'"
        ),
    ):
        extract_pr_details(result_str)


def test_extract_pr_details_empty_string():
    """Test extracting PR details from an empty string."""
    result_str = ""

    with pytest.raises(ValueError, match="The result string is empty."):
        extract_pr_details(result_str)


@pytest.mark.asyncio
async def test_create_and_merge_pr__github_token_exception():
    """Test the workflow when setting GitHub token raises an exception."""
    with patch("flows.create_classifiers_specs_pr.commit_and_create_pr") as mock_commit:
        # Call the main function with a SecretStr that raises an exception
        github_token_mock = Mock(SecretStr("mock-token"))
        github_token_mock.get_secret_value.side_effect = Exception("Token error")

        results = await create_and_merge_pr(
            spec_file="testfile",
            aws_env=AwsEnv.staging,
            flow_run_name="Test Run",
            flow_run_url="http://example.com",
            github_token=github_token_mock,
            auto_merge=True,
        )

        mock_commit.assert_not_called()

        errors = unwrap_err(results)
        assert is_err(results)
        assert "Failed to set GitHub token environment var." in errors.msg


@pytest.fixture
def temp_git_repo():
    """Create a temporary git repository for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)

        # Initialise git repo
        subprocess.run(
            ["git", "init"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "Test User"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )

        # Create initial commit
        test_file = repo_path / "test.txt"
        test_file.write_text("initial content")
        subprocess.run(
            ["git", "add", "test.txt"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "Initial commit"],
            cwd=repo_path,
            capture_output=True,
            check=True,
        )

        yield repo_path


@pytest.fixture(params=["cli", "gitpython"])
def git_ops(request, temp_git_repo):
    """Parameterised fixture providing both GitOps implementations."""
    if request.param == "cli":
        return GitCliOps(temp_git_repo)
    else:
        return GitPyOps(temp_git_repo)


def test_gitops_config(git_ops, temp_git_repo):
    """Test that both implementations set git config correctly."""
    git_ops.config("user.test", "test-value")

    # Verify using git CLI
    result = subprocess.run(
        ["git", "config", "user.test"],
        cwd=temp_git_repo,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "test-value"


def test_gitops_status_porcelain_no_changes(git_ops, temp_git_repo):
    """Test that both implementations return empty status for unchanged files."""
    status = git_ops.status_porcelain("test.txt")
    assert status.strip() == ""


def test_gitops_status_porcelain_with_changes(git_ops, temp_git_repo):
    """Test that both implementations detect file changes."""
    # Modify the file
    test_file = temp_git_repo / "test.txt"
    test_file.write_text("modified content")

    status = git_ops.status_porcelain("test.txt")
    assert status.strip() != ""
    assert "test.txt" in status


def test_gitops_status_porcelain_new_file(git_ops, temp_git_repo):
    """Test that both implementations detect new files."""
    # Create a new file
    new_file = temp_git_repo / "new.txt"
    new_file.write_text("new content")

    status = git_ops.status_porcelain("new.txt")
    assert status.strip() != ""
    assert "new.txt" in status


def test_gitops_checkout_new_branch(git_ops, temp_git_repo):
    """Test that both implementations create and checkout new branches."""
    branch_name = "test-branch"
    git_ops.checkout_new_branch(branch_name)

    # Verify using git CLI
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=temp_git_repo,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == branch_name


def test_gitops_add(git_ops, temp_git_repo):
    """Test that both implementations stage files correctly."""
    # Create a new file
    new_file = temp_git_repo / "add-test.txt"
    new_file.write_text("content to add")

    git_ops.add("add-test.txt")

    # Verify using git CLI
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=temp_git_repo,
        capture_output=True,
        text=True,
        check=True,
    )
    # Staged files show as "A " in porcelain format
    assert "A  add-test.txt" in result.stdout or "A add-test.txt" in result.stdout


def test_cli_gitops_checks_git_installed():
    """Test that CliGitOps checks for git CLI installation."""
    with patch("subprocess.run") as mock_run:
        # Simulate git not being installed (FileNotFoundError)
        mock_run.side_effect = FileNotFoundError("git not found")

        with pytest.raises(RuntimeError, match="Git CLI is not installed"):
            GitCliOps(Path("/tmp"))

        # Verify it tried to check git version
        mock_run.assert_called_once_with(
            ["git", "--version"],
            capture_output=True,
            check=True,
        )


def test_gitops_enable_sparse_checkout(git_ops, temp_git_repo):
    """Test that both gitops perform sparse checkout."""
    file_path = "subdir/testfile.yaml"
    dir_path = str(Path(file_path).parent)

    (temp_git_repo / file_path).parent.mkdir(parents=True, exist_ok=True)
    (temp_git_repo / file_path).write_text("new content")
    (temp_git_repo / "otherfile.yaml").write_text("other content")
    (temp_git_repo / "text.txt").write_text("extra content")

    git_ops.enable_sparse_checkout(file_path)

    # Check only the specified file is tracked by git
    result = subprocess.run(
        ["git", "sparse-checkout", "list"],
        cwd=temp_git_repo,
        capture_output=True,
        text=True,
        check=True,
    )
    tracked_dirs = result.stdout.strip().splitlines()
    assert dir_path in tracked_dirs


def test_cli_gitops_succeeds_when_git_installed(temp_git_repo):
    """Test that CliGitOps initialises successfully when git is installed."""
    # This should not raise any exception. Assumes git is installed.
    git_ops = GitCliOps(temp_git_repo)
    assert git_ops.repo_path == temp_git_repo


@pytest.mark.asyncio
async def test_commit_and_create_pr_only_stages_and_commits_specified_file(
    git_ops, temp_git_repo
):
    """Test that only the specified file is staged and committed in a temp git repo."""

    # Create two files in the repo
    file_path = temp_git_repo / "subdir" / "testfile.yaml"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    other_file = temp_git_repo / "otherfile.yaml"
    file_path.write_text("new content")
    other_file.write_text("untouched content")

    # Only run for GitPyOps to avoid subprocess patching
    if isinstance(git_ops, GitPyOps):
        # GitPyOps only patch setup and push
        with (
            patch(
                "flows.create_classifiers_specs_pr._run_subprocess_with_error_logging"
            ) as mock_run_subprocess,
            patch.object(git_ops, "push", autospec=True) as mock_push,
        ):
            mock_run_subprocess.side_effect = [
                Mock(stdout=""),  # gh auth setup-git
                Mock(stdout=""),  # git remote set url
                Mock(
                    stdout="https://github.com/climatepolicyradar/knowledge-graph/pull/123"
                ),  # gh pr create
            ]

            pr_number = await commit_and_create_pr(
                file_path=str(file_path.relative_to(temp_git_repo)),
                commit_message="Update testfile",
                pr_title="Update testfile",
                pr_body="Automated update",
                git=git_ops,
                repo="test_repo",
                base_branch="main",
                repo_path=temp_git_repo,
            )

        assert pr_number == 123
        mock_push.assert_called_once()

        # Check git status: only testfile.yaml should be committed
        result = git_ops.status_porcelain(str(file_path.relative_to(temp_git_repo)))
        assert result.strip() == ""  # No changes left for testfile.yaml

        # Check that otherfile.yaml is still unstaged
        other_status = git_ops.status_porcelain(
            str(other_file.relative_to(temp_git_repo))
        )
        assert other_status.strip() == "?? otherfile.yaml"

        # Check git log to confirm only testfile.yaml was committed
        log_result = subprocess.run(
            ["git", "log", "--name-only", "--pretty=oneline"],
            cwd=temp_git_repo,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "subdir/testfile.yaml" in log_result.stdout
        assert "otherfile.yaml" not in log_result.stdout
