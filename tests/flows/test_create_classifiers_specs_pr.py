from unittest.mock import Mock, patch

import pytest

from flows.create_classifiers_specs_pr import (
    commit_and_create_pr,
    create_and_merge_pr,
    enable_auto_merge,
    extract_pr_details,
    wait_for_pr_merge,
)
from flows.result import Err, Error, Ok, is_err, is_ok, unwrap_err
from knowledge_graph.cloud import AwsEnv


@pytest.mark.asyncio
async def test_commit_and_create_pr__with_changes():
    """Test commit_and_create_pr when there are changes to commit."""
    with patch("subprocess.run") as mock_run:
        # Mock subprocess.run for git and gh commands
        mock_run.side_effect = [
            Mock(stdout="M testfile\n"),  # git status
            Mock(),  # git config user.email
            Mock(),  # git config user.name
            Mock(),  # git checkout -b
            Mock(),  # git add
            Mock(),  # git commit
            Mock(),  # gh auth setup-git
            Mock(),  # git push
            Mock(
                stdout="https://github.com/climatepolicyradar/knowledge-graph/pull/123",
                returncode=0,
            ),  # gh pr create
        ]

        pr_number = await commit_and_create_pr(
            file_path="testfile",
            commit_message="Update testfile",
            pr_title="Update testfile",
            pr_body="Automated update",
            repo="climatepolicyradar/knowledge-graph",
            base_branch="main",
            repo_path=Mock(),
        )

        assert pr_number == 123
        assert mock_run.call_count == 9


@pytest.mark.asyncio
async def test_commit_and_create_pr__no_changes():
    """Test commit_and_create_pr when there are no changes."""
    with patch("subprocess.run") as mock_run:
        # Mock subprocess.run for git status
        mock_run.return_value = Mock(stdout="")

        repo_path_mock = Mock()
        pr_number = await commit_and_create_pr(
            file_path="testfile",
            commit_message="Update testfile",
            pr_title="Update testfile",
            pr_body="Automated update",
            repo="climatepolicyradar/knowledge-graph",
            base_branch="main",
            repo_path=repo_path_mock,
        )

        assert pr_number is None
        mock_run.assert_called_once_with(
            ["git", "status", "--porcelain", "testfile"],
            cwd=repo_path_mock,
            capture_output=True,
            text=True,
            check=True,
        )


@pytest.mark.asyncio
async def test_enable_auto_merge():
    """Test enabling auto-merge on a PR."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = Mock()
        result = await enable_auto_merge(
            pr_number=123,
            repo="climatepolicyradar/knowledge-graph",
            merge_method="squash",
        )

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
            check=True,
        )
        assert result == Ok(None)


@pytest.mark.asyncio
async def test_enable_auto_merge__exception():
    """Test enabling auto-merge on a PR where exception is raised."""
    with patch("subprocess.run") as mock_run:
        mock_run.side_effect = Exception("GitHub CLI error")
        result = await enable_auto_merge(
            pr_number=123,
            repo="climatepolicyradar/knowledge-graph",
            merge_method="squash",
        )

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
            check=True,
        )
        assert is_err(result)
        error = unwrap_err(result)
        assert "GitHub CLI error" in error.msg


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

        result = await wait_for_pr_merge(
            pr_number=123,
            repo="climatepolicyradar/knowledge-graph",
            timeout_minutes=1,
            poll_interval_seconds=0.1,
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
        assert result == Ok(None)


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
            timeout_minutes=0,
            poll_interval_seconds=0.1,
        )

        assert is_err(result)
        assert (
            "TimeoutError: PR #123 did not merge within 0 minutes. Check: https://github.com/climatepolicyradar/knowledge-graph/pull/123."
            in unwrap_err(result).msg
        )


@pytest.mark.asyncio
async def test_wait_for_pr_merge__closed():
    """Test waiting for a PR to merge with error from closing without merging."""
    with patch("subprocess.run") as mock_run:
        # Mock subprocess.run for gh pr view
        mock_run.return_value = Mock(
            stdout='{"state": "CLOSED", "mergedAt": null}', returncode=0
        )

        result = await wait_for_pr_merge(
            pr_number=123,
            repo="climatepolicyradar/knowledge-graph",
            timeout_minutes=1,
            poll_interval_seconds=0.1,
        )

        assert is_err(result)
        assert (
            "RuntimeError: PR #123 was closed without merging. Check: https://github.com/climatepolicyradar/knowledge-graph/pull/123."
            in unwrap_err(result).msg
        )


@pytest.mark.asyncio
async def test_wait_for_pr_merge__failed_to_get_pr_timeout():
    """Test waiting for a PR to merge with timeout error from when non-zero returncode."""
    with patch("subprocess.run") as mock_run:
        # Mock subprocess.run for gh pr view
        # Returncode 1 simulates failure to get PR info until timeout
        mock_run.return_value = Mock(
            stdout='{"state": "OPEN", "mergedAt": null}', returncode=1
        )

        result = await wait_for_pr_merge(
            pr_number=123,
            repo="climatepolicyradar/knowledge-graph",
            timeout_minutes=0.01,  # Short timeout for test 0.6s
            poll_interval_seconds=0.2,
        )

        assert is_err(result)
        assert (
            "TimeoutError: PR #123 did not merge within 0.01 minutes. Check: https://github.com/climatepolicyradar/knowledge-graph/pull/123."
            in unwrap_err(result).msg
        )


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
            timeout_minutes=30,
            poll_interval_seconds=30,
        )
        assert all(is_ok(r) for r in results)


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
            auto_merge=False,
        )

        mock_commit.assert_called_once()
        mock_enable_merge.assert_not_called()
        mock_wait_merge.assert_not_called()

        assert all(is_ok(r) for r in results)


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
            auto_merge=True,
        )

        mock_commit.assert_called_once()
        mock_enable_merge.assert_called_once_with(
            pr_number=123,
            repo="climatepolicyradar/knowledge-graph",
            merge_method="REBASE",
        )
        mock_wait_merge.assert_not_called()

        assert results[0] == Ok(123)
        errors = [r._error for r in results if isinstance(r, Err)]
        assert any("Test error" in e.msg for e in errors)


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
