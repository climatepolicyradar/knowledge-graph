import asyncio
import json
import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

from prefect import task
from pydantic import SecretStr

from flows.result import Err, Error, Ok, Result, is_err
from flows.utils import get_logger, total_minutes
from knowledge_graph.cloud import AwsEnv


def _run_subprocess_with_error_logging(
    cmd: list[str], cwd: Path, check: bool = True
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess command, capturing output and logging error."""
    logger = get_logger()
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=check,
        )
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command `{' '.join(cmd)}` failed with exit code {e.returncode}")
        if e.stdout:
            logger.error(f"STDOUT: {e.stdout.strip()}")
        if e.stderr:
            logger.error(f"STDERR: {e.stderr.strip()}")
        raise


class GitOps(Protocol):
    """Protocol defining Git operations interface."""

    def config(self, key: str, value: str) -> None:
        """Set a git config value."""
        ...

    def status_porcelain(self, file_path: str) -> str:
        """Get git status output for a specific file in porcelain format."""
        ...

    def checkout_new_branch(self, branch_name: str) -> None:
        """Create and checkout a new branch."""
        ...

    def add(self, file_path: str) -> None:
        """Stage a file for commit."""
        ...

    def commit(self, message: str) -> None:
        """Commit staged changes with a message."""
        ...

    def push(self, branch_name: str, remote: str = "origin") -> None:
        """Push a branch to remote."""
        ...

    def enable_sparse_checkout(self, file_path: str) -> None:
        """Enable a sparse checkout of a specific directory path."""
        ...


class GitCliOps:
    """Git operations implemented using CLI subprocess commands."""

    def __init__(self, repo_path: Path) -> None:
        """Initialise with repository path."""
        # Check if git CLI is installed
        try:
            subprocess.run(
                ["git", "--version"],
                capture_output=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                "Git CLI is not installed or not available in PATH. "
                "Please install git: https://git-scm.com/downloads"
            ) from e

        self.repo_path = repo_path

    def config(self, key: str, value: str) -> None:
        """Set a git config value."""
        _run_subprocess_with_error_logging(
            ["git", "config", key, value],
            cwd=self.repo_path,
        )

    def status_porcelain(self, file_path: str) -> str:
        """Get git status output for a specific file in porcelain format."""
        result = _run_subprocess_with_error_logging(
            ["git", "status", "--porcelain", file_path],
            cwd=self.repo_path,
        )
        return result.stdout

    def checkout_new_branch(self, branch_name: str) -> None:
        """Create and checkout a new branch."""
        _run_subprocess_with_error_logging(
            ["git", "checkout", "-b", branch_name],
            cwd=self.repo_path,
        )

    def add(self, file_path: str) -> None:
        """Stage a file for commit."""
        _run_subprocess_with_error_logging(
            ["git", "add", file_path],
            cwd=self.repo_path,
        )

    def commit(self, message: str) -> None:
        """Commit staged changes with a message."""
        _run_subprocess_with_error_logging(
            ["git", "commit", "--no-verify", "-m", message],
            cwd=self.repo_path,
        )

    def push(self, branch_name: str, remote: str = "origin") -> None:
        """Push a branch to remote."""
        _run_subprocess_with_error_logging(
            ["git", "push", "-u", remote, branch_name],
            cwd=self.repo_path,
        )

    def enable_sparse_checkout(self, file_path: str) -> None:
        """Enable a sparse checkout of a specific directory path."""
        dir_path = str(Path(file_path).parent)
        _run_subprocess_with_error_logging(
            ["git", "sparse-checkout", "set", dir_path],
            cwd=self.repo_path,
        )


class GitPyOps:
    """Git operations implemented using GitPython library."""

    def __init__(self, repo_path: Path) -> None:
        """Initialise with repository path."""
        try:
            import git
        except ImportError:
            raise ImportError(
                "GitPython is not installed. Install it with: pip install GitPython"
            )

        self.repo = git.Repo(repo_path)

    def config(self, key: str, value: str) -> None:
        """Set a git config value."""
        with self.repo.config_writer() as config:
            section, option = key.split(".", 1)
            config.set_value(section, option, value)

    def status_porcelain(self, file_path: str) -> str:
        """Get git status output for a specific file in porcelain format."""
        # GitPython doesn't have direct porcelain output, so we simulate it
        diff_index = self.repo.index.diff(None)  # Unstaged changes
        diff_staged = self.repo.index.diff("HEAD")  # Staged changes
        untracked = self.repo.untracked_files

        result = []
        for diff in diff_staged:
            if diff.a_path == file_path or diff.b_path == file_path:
                result.append(f"M  {file_path}")

        for diff in diff_index:
            if diff.a_path == file_path or diff.b_path == file_path:
                result.append(f" M {file_path}")

        if file_path in untracked:
            result.append(f"?? {file_path}")

        return "\n".join(result)

    def checkout_new_branch(self, branch_name: str) -> None:
        """Create and checkout a new branch."""
        new_branch = self.repo.create_head(branch_name)
        new_branch.checkout()

    def add(self, file_path: str) -> None:
        """Stage a file for commit."""
        self.repo.index.add([file_path])

    def commit(self, message: str) -> None:
        """Commit staged changes with a message."""
        self.repo.index.commit(message, skip_hooks=True)

    def push(self, branch_name: str, remote: str = "origin") -> None:
        """Push a branch to remote."""
        origin = self.repo.remote(name=remote)
        origin.push(refspec=f"{branch_name}:{branch_name}", set_upstream=True)

    def enable_sparse_checkout(self, file_path: str) -> None:
        """Enable a sparse checkout of a specific directory path."""
        dir_path = str(Path(file_path).parent)
        self.repo.git.sparse_checkout("set", dir_path)


async def commit_and_create_pr(
    file_path: str,
    commit_message: str,
    pr_title: str,
    pr_body: str,
    git: GitOps,
    repo: str = "climatepolicyradar/knowledge-graph",
    base_branch: str = "main",
    repo_path: Path = Path("/app"),
) -> int | None:
    """Commits changes and creates a GitHub PR using gh CLI."""
    logger = get_logger()

    # Check if there are changes
    status_output = git.status_porcelain(file_path)

    if not status_output.strip():
        logger.info("No changes detected, skipping PR creation")
        return None

    # Configure git (in case not set)
    git.config("user.email", "tech@climatepolicyradar.org")
    git.config("user.name", "cpr-tech-admin")

    # removing sparse_checkout, throws git index version error
    # git.enable_sparse_checkout(file_path)
    # Some files that are not copied in the docker build may
    # show as D (deleted) however only specified file_path will
    # be committed

    # Create and checkout new branch
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    branch_name = f"auto/classifier-specs-{timestamp}"

    git.checkout_new_branch(branch_name)

    # Add and commit changes
    git.add(file_path)
    git.commit(commit_message)

    # Ensure gh is configured as git credential helper
    logger.info("Setting up gh as git credential helper")
    _ = _run_subprocess_with_error_logging(
        ["gh", "auth", "setup-git"],
        cwd=repo_path,
    )

    # Push branch to remote
    logger.info(f"Pushing branch {branch_name} to remote")
    git.push(branch_name)

    # Create PR using gh CLI
    logger.info(f"Creating pull request: {pr_title}")

    result = _run_subprocess_with_error_logging(
        [
            "gh",
            "pr",
            "create",
            "--title",
            pr_title,
            "--body",
            pr_body,
            "--base",
            base_branch,
            "--head",
            branch_name,
            "--repo",
            repo,
        ],
        cwd=repo_path,
    )

    # Extract PR number from gh output (URL is in stdout)
    pr_number, pr_url = extract_pr_details(result.stdout)

    logger.info(f"Created PR #{pr_number}: {pr_url}")

    return pr_number


def extract_pr_details(result_str: str) -> tuple[int, str]:
    if not result_str.strip():
        raise ValueError("The result string is empty.")

    try:
        pr_url = result_str.strip()
        pr_number = int(pr_url.split("/")[-1])
        print(f"Extracted PR number: {pr_number}, URL: {pr_url}")
        return pr_number, pr_url
    except Exception as e:
        raise ValueError(f"Failed to extract PR details: {e}")


async def enable_auto_merge(
    pr_number: int,
    merge_method: str,
    repo: str = "climatepolicyradar/knowledge-graph",
) -> Result[int | None, Error]:
    """Enable auto-merge on a GitHub PR."""
    logger = get_logger()

    try:
        _ = _run_subprocess_with_error_logging(
            [
                "gh",
                "pr",
                "merge",
                str(pr_number),
                f"--{merge_method.lower()}",
                "--auto",
                "--repo",
                repo,
            ],
            cwd=Path("./"),
        )

        logger.info(f"Enabled auto-merge on PR #{pr_number}")
        return Ok(pr_number)
    except Exception as e:
        logger.error(f"Failed to enable auto-merge: {e}")
        return Err(
            Error(
                msg="Failed to enable auto-merge for PR.",
                metadata={"exception": e, "pr_number": pr_number},
            )
        )


async def wait_for_pr_merge(
    pr_number: int,
    timeout: timedelta,
    poll_interval: timedelta,
    repo: str = "climatepolicyradar/knowledge-graph",
) -> Result[int | None, Error]:
    """Wait for a PR to be merged by polling with gh CLI."""
    logger = get_logger()

    try:
        start_time = time.time()
        timeout_minutes = total_minutes(timeout)
        poll_interval_seconds = poll_interval.total_seconds()

        logger.info(
            f"Waiting for PR #{pr_number} to merge (timeout: {timeout_minutes}m, "
            f"poll interval: {poll_interval_seconds}s)"
        )

        while True:
            elapsed = time.time() - start_time

            if elapsed > timeout.total_seconds():
                logger.error(
                    f"PR #{pr_number} did not merge within {timeout_minutes} minutes. Check: https://github.com/{repo}/pull/{pr_number}."
                )
                return Err(
                    Error(
                        msg="TimeoutError: PR did not merge within the timeout period.",
                        metadata={
                            "pr_number": pr_number,
                            "timeout_minutes": timeout_minutes,
                            "elapsed_time": elapsed,
                            "repo": repo,
                        },
                    )
                )

            # Check PR status using gh
            result = subprocess.run(
                [
                    "gh",
                    "pr",
                    "view",
                    str(pr_number),
                    "--json",
                    "state,mergedAt",
                    "--repo",
                    repo,
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                logger.warning(f"Failed to get PR status: {result.stderr}")
                await asyncio.sleep(poll_interval_seconds)
                continue

            pr_data = json.loads(result.stdout)

            # Check if merged (mergedAt will be non-null if merged)
            if pr_data.get("mergedAt"):
                logger.info(f"✓ PR #{pr_number} successfully merged!")
                return Ok(pr_number)

            # Check if closed without merging
            if pr_data.get("state") == "CLOSED" and not pr_data.get("mergedAt"):
                logger.error(
                    f"PR #{pr_number} was closed without merging. Check: https://github.com/{repo}/pull/{pr_number}."
                )
                return Err(
                    Error(
                        msg="RuntimeError: PR was closed without merging.",
                        metadata={
                            "pr_number": pr_number,
                            "repo": repo,
                            "pr_state": pr_data.get("state"),
                        },
                    )
                )

            # Log current status
            elapsed_mins = int(elapsed / 60)
            logger.info(
                f"PR #{pr_number} - state: {pr_data['state']}, elapsed: {elapsed_mins}m"
            )

            # Wait before next check
            await asyncio.sleep(poll_interval_seconds)

    except Exception as e:
        logger.error(f"Error while waiting for PR merge: {e}")
        return Err(
            Error(
                msg="Error while waiting for PR to merge",
                metadata={"exception": e, "pr_number": pr_number},
            )
        )


@task
async def create_and_merge_pr(
    spec_file: str,
    aws_env: AwsEnv,
    flow_run_name: str,
    flow_run_url: str,
    github_token: SecretStr,
    auto_merge: bool = False,
) -> Result[None | int, Error]:
    """
    Main workflow for creating and managing a PR

    Returns:
        Ok(None) if no changes to file,
        Ok(pr_number) for successes,
        Err on failures.
    """
    logger = get_logger()

    try:
        os.environ["GITHUB_TOKEN"] = github_token.get_secret_value()
    except Exception as e:
        logger.error(f"Failed to set GitHub token environment var: {e}")
        return Err(
            Error(
                msg="Failed to set GitHub token environment var.",
                metadata={"exception": e, "aws_env": aws_env},
            )
        )

    try:
        repo_path = Path("./")
        git_ops = GitPyOps(repo_path)

        pr_no = await commit_and_create_pr(
            file_path=spec_file,
            commit_message="Update classifier specs (automated)",
            pr_title=f"Update classifier specs for {aws_env} (automated)",
            pr_body=f"Sync to Classifier Profiles updates to classifier specs YAML files. During Flow Run {flow_run_name}, see {flow_run_url}",
            git=git_ops,
            repo="climatepolicyradar/knowledge-graph",
            base_branch="main",
            repo_path=repo_path,
        )

        if pr_no is None:
            logger.info("No PR created as there were no changes.")
            return Ok(None)
    except Exception as e:
        logger.error(f"Failed to create PR: {e}")
        return Err(
            Error(
                msg="Failed to create PR for classifiers specs changes.",
                metadata={
                    "exception": e,
                    "aws_env": aws_env,
                    "spec_file": spec_file,
                },
            )
        )

    logger.info(f"PR #{pr_no} created.")

    if not auto_merge:
        logger.info("Auto-merge not enabled as per configuration.")
        return Ok(pr_no)
    else:
        logger.info("Auto-approving and merging PR...")
        auto_merge_results = await enable_auto_merge(
            pr_number=pr_no,
            merge_method="REBASE",
            repo="climatepolicyradar/knowledge-graph",
        )

        # if error in results return early
        if is_err(auto_merge_results):
            return auto_merge_results

        pr_merge_results = await wait_for_pr_merge(
            pr_number=pr_no,
            timeout=timedelta(minutes=30),
            poll_interval=timedelta(seconds=30),
            repo="climatepolicyradar/knowledge-graph",
        )

        return pr_merge_results
