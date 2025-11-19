import asyncio
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

from prefect import task
from pydantic import SecretStr

from flows.result import Err, Error, Ok, Result, is_err
from flows.utils import get_logger
from knowledge_graph.cloud import AwsEnv


async def commit_and_create_pr(
    file_path: str,
    commit_message: str,
    pr_title: str,
    pr_body: str,
    repo: str = "climatepolicyradar/knowledge-graph",
    base_branch: str = "main",
    repo_path: Path = Path("/app"),
) -> int | None:
    """
    Commits changes and creates a GitHub PR using gh CLI.

    Args:
        file_path: Path to classifiers spec file to update
        commit_message: Git commit message
        pr_title: Pull request title
        pr_body: Pull request body
        repo: Repository in format "owner/repo"
        base_branch: Base branch for PR
        repo_path: Path to git repository

    Returns:
        PR number (or 0 if no changes)
    """
    logger = get_logger()

    # Check if there are changes
    result = subprocess.run(
        ["git", "status", "--porcelain", file_path],
        cwd=repo_path,
        capture_output=True,
        text=True,
        check=True,
    )

    if not result.stdout.strip():
        logger.info("No changes detected, skipping PR creation")
        return None

    # Configure git (in case not set)
    _ = subprocess.run(
        ["git", "config", "user.email", "tech@climatepolicyradar.org"],
        cwd=repo_path,
        check=True,
    )
    _ = subprocess.run(
        ["git", "config", "user.name", "cpr-tech-admin"],
        cwd=repo_path,
        check=True,
    )

    # Create and checkout new branch
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    branch_name = f"auto/classifier-specs-{timestamp}"

    _ = subprocess.run(
        ["git", "checkout", "-b", branch_name],
        cwd=repo_path,
        check=True,
    )

    # Add and commit changes
    _ = subprocess.run(
        ["git", "add", file_path],
        cwd=repo_path,
        check=True,
    )

    _ = subprocess.run(
        ["git", "commit", "-m", commit_message],
        cwd=repo_path,
        check=True,
    )

    # Ensure gh is configured as git credential helper
    logger.info("Setting up gh as git credential helper")
    _ = subprocess.run(
        ["gh", "auth", "setup-git"],
        cwd=repo_path,
        check=True,
    )

    # Push branch to remote
    logger.info(f"Pushing branch {branch_name} to remote")
    _ = subprocess.run(
        ["git", "push", "-u", "origin", branch_name],
        cwd=repo_path,
        check=True,
    )

    # Create PR using gh CLI
    logger.info(f"Creating pull request: {pr_title}")

    result = subprocess.run(
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
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        logger.error(f"Failed to create PR: {result.stderr}")
        raise RuntimeError(f"gh pr create failed: {result.stderr}")

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
) -> Result[None, Error]:
    """
    Enable auto-merge on a GitHub PR.

    Args:
        pr_number: Pull request number
        repo: Repository in format "owner/repo"
        merge_method: Merge method (squash, merge, or rebase)
    """
    logger = get_logger()

    try:
        subprocess.run(
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
            check=True,
        )

        logger.info(f"Enabled auto-merge on PR #{pr_number}")
        return Ok(None)
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
    timeout_minutes: int,
    poll_interval_seconds: int,
    repo: str = "climatepolicyradar/knowledge-graph",
) -> Result[None, Error]:
    """
    Wait for a PR to be merged by polling with gh CLI.

    Args:
        pr_number: Pull request number
        repo: Repository in format "owner/repo"
        timeout_minutes: Maximum time to wait
        poll_interval_seconds: Time between status checks
    """
    logger = get_logger()

    try:
        start_time = time.time()
        timeout_seconds = timeout_minutes * 60

        logger.info(
            f"Waiting for PR #{pr_number} to merge (timeout: {timeout_minutes}m, "
            f"poll interval: {poll_interval_seconds}s)"
        )

        while True:
            elapsed = time.time() - start_time

            if elapsed > timeout_seconds:
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
                return Ok(None)

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
) -> list[Result[None | int, Error]]:
    """
    Main workflow for creating and managing a PR.

    Args:
        spec_file: Path to the classifiers spec yaml file
        aws_env: AWS environment the spec belongs to
        flow_run_name: Name of the Prefect flow run calling the changes
        flow_run_url: URL to the Prefect flow run calling the changes
        github_token: GitHub token for authentication
        auto_merge: Whether to auto-approve and merge the PR

    Returns:
        List of Results indicating success or failure of each step
    """
    logger = get_logger()
    results = []

    # get github token from ssm params
    try:
        os.environ["GITHUB_TOKEN"] = github_token.get_secret_value()
    except Exception as e:
        logger.error(f"Failed to retrieve GitHub token: {e}")
        results.append(
            Err(
                Error(
                    msg="Failed to retrieve GitHub token from SSM.",
                    metadata={"exception": e, "aws_env": aws_env},
                )
            )
        )
        return results

    try:
        pr_no = await commit_and_create_pr(
            file_path=spec_file,
            commit_message="Update classifier specs (automated)",
            pr_title=f"Update classifier specs for {aws_env} (automated)",
            pr_body=f"Sync to Classifier Profiles updates to classifier specs YAML files. During Flow Run {flow_run_name}, see {flow_run_url}",
            repo="climatepolicyradar/knowledge-graph",
            base_branch="main",
            repo_path=Path("./"),
        )
        results.append(Ok(pr_no))
        if pr_no is None:
            logger.info("No PR created as there were no changes.")
            return results
    except Exception as e:
        logger.error(f"Failed to create PR: {e}")
        results.append(
            Err(
                Error(
                    msg="Failed to create PR for classifiers specs changes.",
                    metadata={
                        "exception": e,
                        "aws_env": aws_env,
                        "spec_file": spec_file,
                    },
                )
            )
        )
        return results

    logger.info(f"PR #{pr_no} created.")

    if not auto_merge:
        logger.info("Auto-merge not enabled as per configuration.")
    else:
        logger.info("Auto-approving and merging PR...")
        results.append(
            await enable_auto_merge(
                pr_number=pr_no,
                merge_method="REBASE",
                repo="climatepolicyradar/knowledge-graph",
            )
        )

        # if error in results return
        if any(is_err(result) for result in results):
            return results

        results.append(
            await wait_for_pr_merge(
                pr_number=pr_no,
                timeout_minutes=30,
                poll_interval_seconds=30,
                repo="climatepolicyradar/knowledge-graph",
            )
        )

    return results
