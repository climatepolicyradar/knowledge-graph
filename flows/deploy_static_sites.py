"""
Flow to deploy our little static sites.

See the /static_sites directory for the source code for each site.

This flow runs the static site generator module directly, and then uses the AWS CLI
to sync the generated files to the corresponding S3 bucket from which the site is
served.
"""

import importlib
import os
import subprocess

from cpr_sdk.ssm import get_aws_ssm_param
from prefect import flow, task
from prefect.futures import wait


def _wait_for_futures_and_check_failures(futures_dict: dict, task_type: str):
    """Helper to wait for futures and check for failures"""
    print(f"Waiting for all {task_type.lower()} tasks to complete...")
    futures_list = list(futures_dict.values())
    done_futures, _ = wait(futures_list)

    failed_sites = []
    for future in done_futures:
        if future.state.is_failed():
            app_name = next(name for name, f in futures_dict.items() if f == future)
            failed_sites.append(app_name)
            print(f"{task_type} failed for {app_name}: {future.state.message}")

    if failed_sites:
        raise Exception(f"{task_type} failed for sites: {', '.join(failed_sites)}")

    return done_futures


@task(log_prints=True)
def setup_environment():
    """Set up required environment variables from SSM parameters."""
    print("Setting up environment variables from SSM parameters...")

    os.environ["ARGILLA_API_URL"] = get_aws_ssm_param("/Argilla/APIURL")
    os.environ["ARGILLA_API_KEY"] = get_aws_ssm_param("/Argilla/Owner/APIKey")

    os.environ["WIKIBASE_PASSWORD"] = get_aws_ssm_param(
        "/Wikibase/Cloud/ServiceAccount/Password"
    )
    os.environ["WIKIBASE_USERNAME"] = get_aws_ssm_param(
        "/Wikibase/Cloud/ServiceAccount/Username"
    )
    os.environ["WIKIBASE_URL"] = get_aws_ssm_param("/Wikibase/Cloud/URL")
    print("Environment variables set.")


@task(log_prints=True)
def generate_static_site(app_name: str):
    """Generate the static site by running the generator module directly."""
    print(f"Generating static site for {app_name}...")

    module = importlib.import_module(f"static_sites.{app_name}.__main__")
    module.main()
    print(f"Static site generation complete for {app_name}.")


@task(log_prints=True)
def sync_to_s3(app_name: str, bucket_name: str):
    """Sync the generated static site to S3."""
    print(f"Syncing {app_name} to S3 bucket {bucket_name}...")
    subprocess.run(
        [
            "aws",
            "s3",
            "sync",
            f"static_sites/{app_name}/dist",
            f"s3://{bucket_name}/dist",
            "--delete",
        ],
        check=True,
    )
    print(f"Sync complete for {app_name} to S3 bucket {bucket_name}/dist")


@flow(log_prints=True)
def deploy_static_sites():
    """Flow to deploy all our static sites in parallel."""
    print("Starting deployment of static sites...")

    app_name_to_bucket_name = {
        "concept_librarian": "cpr-knowledge-graph-concept-librarian",
        "labelling_librarian": "cpr-knowledge-graph-labelling-librarian",
        "vibe_check": "cpr-knowledge-graph-vibe-check",
    }

    setup_environment()

    # submit all generation tasks in parallel
    generation_futures = {}
    for app_name in app_name_to_bucket_name:
        print(f"Initiating generation for {app_name}.")
        future = generate_static_site.submit(app_name=app_name)
        generation_futures[app_name] = future

    # wait for all generation tasks to complete
    done_generation_futures = _wait_for_futures_and_check_failures(
        generation_futures, "Generation"
    )

    # submit all sync tasks in parallel
    sync_futures = {}
    for app_name, gen_future in generation_futures.items():
        if gen_future in done_generation_futures and not gen_future.state.is_failed():
            print(f"Generation completed for {app_name}, starting sync...")
            sync_future = sync_to_s3.submit(
                app_name=app_name, bucket_name=app_name_to_bucket_name[app_name]
            )
            sync_futures[app_name] = sync_future

    # wait for all sync tasks to complete
    _wait_for_futures_and_check_failures(sync_futures, "Sync")

    print("All static site deployments have finished.")


if __name__ == "__main__":
    deploy_static_sites()
