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
            f"s3://{bucket_name}",
            "--delete",
        ],
        check=True,
    )
    print(f"Sync complete for {app_name} to S3 bucket {bucket_name}.")


@flow(log_prints=True)
def deploy_one_site_pipeline(app_name: str, bucket_name: str):
    """Generates and deploys a single static site."""
    print(f"Starting deployment pipeline for {app_name} to bucket {bucket_name}.")

    try:
        gen_future = generate_static_site(app_name)
        sync_to_s3(app_name, bucket_name, wait_for=[gen_future])  # type: ignore
        print(f"Deployment tasks for {app_name} completed.")
    except Exception as e:
        print(f"Error in deployment pipeline for {app_name}: {e}")
        raise


@flow(log_prints=True)
def deploy_static_sites():
    """Flow to deploy all our static sites in parallel."""
    print("Starting deployment of all static sites.")

    sites_to_deploy = [
        {
            "app_name": "concept_librarian",
            "bucket_name": "cpr-knowledge-graph-concept-librarian",
        },
        {
            "app_name": "labelling_librarian",
            "bucket_name": "cpr-knowledge-graph-labelling-librarian",
        },
        {"app_name": "vibe_check", "bucket_name": "cpr-knowledge-graph-vibe-check"},
    ]

    setup_env_future = setup_environment()

    active_deployments = []
    for site in sites_to_deploy:
        print(f"Initiating deployment for {site['app_name']}.")
        deployment_future = deploy_one_site_pipeline(
            app_name=site["app_name"],
            bucket_name=site["bucket_name"],
            wait_for=[setup_env_future],  # type: ignore
        )
        active_deployments.append(deployment_future)

    print(
        f"All {len(sites_to_deploy)} site deployment pipelines initiated. Main flow will wait for their completion."
    )

    print("All static site deployment pipelines have concluded.")


if __name__ == "__main__":
    deploy_static_sites()
