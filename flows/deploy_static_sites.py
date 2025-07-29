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
            f"s3://{bucket_name}/dist",
            "--delete",
        ],
        check=True,
    )
    print(f"Sync complete for {app_name} to S3 bucket {bucket_name}/dist")


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

    setup_environment()

    generation_futures = []
    for site in sites_to_deploy:
        print(f"Initiating generation for {site['app_name']}.")
        future = generate_static_site.submit(app_name=site["app_name"])
        generation_futures.append((future, site))

    sync_futures = []
    for future, site in generation_futures:
        try:
            future.result()  # Wait for generation to complete
            print(f"Generation completed for {site['app_name']}, starting sync...")
            sync_future = sync_to_s3.submit(
                app_name=site["app_name"], bucket_name=site["bucket_name"]
            )
            sync_futures.append((sync_future, site))
        except Exception as e:
            print(f"Generation failed for {site['app_name']}: {e}")
            raise

    for future, site in sync_futures:
        try:
            future.result()
            print(f"Successfully completed deployment for {site['app_name']}")
        except Exception as e:
            print(f"Sync failed for {site['app_name']}: {e}")
            raise

    print("All static site deployment pipelines have concluded.")


if __name__ == "__main__":
    deploy_static_sites()
