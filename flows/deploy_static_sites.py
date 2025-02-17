"""
Flow to deploy our little static sites.

See the /static_sites directory for the source code for each site.

This flow uses our high-level just commands to generate the static sites and sync
them to the corresponding s3 buckets from which they're served.
"""

import os
import subprocess

from cpr_sdk.ssm import get_aws_ssm_param
from prefect import flow, task


@task
def setup_environment():
    """Set up required environment variables from SSM parameters."""
    wikibase_password = get_aws_ssm_param("/Wikibase/Cloud/ServiceAccount/Password")
    wikibase_username = get_aws_ssm_param("/Wikibase/Cloud/ServiceAccount/Username")
    wikibase_url = get_aws_ssm_param("/Wikibase/Cloud/URL")

    os.environ["WIKIBASE_PASSWORD"] = wikibase_password
    os.environ["WIKIBASE_USERNAME"] = wikibase_username
    os.environ["WIKIBASE_URL"] = wikibase_url


@task
def generate_static_site(app_name: str):
    """Generate the static site using the existing command."""
    subprocess.run(["just", "generate-static-site", app_name], check=True)


@task
def sync_to_s3(app_name: str, bucket_name: str):
    """Sync the generated static site to S3."""
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


@flow
def deploy_static_sites():
    """Flow to deploy all our static sites."""
    # Set up environment variables first
    setup_environment()

    names = {
        "concept_librarian": "cpr-knowledge-graph-concept-librarian",
        # add more here as we create more static sites, eg the labelling librarian
    }
    for app_name, bucket_name in names.items():
        generate_static_site(app_name)
        sync_to_s3(app_name, bucket_name)


if __name__ == "__main__":
    deploy_static_sites()
