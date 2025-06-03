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


@task
def setup_environment():
    """Set up required environment variables from SSM parameters."""

    os.environ["ARGILLA_API_URL"] = get_aws_ssm_param("/Argilla/APIURL")
    os.environ["ARGILLA_API_KEY"] = get_aws_ssm_param("/Argilla/APIKey")

    os.environ["WIKIBASE_PASSWORD"] = get_aws_ssm_param(
        "/Wikibase/Cloud/ServiceAccount/Password"
    )
    os.environ["WIKIBASE_USERNAME"] = get_aws_ssm_param(
        "/Wikibase/Cloud/ServiceAccount/Username"
    )
    os.environ["WIKIBASE_URL"] = get_aws_ssm_param("/Wikibase/Cloud/URL")


@task
def generate_static_site(app_name: str):
    """Generate the static site by running the generator module directly."""
    module = importlib.import_module(f"static_sites.{app_name}.__main__")
    module.main()


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
        "labelling_librarian": "cpr-knowledge-graph-labelling-librarian",
        "vibe_check": "cpr-knowledge-graph-vibe-check",
    }
    for app_name, bucket_name in names.items():
        generate_static_site(app_name)
        sync_to_s3(app_name, bucket_name)


if __name__ == "__main__":
    deploy_static_sites()
