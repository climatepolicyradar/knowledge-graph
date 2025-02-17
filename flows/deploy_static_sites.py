"""
Flow to deploy our little static sites.

See the /static_sites directory for the source code for each site.

This flow uses our high-level just commands to generate the static sites and sync
them to the corresponding s3 buckets from which they're served.
"""

import subprocess

from prefect import flow, task

from static_sites.concept_librarian.infra.__main__ import (
    app_name as concept_librarian_app_name,
)


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
            "--profile=labs",
            "--delete",
        ],
        check=True,
    )


@flow
def deploy_static_sites():
    """Flow to deploy all our static sites."""
    names = {
        "concept_librarian": concept_librarian_app_name,
        # add more here as we create more static sites, eg the labelling librarian
    }
    for app_name, bucket_name in names.items():
        generate_static_site(app_name)
        sync_to_s3(app_name, bucket_name)


if __name__ == "__main__":
    deploy_static_sites()
