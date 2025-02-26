"""
Flow to deploy an automated backup of Argilla to Huggingface
"""

import os

from cpr_sdk.ssm import get_aws_ssm_param
from prefect import flow, task

from src.huggingface import HuggingfaceSession
from src.argilla_v2 import (
    dataset_to_labelled_passages,
    get_all_datasets,
)


@task
def setup_environment():
    """Set up required environment variables from SSM parameters."""

    os.environ["ARGILLA_API_URL"] = get_aws_ssm_param("/Argilla/APIURL")
    os.environ["ARGILLA_API_KEY"] = get_aws_ssm_param("/Argilla/APIKey")

    os.environ["HF_TOKEN"] = get_aws_ssm_param("/Huggingface/Token")


@flow
def data_backup():
    """Flow to deploy all our static sites."""
    setup_environment()

    hf_session = HuggingfaceSession()

    datasets = get_all_datasets("knowledge-graph")
    for dataset in datasets:
        labelled_passages = dataset_to_labelled_passages(dataset)
        hf_session.push(dataset.name, labelled_passages)


if __name__ == "__main__":
    data_backup()
