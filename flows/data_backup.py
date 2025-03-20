"""Flow to deploy an automated backup of Argilla to Huggingface"""

import os

from cpr_sdk.ssm import get_aws_ssm_param
from prefect import flow, task

from src.argilla_v2 import ArgillaSession
from src.huggingface import HuggingfaceSession


@task
def setup_environment():
    """Set up required environment variables from SSM parameters."""

    os.environ["ARGILLA_API_URL"] = get_aws_ssm_param("/Argilla/APIURL")
    os.environ["ARGILLA_API_KEY"] = get_aws_ssm_param("/Argilla/Owner/APIKey")

    os.environ["HF_TOKEN"] = get_aws_ssm_param("/Huggingface/Token")


@flow
def data_backup():
    """Flow to deploy all our static sites."""
    setup_environment()
    argilla_session = ArgillaSession()
    hf_session = HuggingfaceSession()

    datasets = argilla_session.get_all_datasets("knowledge-graph")
    for dataset in datasets:
        labelled_passages = argilla_session.dataset_to_labelled_passages(dataset)
        hf_session.push(dataset.name, labelled_passages)


if __name__ == "__main__":
    data_backup()
