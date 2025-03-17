import os
import re
from typing import TypeAlias

import boto3
from botocore.exceptions import ClientError
from prefect.settings import PREFECT_UI_URL
from prefect_slack.credentials import SlackWebhook

# Example: CCLW.executive.1813.2418
DocumentImportId: TypeAlias = str
DocumentStem: TypeAlias = str


def file_name_from_path(path: str) -> str:
    """Get the file name from a path without the path or extension"""
    return os.path.splitext(os.path.basename(path))[0]


class SlackNotify:
    """Notify a Slack channel through a Prefect Slack webhook."""

    # Message templates
    FLOW_RUN_URL = "{prefect_base_url}/flow-runs/flow-run/{flow_run.id}"
    BASE_MESSAGE = (
        "Flow run {flow.name}/{flow_run.name} observed in "
        "state `{flow_run.state.name}` at {flow_run.state.timestamp}. "
        "For environment: {environment}. "
        "Flow run URL: {ui_url}. "
        "State message: {state.message}"
    )

    # Block name
    slack_channel_name = "platform"
    environment = os.getenv("AWS_ENV", "sandbox")
    slack_block_name = f"slack-webhook-{slack_channel_name}-prefect-mvp-{environment}"

    @classmethod
    def message(cls, flow, flow_run, state):
        """
        Send a notification to a Slack channel about the state of a Prefect flow run.

        Intended to be called from prefect flow hooks:

        ```python
        @flow(on_failure=[SlackNotify.message])
        def my_flow():
            pass
        ```
        """

        ui_url = cls.FLOW_RUN_URL.format(
            prefect_base_url=PREFECT_UI_URL.value(), flow_run=flow_run
        )
        msg = cls.BASE_MESSAGE.format(
            flow=flow,
            flow_run=flow_run,
            state=state,
            ui_url=ui_url,
            environment=cls.environment,
        )

        slack = SlackWebhook.load(cls.slack_block_name)
        slack.notify(body=msg)


def remove_translated_suffix(file_name: str) -> str:
    """
    Remove the suffix from a file name that indicates it has been translated.

    E.g. "CCLW.executive.1.1_en_translated" -> "CCLW.executive.1.1"
    """
    return re.sub(r"(_translated(?:_[a-zA-Z]+)?)$", "", file_name)


# TODO: Add test
def s3_file_exists(bucket_name: str, file_key: str) -> bool:
    """Check if a file exists in an S3 bucket."""
    s3_client = boto3.client("s3")

    try:
        s3_client.head_object(Bucket=bucket_name, Key=file_key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ["404", "403"]:
            return False
        raise


# TODO: Add test
def get_file_stems_for_document_id(
    document_id: DocumentImportId, bucket_name: str, prefix: str
) -> list[DocumentStem]:
    """
    Get the file stems for a document ID.

    This function is used to get the file stems for a document ID. For example we would
    find any translated documents in a directory for a document id as follows:

    Example:
    "CCLW.executive.1.1" -> ["CCLW.executive.1.1_translated_en", "CCLW.executive.1.1"]
    """
    stems = [document_id]

    for target_language in ["en"]:
        stem = f"{document_id}_translated_{target_language}"
        if s3_file_exists(
            bucket_name=bucket_name,
            file_key=f"{prefix}/{stem}.json",
        ):
            stems.append(stem)
    return stems
