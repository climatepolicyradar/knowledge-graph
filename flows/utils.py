import inspect
import json
import os
import re
from collections.abc import Generator
from io import BytesIO
from pathlib import Path
from typing import TypeAlias, TypeVar

import boto3
from botocore.exceptions import ClientError
from prefect.settings import PREFECT_UI_URL
from prefect_slack.credentials import SlackWebhook

from scripts.cloud import ClassifierSpec

# Example: CCLW.executive.1813.2418
DocumentImportId: TypeAlias = str
DocumentStem: TypeAlias = str

DOCUMENT_ID_PATTERN = re.compile(r"^((?:[^.]+\.){3}[^._]+)")


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
    async def message(cls, flow, flow_run, state):
        """
        Send a notification to a Slack channel about the state of a Prefect flow run.

        Intended to be called from prefect flow hooks:

        ```python
        @flow(on_failure=[SlackNotify.message])
        def my_flow():
            pass
        ```
        """

        if cls.environment != "prod":
            return None

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
        if inspect.isawaitable(slack):
            slack = await slack
        result = slack.notify(body=msg)
        if inspect.isawaitable(result):
            _ = await result

        return None


def remove_translated_suffix(file_name: DocumentStem) -> DocumentImportId:
    """
    Remove the suffix from a file name that indicates it has been translated.

    Often used for querying Vespa.

    E.g. "CCLW.executive.1.1_en_translated" -> "CCLW.executive.1.1"
    """
    return re.sub(r"(_translated(?:_[a-zA-Z]+)?)$", "", file_name)


T = TypeVar("T")


def iterate_batch(
    data: list[T] | Generator[T, None, None],
    batch_size: int,
) -> Generator[list[T], None, None]:
    """Generate batches from a list or generator with a specified size."""
    if isinstance(data, list):
        # For lists, we can use list slicing
        for i in range(0, len(data), batch_size):
            yield data[i : i + batch_size]
    else:
        # For generators, accumulate items until we reach batch size
        batch: list[T] = []
        for item in data:
            batch.append(item)
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:  # Don't forget to yield the last partial batch
            yield batch


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


def get_file_stems_for_document_id(
    document_id: DocumentImportId, bucket_name: str, document_key: str
) -> list[DocumentStem]:
    """
    Get the file stems for a document ID.

    This function is used to get the file stems for a document ID. For example we would
    find any translated documents in a directory for a document id as follows:

    Example:
    "CCLW.executive.1.1" -> ["CCLW.executive.1.1_translated_en"]

    Note that we don't include the original document ID in the list of stems.
    This is because we are looking for only English language documents.
    """
    stems = []

    for target_language in ["en"]:
        translated_file_key = (
            Path(document_key)
            .with_stem(f"{document_id}_translated_{target_language}")
            .with_suffix(".json")
        )
        file_exists = s3_file_exists(
            bucket_name=bucket_name,
            file_key=translated_file_key.__str__(),
        )
        if file_exists:
            stems.append(translated_file_key.stem)

    if not stems:
        stems.append(document_id)

    return stems


def get_labelled_passage_paths(
    document_ids: list[DocumentImportId],
    classifier_specs: list[ClassifierSpec],
    cache_bucket: str,
    labelled_passages_prefix: str,
) -> list[str]:
    """
    Get document paths from a list of document IDs with translated paths if they exist.

    This function is used to get all document paths from a list of document IDs. For
    example CCLW.executive.1.1 is a document id that may have multiple files associated
    with it. This function will return all the paths to those files.

    Namely the translated versions of the file. This is done by checking whether a
    translated file exists in the target language.
    """

    document_paths = []

    for classifier_spec in classifier_specs:
        for document_id in document_ids:
            document_key = os.path.join(
                labelled_passages_prefix,
                classifier_spec.name,
                classifier_spec.alias,
                f"{document_id}.json",
            )

            document_stems = get_file_stems_for_document_id(
                document_id=document_id,
                bucket_name=cache_bucket,
                document_key=document_key,
            )

            for file_stem in document_stems:
                document_paths.append(
                    "s3://"
                    + os.path.join(
                        cache_bucket,
                        labelled_passages_prefix,
                        classifier_spec.name,
                        classifier_spec.alias,
                        f"{file_stem}.json",
                    )
                )

    return document_paths


def _s3_object_write_text(s3_uri: str, text: str) -> None:
    """Write text content to an S3 object."""
    # Parse the S3 URI
    s3_path: Path = Path(s3_uri)
    if len(s3_path.parts) < 3:
        raise ValueError(f"Invalid S3 path: {s3_path}")

    bucket: str = s3_path.parts[1]
    key = str(Path(*s3_path.parts[2:]))

    # Create BytesIO buffer with the text content
    body = BytesIO(text.encode("utf-8"))

    # Upload to S3
    s3 = boto3.client("s3")
    _ = s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")


def _s3_object_write_bytes(s3_uri: str, bytes: BytesIO) -> None:
    """Write text content to an S3 object."""
    # Parse the S3 URI
    s3_path: Path = Path(s3_uri)
    if len(s3_path.parts) < 3:
        raise ValueError(f"Invalid S3 path: {s3_path}")

    bucket: str = s3_path.parts[1]
    key = str(Path(*s3_path.parts[2:]))

    # Upload to S3
    s3 = boto3.client("s3")
    _ = s3.put_object(
        Bucket=bucket, Key=key, Body=bytes, ContentType="application/json"
    )


def is_file_stem_for_english_language_document(
    file_stem: DocumentStem,
    file_stems: list[DocumentStem],
    english_translation_suffix: str = "_translated_en",
) -> bool:
    """
    Check if a file stem is in English language.

    - If the file stem has the translated_en suffix then we can infer that it's english.
    - If there's a translated version of the document in the list, we can infer that it's
        not english.
    """
    if english_translation_suffix in file_stem:
        return True
    if file_stem + english_translation_suffix in file_stems:
        return False
    return True


def filter_non_english_language_file_stems(
    file_stems: list[DocumentStem],
) -> list[DocumentStem]:
    """Filter out file stems that are for non-English language documents."""
    return list(
        filter(
            lambda f: is_file_stem_for_english_language_document(f, file_stems),
            file_stems,
        )
    )


class S3FileStemFetcher:
    """Fetch file stems from S3"""

    def __init__(
        self,
        bucket_region: str,
        cache_bucket: str,
        document_source_prefix: str,
        pipeline_state_prefix: str,
        use_new_and_updated: bool,
        document_ids: None | list[DocumentImportId],
    ):
        self.bucket_region = bucket_region
        self.cache_bucket = cache_bucket
        self.document_source_prefix = document_source_prefix
        self.pipeline_state_prefix = pipeline_state_prefix
        self.use_new_and_updated = use_new_and_updated
        self.document_ids = document_ids

    def get_bucket_paginator(self, prefix: str):
        """Returns an s3 paginator for the pipeline cache bucket"""
        s3 = boto3.client("s3", region_name=self.bucket_region)
        paginator = s3.get_paginator("list_objects_v2")
        return paginator.paginate(
            Bucket=self.cache_bucket,
            Prefix=prefix,
        )

    def list_bucket_file_stems(self) -> list[DocumentImportId]:
        """
        Scan configured bucket and return all file stems.

        Where a stem refers to a file name without the extension. Often, this is the same as
        the document id, but not always as we have translated documents.
        """
        page_iterator = self.get_bucket_paginator(self.document_source_prefix)
        file_stems = []

        for p in page_iterator:
            if "Contents" in p:
                for o in p["Contents"]:
                    file_stem = Path(o["Key"]).stem
                    file_stems.append(file_stem)

        return file_stems

    def determine_file_stems(
        self,
        use_new_and_updated: bool,
        requested_document_ids: None | list[DocumentImportId],
        current_bucket_file_stems: list[DocumentImportId],
    ) -> list[DocumentImportId]:
        """
        Function for identifying the file stems to process.

        File stems refer to the file name without the extension. Often, this is the same as
        the document id, but not always as we have translated documents.

        Compares the requested_document_ids to what actually exists in the bucket.
        If a document id has been requested but does not exist this will
        raise a `ValueError`. If no document ids were requested, this will
        instead return the `current_bucket_file_stems`.

        For requested document ids we identify whether there are any translated files that
        should also be processed by identifying their file stems as well.
        """
        if use_new_and_updated and requested_document_ids:
            raise ValueError(
                "`use_new_and_updated`, and `requested_document_ids` are mutually exclusive"
            )
        elif use_new_and_updated:
            requested_document_ids = self.get_latest_ingest_documents()
        elif requested_document_ids is None:
            current_bucket_file_stems__filtered = (
                filter_non_english_language_file_stems(
                    file_stems=current_bucket_file_stems
                )
            )
            return current_bucket_file_stems__filtered

        requested_document_stems = []
        for doc_id in requested_document_ids:
            document_key = os.path.join(self.document_source_prefix, f"{doc_id}.json")
            requested_document_stems += get_file_stems_for_document_id(
                doc_id, self.cache_bucket, document_key
            )

        missing_from_bucket = list(
            set(requested_document_stems) - set(current_bucket_file_stems)
        )
        if len(missing_from_bucket) > 0:
            raise ValueError(
                f"Requested document_ids not found in bucket: {missing_from_bucket}"
            )

        return requested_document_stems

    def get_latest_ingest_documents(self) -> list[str]:
        """
        Get IDs of changed documents from the latest ingest run

        Retrieves the `new_and_updated_docs.json` file from the latest ingest.
        Extracts the ids from the file, and returns them as a single list.
        """
        page_iterator = self.get_bucket_paginator(self.pipeline_state_prefix)
        file_name = "new_and_updated_documents.json"

        # First get all matching files, then sort them
        matching_files = [
            item
            for item in page_iterator.search(f"Contents[?contains(Key, '{file_name}')]")
            if item is not None
        ]

        if not matching_files:
            raise ValueError(
                f"failed to find any `{file_name}` files in "
                f"`{self.cache_bucket}/{self.pipeline_state_prefix}`"
            )

        # Sort by Key and get the last one
        latest = sorted(matching_files, key=lambda x: x["Key"])[-1]

        data = download_s3_file(self.bucket_region, self.cache_bucket, latest["Key"])
        content = json.loads(data)
        updated = list(content["updated_documents"].keys())
        new = [d["import_id"] for d in content["new_documents"]]

        print(
            f"Retrieved {len(new)} new, and {len(updated)} updated from {latest['Key']}"
        )
        return new + updated

    def remove_sabin_file_stems(
        self, file_stems: list[DocumentImportId]
    ) -> list[DocumentImportId]:
        """
        Remove Sabin document file stems from the list of file stems.

        File stems of the Sabin source follow the below naming convention:
        - "Sabin.document.16944.17490"
        """
        return [
            stem
            for stem in file_stems
            if not stem.startswith(("Sabin", "sabin", "SABIN"))
        ]

    def fetch(self) -> list[DocumentImportId]:
        """Fetch file stems for a list of document IDs"""
        current_bucket_file_stems = self.list_bucket_file_stems()

        validated_file_stems = self.determine_file_stems(
            use_new_and_updated=self.use_new_and_updated,
            requested_document_ids=self.document_ids,
            current_bucket_file_stems=current_bucket_file_stems,
        )
        filtered_file_stems = self.remove_sabin_file_stems(validated_file_stems)
        return filtered_file_stems


def download_s3_file(bucket_region: str, cache_bucket: str, key: str):
    """Retrieve an s3 file from the pipeline cache"""

    s3 = boto3.client("s3", region_name=bucket_region)
    response = s3.get_object(Bucket=cache_bucket, Key=key)
    content = response["Body"].read().decode("utf-8")
    return content
