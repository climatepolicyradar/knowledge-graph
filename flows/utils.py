import asyncio
import functools
import inspect
import os
import re
import time
from collections.abc import Generator, Sequence
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Callable, NewType, TypeVar

import boto3
from botocore.exceptions import ClientError
from prefect.settings import PREFECT_UI_URL
from prefect_slack.credentials import SlackWebhook
from typing_extensions import Self

from scripts.cloud import ClassifierSpec

# Needed to get document passages from Vespa
# Example: CCLW.executive.1813.2418
DocumentImportId = NewType("DocumentImportId", str)
# Needed to load the inference results
# Example: s3://cpr-sandbox-data-pipeline-cache/labelled_passages/Q787/v4/CCLW.executive.1813.2418.json
DocumentObjectUri = NewType("DocumentObjectUri", str)
# A filename without the extension
DocumentStem = NewType("DocumentStem", str)
# Passed to a self-sufficient flow run
DocumentImporter = NewType("DocumentImporter", tuple[DocumentStem, DocumentObjectUri])

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
    return DocumentImportId(re.sub(r"(_translated(?:_[a-zA-Z]+)?)$", "", file_name))


T = TypeVar("T")


def iterate_batch(
    data: Sequence[T] | Generator[T, None, None],
    batch_size: int,
) -> Generator[Sequence[T], None, None]:
    """Generate batches from a list or generator with a specified size."""
    if isinstance(data, Sequence):
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


def collect_unique_file_stems_under_prefix(
    bucket_name: str,
    prefix: str,
) -> list[DocumentImportId]:
    """Collect all unique file stems under a prefix."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    file_stems = []
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".json"):
                file_stems.append(Path(obj["Key"]).stem)
    return list(set(file_stems))


def get_labelled_passage_paths(
    document_ids: Sequence[DocumentImportId],
    classifier_specs: Sequence[ClassifierSpec],
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


@dataclass
class Profiler:
    """Context manager for profiling and printing the duration."""

    printer: Callable[[str], None] | None = None
    name: str | None = None
    # Set this so it's not `None` later on
    start_time: float = field(init=False, default_factory=time.perf_counter)
    end_time: float | None = field(init=False, default=None)
    duration: float | None = field(init=False, default=None)

    def __enter__(self) -> Self:
        """Start the timer."""
        self.start_time = time.perf_counter()  # Reset it now
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        """Stop the timer and conditionally print the duration."""
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time

        if self.printer:
            self.printer(
                (
                    f"{self.name + ' ' if self.name else ''}done "
                    f"in: {self.duration:.2f} seconds"
                )
            )

    def __call__(self, func):
        """Enable usage as a decorator for synchronous functions."""
        if asyncio.iscoroutinefunction(func):
            raise TypeError("Use AsyncProfiler for async functions")

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            profiler_name = self.name or func.__name__
            with Profiler(printer=self.printer, name=profiler_name):
                result = func(*args, **kwargs)
            return result

        return wrapper


@dataclass
class AsyncProfiler:
    """Async context manager for profiling and printing the duration."""

    printer: Callable[[str], None] | None = None
    name: str | None = None
    # Set this so it's not `None` later on
    start_time: float = field(init=False, default_factory=time.perf_counter)
    end_time: float | None = field(init=False, default=None)
    duration: float | None = field(init=False, default=None)

    async def __aenter__(self) -> Self:
        """Start the timer for async context."""
        self.start_time = time.perf_counter()
        return self

    async def __aexit__(self, _exc_type, _exc_value, _traceback) -> None:
        """Stop the timer and conditionally print the duration for async context."""
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time

        if self.printer:
            self.printer(
                (
                    f"{self.name + ' ' if self.name else ''}done "
                    f"in: {self.duration:.2f} seconds"
                )
            )

    def __call__(self, func):
        """Enable usage as a decorator for asynchronous functions."""
        if not asyncio.iscoroutinefunction(func):
            raise TypeError("Use Profiler for sync functions")

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            profiler_name = self.name or func.__name__
            async with AsyncProfiler(printer=self.printer, name=profiler_name):
                result = await func(*args, **kwargs)
            return result

        return async_wrapper


async def wait_for_semaphore(
    semaphore: asyncio.Semaphore,
    fn,
):
    async with semaphore:
        return await fn
