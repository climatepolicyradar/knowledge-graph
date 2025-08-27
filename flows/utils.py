import asyncio
import functools
import inspect
import json
import os
import re
import textwrap
import time
from collections.abc import Awaitable, Generator, Sequence
from dataclasses import dataclass, field
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import (
    Annotated,
    Any,
    Callable,
    Literal,
    NewType,
    ParamSpec,
    TypeVar,
    overload,
)
from uuid import UUID

import boto3
from botocore.exceptions import ClientError
from prefect.artifacts import (
    create_progress_artifact,
    update_progress_artifact,
)
from prefect.client.schemas.objects import FlowRun, State, StateType
from prefect.deployments import run_deployment
from prefect.flows import Flow
from prefect.settings import PREFECT_UI_URL
from prefect.utilities.names import generate_slug
from prefect_slack.credentials import SlackWebhook
from pydantic import Field, PositiveInt, RootModel
from typing_extensions import Self

from scripts.cloud import (
    AwsEnv,
    ClassifierSpec,
    function_to_flow_name,
    generate_deployment_name,
)

T = TypeVar("T")
U = TypeVar("U")

JsonDict = NewType("JsonDict", dict[str, Any])

# Needed to get document passages from Vespa
# Example: CCLW.executive.1813.2418
DocumentImportId = NewType("DocumentImportId", str)
# Needed to load the inference results
# Example: s3://cpr-sandbox-data-pipeline-cache/labelled_passages/Q787/v4/CCLW.executive.1813.2418.json
DocumentObjectUri = NewType("DocumentObjectUri", str)
# A filename without the extension. May include a suffix indicating
# translation.
DocumentStem = NewType("DocumentStem", str)
# Passed to a self-sufficient flow run
DocumentImporter = NewType("DocumentImporter", tuple[DocumentStem, DocumentObjectUri])

DOCUMENT_ID_PATTERN = re.compile(r"^((?:[^.]+\.){3}[^._]+)")

DEFAULT_GPU_VM_TYPES: list[str] = [
    "g5.xlarge",
    "g6.xlarge",
    "g5.2xlarge",
    "g6.2xlarge",
]


def file_name_from_path(path: str) -> str:
    """Get the file name from a path without the path or extension"""
    return os.path.splitext(os.path.basename(path))[0]


class SlackNotify:
    """Notify a Slack channel through a Prefect Slack webhook."""

    # Must be ≤ this length
    MAX_SLACK_TEXT_LENGTH = 3000

    # Message templates
    FLOW_RUN_URL = "{prefect_base_url}/flow-runs/flow-run/{flow_run.id}"

    # Block name
    slack_channel_name = "alerts-platform"
    environment = AwsEnv(os.getenv("AWS_ENV", "sandbox"))
    slack_block_name = (
        f"slack-webhook-{slack_channel_name}-prefect-mvp-{environment.value}"
    )

    @classmethod
    async def message(
        cls,
        flow: Flow,
        flow_run: FlowRun,
        state: State,
    ):
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

        slack_webhook = SlackWebhook.load(cls.slack_block_name)
        if inspect.isawaitable(slack_webhook):
            slack_webhook = await slack_webhook

        blocks = cls.slack_blocks(flow, flow_run, state, ui_url)

        client = slack_webhook.get_client()
        result = client.send(
            blocks=blocks,
        )
        if inspect.isawaitable(result):
            result = await result

        print(
            f"Posted message to provided webhook: {result.status_code=} | {result.body=}"
        )

        return None

    @classmethod
    def slack_blocks(
        cls,
        flow: Flow,
        flow_run: FlowRun,
        state: State,
        ui_url: str,
    ):
        """Create all Slack Blocks"""

        header = f"{cls.state_type_to_emoji(state.type)} Flow run *{flow.name}/{flow_run.name}* observed state `{state.name}`."  # pyright: ignore[reportOptionalMemberAccess]

        state_message = textwrap.shorten(
            state.message or "No message",
            width=cls.MAX_SLACK_TEXT_LENGTH,
            placeholder="...",
        )

        return [
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": header,
                },
                "accessory": {
                    "type": "button",
                    "text": {
                        "type": "plain_text",
                        "text": "View in Prefect",
                        "emoji": True,
                    },
                    "value": "view_in_prefect",
                    "url": ui_url,
                    "action_id": "button-action",
                },
            },
            {"type": "divider"},
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Environment*\n`{cls.environment}`",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Version*\n`{flow_run.deployment_version}`",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Work Pool*\n`{flow_run.work_pool_name}`",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Timestamp*\n`{state.timestamp}`",
                    },
                    cls.slack_runtime_block(flow_run),
                ],
            },
            {"type": "divider"},
            {
                "type": "section",
                "expand": False,
                "text": {
                    "type": "mrkdwn",
                    "text": f"*State message:*\n\n>{state_message}",
                },
            },
        ]

    @staticmethod
    def slack_runtime_block(flow_run: FlowRun):
        """Create the runtime Slack Block"""

        match (flow_run.start_time, flow_run.end_time):
            case (start, end) if start is not None and end is not None:
                return {
                    "type": "mrkdwn",
                    "text": f"*Duration*\n`{flow_run.total_run_time}` from {flow_run.start_time} → {flow_run.end_time}",
                }
            case _:
                # At least one is imissing
                return {
                    "type": "mrkdwn",
                    "text": f"*Duration*\n`{flow_run.total_run_time}`",
                }

    @staticmethod
    def state_type_to_emoji(state_type: StateType) -> str:
        """Convert a Prefect StateType to an emoji."""
        match state_type:
            case StateType.SCHEDULED:
                return "⏰"
            case StateType.PENDING:
                return "⏳"
            case StateType.RUNNING:
                return "🏃"
            case StateType.COMPLETED:
                return "✅"
            case StateType.FAILED:
                return "❌"
            case StateType.CANCELLED:
                return "🚫"
            case StateType.CRASHED:
                return "💥"
            case StateType.PAUSED:
                return "⏸️"
            case StateType.CANCELLING:
                return "🛑"


class S3Uri:
    """A URI for an S3 object."""

    def __init__(self, bucket: str, key: str, protocol: str = "s3"):
        self.protocol = protocol
        self.bucket = bucket
        self.key = key

    def __str__(self) -> str:
        """Return the string representation of the S3 URI."""
        return f"{self.protocol}://{self.bucket}/{self.key}"

    @property
    def uri(self) -> str:
        """Return the string representation of the S3 URI."""
        return os.path.join(self.bucket, self.key)

    @property
    def stem(self) -> str:
        """Return the stem of the S3 URI (the key without the extension)."""
        return Path(self.key).stem


def remove_translated_suffix(file_name: DocumentStem) -> DocumentImportId:
    """
    Remove the suffix from a file name that indicates it has been translated.

    Often used for querying Vespa.

    E.g. "CCLW.executive.1.1_en_translated" -> "CCLW.executive.1.1"
    """
    return DocumentImportId(re.sub(r"(_translated(?:_[a-zA-Z]+)?)$", "", file_name))


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
        if e.response["Error"]["Code"] in [  # pyright: ignore[reportTypedDictNotRequiredAccess]
            "404",
            "403",
        ]:  # pyright: ignore[reportTypedDictNotRequiredAccess]
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

        if s3_file_exists(
            bucket_name=bucket_name,
            file_key=translated_file_key.__str__(),
        ):
            stems.append(translated_file_key.stem)

    if not stems:
        stems.append(document_id)

    return stems


def collect_unique_file_stems_under_prefix(
    bucket_name: str,
    prefix: str,
) -> list[DocumentStem]:
    """Collect all unique file stems under a prefix."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    file_stems = []
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            if obj["Key"].endswith(".json"):  # pyright: ignore[reportTypedDictNotRequiredAccess]
                file_stems.append(DocumentStem(Path(obj["Key"]).stem))  # pyright: ignore[reportTypedDictNotRequiredAccess]
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
    """Block waiting for a semaphore and then execute the function"""
    async with semaphore:
        return await fn


async def return_with(
    accompaniment: U,
    fn: Awaitable[T | Exception],
) -> tuple[U, T | Exception]:
    """Wrap a function's return value as a tuple with an accompanying value."""
    try:
        result = await fn
        return (accompaniment, result)
    except Exception as e:
        return (accompaniment, e)


# Match what Prefect uses for Flows:
#
# > .. we use the generic type variables `P` and `R` for "Parameters"
# > and "Returns" respectively.
P = ParamSpec("P")
R = TypeVar("R")


def fn_is_async(fn: Callable[..., Any] | Flow[P, R]) -> bool:
    """Check if a function is async."""
    if isinstance(fn, Flow):
        return fn.isasync  # type: ignore[reportFunctionMemberAccess]
    return inspect.iscoroutinefunction(fn)


class Percentage(
    RootModel[
        Annotated[
            float,
            Field(ge=0.0, le=100.0),
        ]
    ]
):
    """A percentage"""

    def __str__(self) -> str:
        """Return as string"""
        return f"{self.root}%"

    def __repr__(self) -> str:
        """Return as string representation"""
        return f"{self.__name__}({self.root})"

    def __float__(self) -> float:
        """Enable automatic conversion to float"""
        return self.root

    def to_float(self: Self) -> float:
        """Return as a float"""
        return float(self)

    @staticmethod
    def from_lists(r: Sequence[T], t: Sequence[U]) -> "Percentage":
        """Relative size of 2 lists as a percentage."""
        return Percentage((len(r) / len(t)) * 100.0)


@overload
async def map_as_sub_flow(
    fn: Flow[P, R],
    aws_env: AwsEnv,
    counter: PositiveInt,
    parameterised_batches: Generator[dict[str, Any], None, None],
    unwrap_result: Literal[True],
) -> tuple[Sequence[R], Sequence[BaseException | FlowRun]]: ...


@overload
async def map_as_sub_flow(
    fn: Flow[P, R],
    aws_env: AwsEnv,
    counter: PositiveInt,
    parameterised_batches: Generator[dict[str, Any], None, None],
    unwrap_result: Literal[False],
) -> tuple[Sequence[FlowRun], Sequence[BaseException | FlowRun]]: ...


async def map_as_sub_flow(
    fn: Flow[P, R],
    aws_env: AwsEnv,
    counter: PositiveInt,
    parameterised_batches: Generator[dict[str, Any], None, None],
    unwrap_result: bool,
) -> tuple[Sequence[R | FlowRun], Sequence[BaseException | FlowRun]]:
    """
    Map over an iterable, running the function as a sub-flow.

    The concurrency is limited to a semaphore with a counter.

    The results are grouped by success and failure, based on if an
    exception was returned or a flow run didn't complete, or some
    value was returned.

    The sub-flows are waited on until they complete, with no
    timeout.

    Either return the flow run itself, or unwrap the result from it.

    This assumes that the same parameters are used for each sub-flow run
    """
    flow_name = function_to_flow_name(fn.fn)
    deployment_name = generate_deployment_name(flow_name=flow_name, aws_env=aws_env)
    qualified_name = f"{flow_name}/{deployment_name}"
    semaphore = asyncio.Semaphore(counter)

    tasks = [
        wait_for_semaphore(
            semaphore,
            run_deployment(
                name=qualified_name,
                parameters=parameterised_batch,
                # Rely on the flow's own timeout, if any, to make sure it
                # eventually ends[1].
                #
                # [1]:
                # > Setting timeout to None will allow this function to
                # > poll indefinitely.
                timeout=None,
            ),
        )
        for parameterised_batch in parameterised_batches
    ]

    def desc_update_fn(tasks, results) -> str:
        return f"Finished sub-flow for {qualified_name}, progressing to {len(results)}/{len(tasks)} finished"

    results: Sequence[FlowRun | BaseException] = await gather_and_report(
        tasks=tasks,
        return_exceptions=True,
        key=f"progress-sub-flows-{generate_slug(2)}",
        desc_create=f"Starting sub-flows for {qualified_name} for {len(tasks)} tasks",
        desc_update_fn=desc_update_fn,
    )

    successes: list[R | FlowRun] = []
    failures: list[BaseException | FlowRun] = []
    for result in results:
        if isinstance(result, BaseException):
            failures.append(result)
        elif isinstance(result, FlowRun):
            if result.state and result.state.type == StateType.COMPLETED:
                # For completed flows, extract the actual return value
                try:
                    if unwrap_result:
                        result_fn = partial(
                            result.state.result,
                            # Doing it this way, makes it easier to rely
                            # on the type system, instead of doing `False`
                            # and then allowing for a union of types in
                            # the return.
                            raise_on_failure=True,
                        )
                        flow_result: R = (
                            await result_fn() if fn_is_async(fn) else result_fn()
                        )
                        successes.append(flow_result)
                    else:
                        successes.append(result)

                except Exception as e:
                    failures.append(e)
            else:
                failures.append(result)

    return successes, failures


@dataclass
class Fault(Exception):
    """A simple and generic exception with optional, helpful metadata"""

    msg: str
    metadata: dict[str, Any] | None
    data: Any | None = None

    def __str__(self) -> str:
        """Return a string representation"""
        if self.metadata is None:
            return self.msg
        try:
            data_str = str(self.data)
        except Exception as e:
            print(f"could not represent fault's data as a string: {e}")
            data_str = ""

        # Prefect logs should have no more than 25,000 characters, so truncate
        # the fault string if it's too long.
        message_str = textwrap.shorten(self.msg, width=8_000, placeholder="...")
        metadata_str = textwrap.shorten(
            json.dumps(self.metadata, default=str), width=8_000, placeholder="..."
        )
        data_str = textwrap.shorten(data_str, width=8_000, placeholder="...")

        fault_str = f"{message_str} | metadata: {metadata_str} | data: {data_str}"

        # Also truncate the total string as a safety precaution.
        fault_str = textwrap.shorten(fault_str, width=24_997, placeholder="...")

        return fault_str


def default_desc(tasks, results) -> str:
    return f"Finished task {len(results)} of {len(tasks)}"


@overload
async def gather_and_report(
    tasks: Sequence[Awaitable[T]],
    return_exceptions: Literal[True],
    key: str,
    desc_create: str,
    desc_update_fn: Callable[[Sequence[Any], list[Any]], str] = default_desc,
) -> Sequence[T | Exception]: ...


@overload
async def gather_and_report(
    tasks: Sequence[Awaitable[T]],
    return_exceptions: Literal[False],
    key: str,
    desc_create: str,
    desc_update_fn: Callable[[Sequence[Any], list[Any]], str] = default_desc,
) -> Sequence[T]: ...


async def gather_and_report(
    tasks: Sequence[Awaitable[T]],
    return_exceptions: bool,
    key: str,
    desc_create: str,
    desc_update_fn: Callable[[Sequence[Any], list[Any]], str] = default_desc,
) -> Sequence[T] | Sequence[T | Exception]:
    progress_artifact_id: UUID = await create_progress_artifact(  # pyright: ignore[reportGeneralTypeIssues]
        progress=0.0,
        key=key,
        description=desc_create,
    )

    results = []

    for future in asyncio.as_completed(tasks):
        try:
            result = await future
            results.append(result)
        except Exception as e:
            if return_exceptions:
                results.append(e)
            else:
                raise e
        finally:
            await update_progress_artifact(  # pyright: ignore[reportGeneralTypeIssues]
                artifact_id=progress_artifact_id,
                progress=Percentage.from_lists(results, tasks).to_float(),
                description=desc_update_fn(tasks, results),
            )

    return results
