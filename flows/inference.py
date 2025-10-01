import asyncio
import json
import os
from collections import defaultdict
from collections.abc import Generator, Sequence
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Final, NamedTuple, Optional, TypeAlias

import aioboto3
import tenacity
import wandb
from botocore.exceptions import ClientError
from cpr_sdk.parser_models import BaseParserOutput, BlockType
from mypy_boto3_s3.type_defs import (
    ObjectTypeDef,
    PutObjectOutputTypeDef,
)
from prefect import flow
from prefect.artifacts import acreate_table_artifact
from prefect.concurrency.asyncio import concurrency
from prefect.context import FlowRunContext, get_run_context
from prefect.exceptions import MissingContextError
from prefect.utilities.names import generate_slug
from pydantic import BaseModel, ConfigDict, PositiveInt, SecretStr, ValidationError
from tenacity import RetryCallState
from types_aiobotocore_s3.client import S3Client
from wandb.sdk.wandb_run import Run

from flows.classifier_specs.spec_interface import (
    ClassifierSpec,
    disallow_latest_alias,
    load_classifier_specs,
    should_skip_doc,
)
from flows.config import Config
from flows.utils import (
    DocumentImportId,
    DocumentStem,
    Fault,
    JsonDict,
    ParameterisedFlow,
    Profiler,
    S3Uri,
    SlackNotify,
    filter_non_english_language_file_stems,
    get_file_stems_for_document_id,
    get_logger,
    iterate_batch,
    map_as_sub_flow,
    return_with,
    wait_for_semaphore,
)
from knowledge_graph.classifier import Classifier, ModelPath
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.span import Span

# The "parent" AKA the higher level flows that do multiple things
PARENT_TIMEOUT_S: int = int(timedelta(hours=12).total_seconds())
# A singular task doing one thing
TASK_TIMEOUT_S: int = int(timedelta(minutes=60).total_seconds())

# NOTE: Comparable list being maintained at https://github.com/climatepolicyradar/navigator-search-indexer/blob/91e341b8a20affc38cd5ce90c7d5651f21a1fd7a/src/config.py#L13.
BLOCKED_BLOCK_TYPES: Final[set[BlockType]] = {
    BlockType.PAGE_NUMBER,
    BlockType.TABLE,
    BlockType.FIGURE,
}

CLASSIFIER_CONCURRENCY_LIMIT: Final[PositiveInt] = 20
INFERENCE_BATCH_SIZE_DEFAULT: Final[PositiveInt] = 1000
AWS_ENV: str = os.environ["AWS_ENV"]
S3_BLOCK_RESULTS_CACHE: str = f"s3-bucket/cpr-{AWS_ENV}-prefect-results-cache"

DocumentRunIdentifier: TypeAlias = tuple[str, str, str]
FilterResult = NamedTuple(
    "FilterResult",
    [("removed", Sequence[DocumentStem]), ("accepted", Sequence[DocumentStem])],
)


class BatchInferenceResult(BaseModel):
    """Result from running inference on a batch of documents."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    batch_document_stems: list[DocumentStem]
    """List of document stems that were included in this batch for processing.
    
    These represent all the documents that were assigned to this batch,
    regardless of whether processing succeeded or failed.
    """

    successful_document_stems: list[DocumentStem]
    """List of document stems that were processed successfully in this batch."""

    classifier_spec: ClassifierSpec
    """The classifier specification used to process this batch of documents."""

    @property
    def all_document_count(self) -> int:
        """Count of all document stems"""
        return len(self.batch_document_stems)

    @property
    def failed_document_count(self) -> int:
        """Count of failed document stems"""
        return len(self.failed_document_stems)

    @property
    def failed_document_stems(self) -> list[DocumentStem]:
        """List of requested document stems that where not successful."""
        return list(
            set(self.batch_document_stems) - set(self.successful_document_stems)
        )

    @property
    def failed(self) -> bool:
        """Whether the batch failed, True if failed."""

        return len(self.batch_document_stems) != len(self.successful_document_stems)


class InferenceParams(BaseModel):
    """Parameters for batch level inference."""

    batch: Sequence[DocumentStem]
    config_json: JsonDict
    classifier_spec_json: JsonDict


class InferenceResult(BaseModel):
    """Result from running inference on all batches of documents."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    classifier_specs: list[ClassifierSpec]
    """List of classifier specifications that were used in this inference run.
    
    These define which classifiers (models) were intended to be used for processing
    the documents, regardless of whether they succeeded or failed.
    """

    batch_inference_results: list[BatchInferenceResult] = []
    """All the batches that made up this inference run."""

    successful_classifier_specs: list[ClassifierSpec] = []
    """List of classifier specifications that completed all processing successfully."""

    failed_classifier_specs: list[ClassifierSpec] = []
    """List of classifier specifications that failed for one or more document."""

    parameterised_batches: list[ParameterisedFlow]
    """List of parameterised batches for inference."""

    @property
    def inference_all_document_stems(self) -> set[DocumentStem]:
        """All documents stems sent to batch level inference."""
        all_documents = set()
        for parameterised_batch in self.parameterised_batches:
            all_documents.update(
                InferenceParams.model_validate(parameterised_batch.params).batch
            )

        return all_documents

    @property
    def failed(self) -> bool:
        """Whether the inference failed."""

        # Check if no batch results
        if not self.batch_inference_results:
            return True

        # Check if any batch failed
        if any(result.failed for result in self.batch_inference_results):
            return True

        # Check if document counts don't match
        if len(self.inference_all_document_stems) != len(
            self.successful_document_stems
        ):
            return True

        return False

    @property
    def successful_document_stems(self) -> set[DocumentStem]:
        """
        The documents that succeeded for every classifier they were expected to run on.

        This means removing any that had a failure in any batch or no results.
        """

        # Collect the successful document stems for each classifier.
        successes_by_classifier: dict[ClassifierSpec, set[DocumentStem]] = defaultdict(
            set
        )
        for batch_inference_result in self.batch_inference_results:
            successes_by_classifier[batch_inference_result.classifier_spec].update(
                batch_inference_result.successful_document_stems
            )

        # Collect the unsuccessful document stems where an expected success for a classifier
        #  was not found.
        failed_document_stems: set[DocumentStem] = set()
        for parameterised_batch in self.parameterised_batches:
            params = InferenceParams.model_validate(parameterised_batch.params)
            expected_document_stems = params.batch
            classifier_spec = ClassifierSpec.model_validate(params.classifier_spec_json)

            failed_document_stems.update(
                set(expected_document_stems) - successes_by_classifier[classifier_spec]
            )

        # Collect the successful document stems as the sum of the expected document stems
        #   minus the unsuccessful document stems.
        successful_documents = self.inference_all_document_stems - failed_document_stems

        return successful_documents


async def get_bucket_paginator(config: Config, prefix: str, s3_client: S3Client):
    """Returns an S3 paginator for the pipeline cache bucket"""
    paginator = s3_client.get_paginator("list_objects_v2")
    return paginator.paginate(
        Bucket=config.cache_bucket,  # pyright: ignore[reportArgumentType]
        Prefix=prefix,
    )


async def list_bucket_file_stems(config: Config) -> list[DocumentStem]:
    """
    Scan configured bucket and return all file stems.

    Where a stem refers to a file name without the extension. Often, this is the same as
    the document id, but not always as we have translated documents.
    """
    session = aioboto3.Session(region_name=config.bucket_region)
    async with session.client("s3") as s3_client:
        page_iterator = await get_bucket_paginator(
            config, config.inference_document_source_prefix, s3_client
        )
        file_stems = []

        async for p in page_iterator:
            if "Contents" in p:
                for o in p["Contents"]:
                    file_stem = Path(o["Key"]).stem  # pyright: ignore[reportTypedDictNotRequiredAccess]
                    file_stems.append(file_stem)
    return file_stems


async def get_latest_ingest_documents(config: Config) -> Sequence[DocumentImportId]:
    """
    Get IDs of changed documents from the latest ingest run

    Retrieves the `new_and_updated_docs.json` file from the latest ingest.
    Extracts the ids from the file, and returns them as a single list.
    """
    logger = get_logger()
    session = aioboto3.Session(region_name=config.bucket_region)
    async with session.client("s3") as s3_client:
        page_iterator = await get_bucket_paginator(
            config, config.pipeline_state_prefix, s3_client
        )
        file_name = "new_and_updated_documents.json"

        # First get all matching files, then sort them
        matching_files: list[ObjectTypeDef] = []

        # Iterate through pages and extract the "Contents" list
        async for page in page_iterator:
            if "Contents" in page:
                contents: list[ObjectTypeDef] = page[
                    "Contents"
                ]  # Explicitly type the contents
                matching_files.extend(contents)

    # Filter files that contain the target file name in their "Key"
    filtered_files = [
        item for item in matching_files if "Key" in item and file_name in item["Key"]
    ]

    if not filtered_files:
        raise ValueError(
            f"failed to find any `{file_name}` files in "
            f"`{config.cache_bucket}/{config.pipeline_state_prefix}`"
        )

    # Sort by "Key" and get the last one
    latest = sorted(filtered_files, key=lambda x: x["Key"])[-1]

    latest_key = latest["Key"]
    session = aioboto3.Session(region_name=config.bucket_region)
    async with session.client("s3") as s3_client:
        data = await download_s3_file(config, latest_key, s3_client)
        content = json.loads(data)
        updated = list(content["updated_documents"].keys())
        new = [d["import_id"] for d in content["new_documents"]]

    logger.info(
        f"Retrieved {len(new)} new, and {len(updated)} updated from {latest_key}"
    )
    return new + updated


async def determine_file_stems(
    config: Config,
    use_new_and_updated: bool,
    requested_document_ids: Optional[Sequence[DocumentImportId]],
    current_bucket_file_stems: list[DocumentStem],
) -> list[DocumentStem]:
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
        requested_document_ids = await get_latest_ingest_documents(config)
    elif requested_document_ids is None:
        current_bucket_file_stems__filtered = filter_non_english_language_file_stems(
            file_stems=current_bucket_file_stems
        )
        return current_bucket_file_stems__filtered

    assert config.cache_bucket

    requested_document_stems = []

    session = aioboto3.Session(region_name=config.bucket_region)
    async with session.client("s3") as s3_client:
        for doc_id in requested_document_ids:
            document_key = os.path.join(
                config.inference_document_source_prefix, f"{doc_id}.json"
            )
            requested_document_stems += await get_file_stems_for_document_id(
                doc_id, config.cache_bucket, document_key, s3_client
            )

    missing_from_bucket = list(
        set(requested_document_stems) - set(current_bucket_file_stems)
    )
    if len(missing_from_bucket) > 0:
        raise ValueError(
            f"Requested document_ids not found in bucket: {missing_from_bucket}"
        )

    return requested_document_stems


async def load_classifier(
    run: Run, config: Config, classifier_spec: ClassifierSpec
) -> Classifier:
    """Load a classifier into memory."""
    async with concurrency("load_classifier", occupy=5):
        wandb_classifier_path = ModelPath(
            wikibase_id=classifier_spec.wikibase_id,
            classifier_id=classifier_spec.classifier_id,
        )
        artifact_id = f"{wandb_classifier_path}:{config.aws_env}"
        artifact = run.use_artifact(artifact_id, type="model")
        download_folder = artifact.download()
        model_path = Path(download_folder) / "model.pickle"
        classifier = Classifier.load(model_path)
    return classifier


def parse_client_error_details(e: ClientError) -> Optional[str]:
    """
    Return extra context for AWS `ClientError`s.

    Intended to be extendable for specific Errors, and to get extra details not
    normally covered by just raising the error.
    """
    error = e.response.get("Error", {})
    code = error.get("Code")
    if code == "RequestTimeTooSkewed":
        request_time = error.get("RequestTime")
        server_time = error.get("ServerTime")
        if request_time and server_time:
            skew = datetime.fromisoformat(server_time) - datetime.fromisoformat(
                request_time
            )
            return f"Request-Server time discrepancy: {' & '.join(e.args)} - {skew.seconds=}"


async def download_s3_file(config: Config, key: str, s3_client: S3Client):
    """Retrieve an S3 file from the pipeline cache"""
    try:
        response = await s3_client.get_object(
            Bucket=config.cache_bucket,  # pyright: ignore[reportArgumentType]
            Key=key,
        )
    except ClientError as e:
        if extra_context := parse_client_error_details(e):
            e.add_note(f"{extra_context}, key: {key}")
        raise
    body = await response["Body"].read()
    return body.decode("utf-8")


def generate_document_source_key(config: Config, document_stem: DocumentStem) -> S3Uri:
    return S3Uri(
        bucket=config.cache_bucket,  # pyright: ignore[reportArgumentType]
        key=os.path.join(
            config.inference_document_source_prefix,
            f"{document_stem}.json",
        ),
    )


async def load_document(
    config: Config, file_stem: DocumentStem, s3_client: S3Client
) -> BaseParserOutput:
    """Download and opens a parser output based on a document ID."""
    file_key = generate_document_source_key(
        config=config,
        document_stem=file_stem,
    ).key
    content = await download_s3_file(config=config, key=file_key, s3_client=s3_client)
    document = BaseParserOutput.model_validate_json(content)
    return document


def _stringify(text: list[str]) -> str:
    return " ".join([line.strip() for line in text])


def document_passages(
    document: BaseParserOutput,
) -> Generator[tuple[str, str], None, None]:
    """Yield the text block irrespective of content type."""
    text_blocks = document.get_text_blocks()

    for text_block in text_blocks:
        if text_block.type not in BLOCKED_BLOCK_TYPES:
            yield _stringify(text_block.text), text_block.text_block_id


def serialise_pydantic_list_as_jsonl[T: BaseModel](models: Sequence[T]) -> BytesIO:
    """
    Serialize a list of Pydantic models as JSONL (JSON Lines) format.

    Each model is serialized on a separate line using model_dump_json().
    """
    jsonl_content = "\n".join(model.model_dump_json() for model in models)
    return BytesIO(jsonl_content.encode("utf-8"))


def deserialise_pydantic_list_from_jsonl[T: BaseModel](
    jsonl_content: str, model_class: type[T]
) -> list[T]:
    """
    Deserialize JSONL (JSON Lines) format to a list of Pydantic models.

    Each line should contain a JSON object that can be parsed by the model_class.
    """
    models = []
    for line in jsonl_content.strip().split("\n"):
        if line.strip():  # Skip empty lines
            model = model_class.model_validate_json(line)
            models.append(model)
    return models


def deserialise_pydantic_list_with_fallback[T: BaseModel](
    content: str, model_class: type[T]
) -> list[T]:
    """
    Deserialize content to a list of Pydantic models with fallback support.

    First tries JSONL format, then falls back to original format (JSON array of JSON strings).
    """
    # Try JSONL format first
    try:
        return deserialise_pydantic_list_from_jsonl(content, model_class)
    except ValidationError:
        # Fall back to original format (array of JSON strings)
        data = json.loads(content)
        return [model_class.model_validate_json(passage) for passage in data]


class SingleDocumentInferenceResult(BaseModel):
    """Labelled passages from inference on a single document."""

    labelled_passages: Sequence[LabelledPassage]
    document_stem: DocumentStem
    wikibase_id: str
    classifier_id: str


def generate_s3_uri_output(
    config: Config, inference: SingleDocumentInferenceResult
) -> S3Uri:
    return S3Uri(
        bucket=config.cache_bucket,  # pyright: ignore[reportArgumentType]
        key=os.path.join(
            config.inference_document_target_prefix,
            inference.wikibase_id,
            inference.classifier_id,
            f"{inference.document_stem}.json",
        ),
    )


async def labels_to_s3(
    config: Config,
    inference: SingleDocumentInferenceResult,
    s3_client: S3Client,
) -> PutObjectOutputTypeDef:
    logger = get_logger()
    s3_uri = generate_s3_uri_output(config, inference)

    logger.info(f"Storing labels for document {inference.document_stem} at {s3_uri}")

    body = serialise_pydantic_list_as_jsonl(inference.labelled_passages)

    response = await s3_client.put_object(
        Bucket=s3_uri.bucket,
        Key=s3_uri.key,
        Body=body,
        ContentType="application/json",
    )

    return response


async def store_labels(
    config: Config,
    inferences: Sequence[SingleDocumentInferenceResult],
) -> tuple[
    list[SingleDocumentInferenceResult],
    list[tuple[DocumentStem, Exception]],
    list[BaseException],
]:
    """Store the labels in the cache bucket."""
    logger = get_logger()

    # Don't get rate-limited by AWS
    semaphore = asyncio.Semaphore(config.s3_concurrency_limit)

    session = aioboto3.Session(region_name=config.bucket_region)
    async with session.client("s3") as s3_client:
        tasks = [
            wait_for_semaphore(
                semaphore,
                return_with(
                    inference,
                    labels_to_s3(config, inference, s3_client),
                ),
            )
            for inference in inferences
        ]

        results: list[
            tuple[SingleDocumentInferenceResult, Exception | PutObjectOutputTypeDef]
            | BaseException
        ] = await asyncio.gather(*tasks, return_exceptions=True)

    successes: list[SingleDocumentInferenceResult] = []
    failures: list[tuple[DocumentStem, Exception]] = []
    # We really don't expect these, since there's a try/catch handler
    # in `return_with_id`. It is technically possible though, for
    # there to be what I'm calling here an _unknown_ failure.
    unknown_failures: list[BaseException] = []
    for result in results:
        if isinstance(result, BaseException):
            unknown_failures.append(result)
        else:
            inference, value = result
            if isinstance(value, Exception):
                logger.error(
                    f"Failed to store label for {inference.document_stem}: {value}"
                )
                failures.append((inference.document_stem, value))
            else:
                if value["ResponseMetadata"]["HTTPStatusCode"] == 200:
                    successes.append(inference)
                else:
                    failures.append((inference.document_stem, ValueError(str(value))))

    return successes, failures, unknown_failures


def batch_text_block_inference(
    classifier: Classifier,
    all_text: list[str],
    all_block_ids: list[str],
    batch_size: int = 10,
) -> list[LabelledPassage]:
    """Runs inference and batches the text blocks"""

    outputs = []
    for batch_idx in range(0, len(all_text), batch_size):
        text_batch = all_text[batch_idx : batch_idx + batch_size]
        block_ids = all_block_ids[batch_idx : batch_idx + batch_size]

        outputs.extend(
            _text_block_inference_for_single_batch(
                classifier=classifier, text_batch=text_batch, block_ids=block_ids
            )
        )
    return outputs


def _text_block_inference_for_single_batch(
    classifier: Classifier, text_batch: list[str], block_ids: list[str]
) -> list[LabelledPassage]:
    """Runs predict on a batch of blocks."""
    spans: list[list[Span]] = classifier.predict_batch(text_batch)

    labelled_passages = [
        _get_labelled_passage_from_prediction(classifier, spans, block_id, text)
        for spans, block_id, text in zip(spans, block_ids, text_batch)
    ]

    return labelled_passages


def text_block_inference(
    classifier: Classifier, block_id: str, text: str
) -> LabelledPassage:
    """Run predict on a single text block."""
    spans: list[Span] = classifier.predict(text)

    if spans_missing_timestamps := [span for span in spans if not span.timestamps]:
        span_ids = ",".join(str(span.id) for span in spans_missing_timestamps)
        raise ValueError(
            f"Found {len(spans_missing_timestamps)} span(s) with missing timestamps. "
            f"Span IDs: {span_ids}"
        )

    if spans_missing_labellers := [span for span in spans if not span.labellers]:
        span_ids = ",".join(str(span.id) for span in spans_missing_labellers)
        raise ValueError(
            f"Found {len(spans_missing_labellers)} span(s) with missing labellers. "
            f"Span IDs: {span_ids}"
        )

    if spans_mismatched_lengths := [
        span for span in spans if len(span.timestamps) != len(span.labellers)
    ]:
        mismatched_info = ",".join(
            f"{span.id} (timestamps: {len(span.timestamps)}, labellers: {len(span.labellers)})"
            for span in spans_mismatched_lengths
        )
        raise ValueError(
            f"Found {len(spans_mismatched_lengths)} span(s) with mismatched timestamp/labeller lengths. "
            f"Details: {mismatched_info}"
        )

    labelled_passage = _get_labelled_passage_from_prediction(
        classifier, spans, block_id, text
    )

    return labelled_passage


def _get_labelled_passage_from_prediction(
    classifier: Classifier, spans: list[Span], block_id: str, text: str
) -> LabelledPassage:
    """Creates the LabelledPassage from the list of spans output by the classifier"""
    # If there were no inference results, don't include the concept
    if not spans:
        metadata = {}
    else:
        # Remove the labelled passages from the concept to reduce the
        # size of the metadata.
        concept_no_labelled_passages = classifier.concept.model_copy(
            update={"labelled_passages": []}
        )

        concept = concept_no_labelled_passages.model_dump()

        metadata = {"concept": concept}

    return LabelledPassage(
        id=block_id,
        text=text,
        spans=spans,
        metadata=metadata,
    )


def retry_callback(retry_state: RetryCallState):
    """Log a message about retries progress."""
    logger = get_logger()

    fn_name = retry_state.fn.__name__ if retry_state.fn else "unknown"
    attempt = retry_state.attempt_number
    if outcome := retry_state.outcome:
        notes = None
        if exception := outcome.exception():
            if hasattr(exception, "__notes__"):
                notes = ", ".join(exception.__notes__)

            logger.warning(
                f"{fn_name} retry #{attempt}. Error notes: {notes or exception})"
            )


# https://tenacity.readthedocs.io/en/latest/#waiting-before-retrying
@tenacity.retry(
    wait=tenacity.wait_fixed(15) + tenacity.wait_random(0, 45),  # each wait = 15-60s
    stop=tenacity.stop_after_attempt(2),
    retry=tenacity.retry_if_exception_type(ClientError)
    | tenacity.retry_if_exception_message(match="RequestTimeTooSkewed"),
    after=retry_callback,
    reraise=True,
)
async def run_classifier_inference_on_document(
    config: Config,
    file_stem: DocumentStem,
    classifier: Classifier,
    s3_client: S3Client,
) -> SingleDocumentInferenceResult:
    """Run the classifier inference flow on a document."""
    document = await load_document(config, file_stem, s3_client)

    # Resolve typing issue as wikibase_id is optional (though required here)
    assert classifier.concept.wikibase_id, f"Classifier invalid: {classifier.id}"

    # Don't run inference on documents that have no text or languages as well as HTML
    # documents with no valid text.
    no_text_and_no_languages: bool = (
        not document.languages
        and document.pdf_data is None
        and document.html_data is None
    )
    html_and_invalid_text: bool = (
        document.html_data is not None and not document.html_data.has_valid_text
    )
    if no_text_and_no_languages or html_and_invalid_text:
        return SingleDocumentInferenceResult(
            labelled_passages=[],
            document_stem=file_stem,
            wikibase_id=classifier.concept.wikibase_id,
            classifier_id=classifier.id,
        )

    # Raise on non-English documents
    if document.languages != ["en"]:
        raise ValueError(
            f"Cannot run inference on {file_stem} as it has non-English language: "
            f"{document.languages}"
        )

    doc_labels: list[LabelledPassage] = []
    for text, block_id in document_passages(document):
        labelled_passages = text_block_inference(
            classifier=classifier, block_id=block_id, text=text
        )
        doc_labels.append(labelled_passages)

    return SingleDocumentInferenceResult(
        labelled_passages=doc_labels,
        document_stem=file_stem,
        wikibase_id=classifier.concept.wikibase_id,
        classifier_id=classifier.id,
    )


async def create_inference_on_batch_summary_artifact(
    successes: Sequence[SingleDocumentInferenceResult],
    failures: Sequence[tuple[DocumentStem, Exception]],
    unknown_failures: Sequence[BaseException],
    flow_run_name: str | None,
):
    """Create an artifact with a summary about a batch inference run."""

    total_documents = len(successes) + len(failures) + len(unknown_failures)
    successful_documents = len(successes)
    failed_documents = len(failures)
    unknown_failures_count = len(unknown_failures)

    overview_description = f"""# Batch Inference Summary

## Overview
- **Flow Run**: {flow_run_name or "Unknown"}
- **Total documents processed**: {total_documents}
- **Successful documents**: {successful_documents}
- **Failed documents**: {failed_documents}
- **Unknown failures**: {unknown_failures_count}
"""

    document_details = (
        [
            {
                "Document stem": single_document_inference_result.document_stem,
                "Status": "✓",
                "Exception": "N/A",
            }
            for single_document_inference_result in successes
        ]
        + [
            {
                "Document stem": document_stem,
                "Status": "✗",
                "Exception": str(exc),
            }
            for document_stem, exc in failures
        ]
        + [
            {
                "Document stem": "Unknown",
                "Status": "✗",
                "Exception": str(exc),
            }
            for exc in unknown_failures
        ]
    )

    if not flow_run_name:
        flow_run_name = f"unknown-{generate_slug(2)}"

    await acreate_table_artifact(
        key=f"batch-inference-{flow_run_name}",
        table=document_details,
        description=overview_description,
    )


async def _inference_batch_of_documents(
    batch: list[DocumentStem],
    config_json: JsonDict,
    classifier_spec_json: JsonDict,
) -> BatchInferenceResult | Fault:
    """
    Run classifier inference on a batch of documents.

    This reflects the unit of work that should be run in one of many
    parallelised Docker containers.
    """
    logger = get_logger()

    config_json["wandb_api_key"] = (
        SecretStr(config_json["wandb_api_key"])
        if config_json["wandb_api_key"]
        else None
    )
    config_json["local_classifier_dir"] = Path(config_json["local_classifier_dir"])
    config = Config(**config_json)

    wandb.login(key=config.wandb_api_key.get_secret_value())  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]
    run = wandb.init(  # pyright: ignore[reportAttributeAccessIssue]
        entity=config.wandb_entity,
        job_type="concept_inference",
    )

    classifier_spec = ClassifierSpec(**classifier_spec_json)
    logger.info(f"Loading classifier {classifier_spec}")
    classifier = await load_classifier(run, config, classifier_spec)

    semaphore = asyncio.Semaphore(config.s3_concurrency_limit)

    session = aioboto3.Session(region_name=config.bucket_region)
    async with session.client("s3") as s3_client:
        tasks = [
            wait_for_semaphore(
                semaphore,
                return_with(
                    file_stem,
                    run_classifier_inference_on_document(
                        config=config,
                        file_stem=file_stem,
                        classifier=classifier,
                        s3_client=s3_client,
                    ),
                ),
            )
            for file_stem in batch
        ]

        results: list[
            tuple[DocumentStem, Exception | SingleDocumentInferenceResult]
            | BaseException
        ] = await asyncio.gather(*tasks, return_exceptions=True)

    inferences_successes: list[SingleDocumentInferenceResult] = []
    inferences_failures: list[tuple[DocumentStem, Exception]] = []
    # We really don't expect these, since there's a try/catch handler
    # in `return_with_id`. It is technically possible though, for
    # there to be what I'm calling here an _unknown_ failure.
    inferences_unknown_failures: list[BaseException] = []
    for result in results:
        if isinstance(result, BaseException):
            inferences_unknown_failures.append(result)
        else:
            document_stem, value = result
            if isinstance(value, Exception):
                logger.error(f"Failed to process document {document_stem}: {value}")
                inferences_failures.append((document_stem, value))
            else:
                inferences_successes.append(value)

    (
        store_labels_successes,
        store_labels_failures,
        store_labels_unknown_failures,
    ) = await store_labels(  # pyright: ignore[reportFunctionMemberAccess, reportArgumentType]
        config=config, inferences=inferences_successes
    )

    # This doesn't need to be combined since successes are funnelled
    # through all steps.
    #
    # Failures are possibly reduced at each step.
    all_successes = store_labels_successes
    # Combine the multiple places that have reports
    all_failures = inferences_failures + store_labels_failures
    all_unknown_failures = inferences_unknown_failures + store_labels_unknown_failures

    # https://docs.prefect.io/v3/concepts/runtime-context#access-the-run-context-directly
    try:
        run_context = get_run_context()
        flow_run_name: str | None
        if isinstance(run_context, FlowRunContext) and run_context.flow_run is not None:
            flow_run_name = str(run_context.flow_run.name)
        else:
            flow_run_name = None
    except MissingContextError:
        flow_run_name = None

    await create_inference_on_batch_summary_artifact(
        all_successes,
        all_failures,
        all_unknown_failures,
        flow_run_name,
    )

    batch_inference_result = BatchInferenceResult(
        batch_document_stems=batch,
        successful_document_stems=[i.document_stem for i in store_labels_successes],
        classifier_spec=classifier_spec,
    )

    if batch_inference_result.failed:
        message = (
            "Failed to run inference on "
            f"{batch_inference_result.failed_document_count}/"
            f"{batch_inference_result.all_document_count} documents."
        )
        raise Fault(
            msg=message,
            metadata={},
            data=batch_inference_result,
        )
    return batch_inference_result


# The default serialiser is cloudpickle, which can handle basic Pydantic types.
# Should the complexity of the returned objects become more complex
# then a custom serialiser should be considered.


@flow(result_storage=S3_BLOCK_RESULTS_CACHE)
async def inference_batch_of_documents_cpu(
    batch: list[DocumentStem],
    config_json: JsonDict,
    classifier_spec_json: JsonDict,
) -> BatchInferenceResult | Fault:
    return await _inference_batch_of_documents(
        batch,
        config_json,
        classifier_spec_json,
    )


@flow(result_storage=S3_BLOCK_RESULTS_CACHE)
async def inference_batch_of_documents_gpu(
    batch: list[DocumentStem],
    config_json: JsonDict,
    classifier_spec_json: JsonDict,
) -> BatchInferenceResult | Fault:
    return await _inference_batch_of_documents(
        batch,
        config_json,
        classifier_spec_json,
    )


def filter_document_batch(
    file_stems: Sequence[DocumentStem], spec: ClassifierSpec
) -> FilterResult:
    removed_file_stems = []
    accepted_file_stems = []
    for stem in file_stems:
        if should_skip_doc(stem, spec):
            removed_file_stems.append(stem)
        else:
            accepted_file_stems.append(stem)
    return FilterResult(removed=removed_file_stems, accepted=accepted_file_stems)


@flow(
    on_failure=[SlackNotify.message],
    on_crashed=[SlackNotify.message],
)
async def inference(
    classifier_specs: Sequence[ClassifierSpec] | None = None,
    document_ids: Sequence[DocumentImportId] | None = None,
    use_new_and_updated: bool = False,
    config: Config | None = None,
    batch_size: int = INFERENCE_BATCH_SIZE_DEFAULT,
    classifier_concurrency_limit: PositiveInt = CLASSIFIER_CONCURRENCY_LIMIT,
) -> InferenceResult | Fault:
    """
    Flow to run inference on documents within a bucket prefix.

    Default behaviour is to run on everything, pass document_ids to
    limit to specific files.

    Iterates: classifiers > documents > passages. Loading output into S3

    params:
    - document_ids: List of document ids to run inference on
    - classifier_spec: List of classifier names and aliases (alias tag
      for the version) to run inference with
    - config: A Config object, uses the default if not given. Usually
      there is no need to change this outside of local dev
    """
    logger = get_logger()
    if not config:
        config = await Config.create()
    logger.info(f"Running with config: {config}")

    current_bucket_file_stems = await list_bucket_file_stems(config=config)
    validated_file_stems = await determine_file_stems(
        config=config,
        use_new_and_updated=use_new_and_updated,
        requested_document_ids=document_ids,
        current_bucket_file_stems=current_bucket_file_stems,
    )

    if classifier_specs is None:
        classifier_specs = load_classifier_specs(config.aws_env)

    disallow_latest_alias(classifier_specs)

    logger.info(
        f"Running with {len(validated_file_stems)} documents and "
        f"{len(classifier_specs)} classifiers"
    )

    def parameters(
        classifier_spec: ClassifierSpec,
        document_batch: Sequence[DocumentStem],
    ) -> InferenceParams:
        return InferenceParams(
            batch=document_batch,
            config_json=JsonDict(config.to_json()),
            classifier_spec_json=JsonDict(classifier_spec.model_dump()),
        )

    # Prepare document batches based on classifier specs
    parameterised_batches: Sequence[ParameterisedFlow] = []
    removal_details: dict[ClassifierSpec, int] = {}

    for classifier_spec in classifier_specs:
        filter_result = filter_document_batch(validated_file_stems, classifier_spec)
        removal_details[classifier_spec] = len(filter_result.removed)

        for document_batch in iterate_batch(filter_result.accepted, batch_size):
            params = parameters(classifier_spec, document_batch)
            if (
                classifier_spec.compute_environment
                and classifier_spec.compute_environment.gpu
            ):
                fn = inference_batch_of_documents_gpu
            else:
                fn = inference_batch_of_documents_cpu

            parameterised_batches.append(ParameterisedFlow(fn=fn, params=params))

    await create_dont_run_on_docs_summary_artifact(
        config=config, removal_details=removal_details
    )

    all_raw_successes = []
    all_raw_failures = []

    with Profiler(
        printer=print,
        name="running classifier inference with map_as_sub_flow",
    ):
        raw_successes, raw_failures = await map_as_sub_flow(
            aws_env=config.aws_env,
            counter=classifier_concurrency_limit,
            parameterised_batches=parameterised_batches,
            unwrap_result=True,
        )

        all_raw_successes.extend(raw_successes)
        all_raw_failures.extend(raw_failures)

    # The type of response when running as a sub deployment is:
    #   <class 'inference.BatchInferenceResult'>
    all_successes = [
        BatchInferenceResult(**result.model_dump()) for result in all_raw_successes
    ]

    successful_classifier_specs = []
    failed_classifier_specs = []
    success_specs = [str(result.classifier_spec) for result in all_successes]
    for spec in classifier_specs:
        if str(spec) in success_specs:
            successful_classifier_specs.append(spec)
        else:
            failed_classifier_specs.append(spec)

    inference_result = InferenceResult(
        classifier_specs=list(classifier_specs),
        batch_inference_results=all_successes,
        successful_classifier_specs=successful_classifier_specs,
        failed_classifier_specs=failed_classifier_specs,
        parameterised_batches=parameterised_batches,
    )

    await create_inference_summary_artifact(
        config=config,
        inference_result=inference_result,
        removal_details=removal_details,
    )

    if inference_result.failed:
        raise Fault(
            msg="Some inference batches had failures.",
            metadata={},
            data=inference_result,
        )
    return inference_result


async def create_dont_run_on_docs_summary_artifact(
    config: Config,
    removal_details: dict[ClassifierSpec, int],
) -> None:
    """Create an artifact with a summary about the inference run."""

    description = "# Document removals per classifier"
    table = [
        {
            "Wikibase ID": spec.wikibase_id,
            "Classifier ID": spec.classifier_id,
            "Dont Run Ons": [s.value for s in (spec.dont_run_on or [])],
            "Removals": count,
        }
        for spec, count in removal_details.items()
    ]
    await acreate_table_artifact(
        key=f"removal-details-{config.aws_env.value}",
        table=table,
        description=description,
    )


async def create_inference_summary_artifact(
    config: Config,
    inference_result: InferenceResult,
    removal_details: dict[ClassifierSpec, int],
) -> None:
    """Create an artifact with a summary about the inference run."""

    # Format the overview information as a string for the description
    overview_description = f"""# Classifier Inference Summary

## Overview
- **Environment**: {config.aws_env.value}
- **Total documents requested**: {len(inference_result.inference_all_document_stems)}
- **Total classifiers**: {len(inference_result.classifier_specs)}
- **Successful classifiers**: {len(inference_result.successful_classifier_specs)}
- **Failed classifiers**: {len(inference_result.failed_classifier_specs)}
- **Classifiers with removals**: {len(removal_details)}
"""
    # Create classifier details table
    classifier_details = [
        {
            "Classifier": str(spec),
            "Filtered Out": removal_details[spec],
            "Status": "✓",
        }
        for spec in inference_result.successful_classifier_specs
    ] + [
        {
            "Classifier": spec.wikibase_id,
            "Filtered Out": removal_details[spec],
            "Status": "✗",
        }
        for spec in inference_result.failed_classifier_specs
    ]

    await acreate_table_artifact(
        key=f"classifier-inference-{config.aws_env.value}",
        table=classifier_details,
        description=overview_description,
    )
