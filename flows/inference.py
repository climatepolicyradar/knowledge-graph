import asyncio
import json
import os
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from datetime import timedelta
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Any, Final, Optional, TypeAlias, TypeVar

import boto3
import wandb
from cpr_sdk.parser_models import BaseParserOutput, BlockType
from cpr_sdk.ssm import get_aws_ssm_param
from more_itertools import flatten
from prefect import flow
from prefect.artifacts import create_table_artifact
from prefect.assets import materialize
from prefect.client.schemas.objects import FlowRun
from prefect.concurrency.asyncio import concurrency
from prefect.context import get_run_context
from prefect.logging import get_run_logger
from prefect.states import Completed, Failed, State
from prefect.utilities.names import generate_slug
from pydantic import BaseModel, ConfigDict, PositiveInt, SecretStr
from types_aiobotocore_s3.type_defs import PutObjectOutputTypeDef
from wandb.sdk.wandb_run import Run

from flows.utils import (
    DocumentImportId,
    DocumentStem,
    Profiler,
    S3Uri,
    SlackNotify,
    filter_non_english_language_file_stems,
    get_file_stems_for_document_id,
    iterate_batch,
    map_as_sub_flow,
    return_with,
    wait_for_semaphore,
)
from scripts.cloud import (
    AwsEnv,
    ClassifierSpec,
    disallow_latest_alias,
    get_prefect_job_variable,
)
from scripts.update_classifier_spec import parse_spec_file
from src.classifier import Classifier
from src.labelled_passage import LabelledPassage
from src.span import Span

# The "parent" AKA the higher level flows that do multiple things
PARENT_TIMEOUT_S: int = int(timedelta(hours=12).total_seconds())
# A singular task doing one thing
TASK_TIMEOUT_S: int = int(timedelta(minutes=60).total_seconds())

DOCUMENT_SOURCE_PREFIX_DEFAULT: str = "embeddings_input"
# NOTE: Comparable list being maintained at https://github.com/climatepolicyradar/navigator-search-indexer/blob/91e341b8a20affc38cd5ce90c7d5651f21a1fd7a/src/config.py#L13.
BLOCKED_BLOCK_TYPES: Final[set[BlockType]] = {
    BlockType.PAGE_NUMBER,
    BlockType.TABLE,
    BlockType.FIGURE,
}
DOCUMENT_TARGET_PREFIX_DEFAULT: str = "labelled_passages"

CLASSIFIER_CONCURRENCY_LIMIT: Final[PositiveInt] = 20
INFERENCE_BATCH_SIZE_DEFAULT: Final[PositiveInt] = 1000
AWS_ENV: str = os.environ["AWS_ENV"]
S3_BLOCK_RESULTS_CACHE: str = f"s3-bucket/cpr-{AWS_ENV}-prefect-results-cache"

DocumentRunIdentifier: TypeAlias = tuple[str, str, str]


@dataclass()
class Config:
    """Configuration used across flow runs."""

    cache_bucket: Optional[str] = None
    document_source_prefix: str = DOCUMENT_SOURCE_PREFIX_DEFAULT
    document_target_prefix: str = DOCUMENT_TARGET_PREFIX_DEFAULT
    pipeline_state_prefix: str = "input"
    bucket_region: str = "eu-west-1"
    local_classifier_dir: Path = Path("data") / "processed" / "classifiers"
    wandb_model_org: str = "climatepolicyradar_UZODYJSN66HCQ"
    wandb_model_registry: str = "wandb-registry-model"
    wandb_entity: str = "climatepolicyradar"
    wandb_api_key: Optional[SecretStr] = None
    aws_env: AwsEnv = AwsEnv(os.environ["AWS_ENV"])

    @classmethod
    async def create(cls) -> "Config":
        """Create a new Config instance with initialized values."""
        config = cls()

        if not config.cache_bucket:
            config.cache_bucket = await get_prefect_job_variable(
                "pipeline_cache_bucket_name"
            )
        if not config.wandb_api_key:
            config.wandb_api_key = SecretStr(get_aws_ssm_param("WANDB_API_KEY"))

        return config

    def to_json(self) -> dict:
        """Convert the config to a JSON serializable dictionary."""
        return {
            "cache_bucket": self.cache_bucket if self.cache_bucket else None,
            "document_source_prefix": self.document_source_prefix,
            "document_target_prefix": self.document_target_prefix,
            "pipeline_state_prefix": self.pipeline_state_prefix,
            "bucket_region": self.bucket_region,
            "local_classifier_dir": self.local_classifier_dir,
            "wandb_model_org": self.wandb_model_org,
            "wandb_model_registry": self.wandb_model_registry,
            "wandb_entity": self.wandb_entity,
            "wandb_api_key": (
                self.wandb_api_key.get_secret_value() if self.wandb_api_key else None
            ),
            "aws_env": self.aws_env,
        }


class BatchInferenceException(Exception):
    """
    Exception raised when batch inference fails.

    The data attribute of the prefect State when type is FAILED must be an object that
    an exception can be raised from. Thus, we declare a custom exception that can also
    be used to transfer the result of the batch inference run.
    """

    def __init__(self, message: str, data: dict[str, Any]):
        super().__init__(message)

        self.message = message
        self.data = data


class BatchInferenceResult(BaseModel):
    """Result from running inference on a batch of documents."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    successful_document_stems: list[DocumentStem] = []
    failed_document_stems: list[tuple[DocumentStem, Exception]] = []
    unknown_failures: list[BaseException] = []
    classifier_name: str
    classifier_alias: str

    @property
    def failed(self) -> bool:
        """Whether the batch failed, True if failed."""

        return self.failed_document_stems != [] or self.unknown_failures != []


class InferenceException(Exception):
    """
    Exception raised when inference fails.

    The data attribute of the prefect State when type is FAILED must be an object that
    an exception can be raised from. Thus, we declare a custom exception that can also
    be used to transfer the results of the all the batch inference runs.
    """

    def __init__(self, message: str, data: dict[str, Any]):
        super().__init__(message)

        self.message = message
        self.data = data


class InferenceResult(BaseModel):
    """Result from running inference on all batches of documents."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    batch_inference_results: list[BatchInferenceResult] = []
    unexpected_failures: list[BaseException | FlowRun] = []
    successful_classifier_specs: list[ClassifierSpec] = []
    failed_classifier_specs: list[ClassifierSpec] = []

    @property
    def failed(self) -> bool:
        """Whether the inference failed, True if failed."""

        return (
            any([result.failed for result in self.batch_inference_results])
            or self.unexpected_failures != []
        )

    @cached_property
    def successful_document_stems(self) -> set[DocumentStem]:
        """
        The set of document stems that were successfully processed.

        A document stem is considered successful if it was succesful across all classifiers. For example,
        if a document successfully had inference run in one batch for classifier A, but failed for classifier B,
        then the document stem is considered unsuccessful.

        This is as the document would fail aggregation if there was a missing inference result for a classifier.
        """

        return set(
            document_stem
            for batch_inference_result in self.batch_inference_results
            for document_stem in batch_inference_result.successful_document_stems
            if document_stem not in self.failed_document_stems
        )

    @cached_property
    def failed_document_stems(self) -> set[DocumentStem]:
        """The set of document stems that failed to be processed."""

        return set(
            document_stem
            for batch_inference_result in self.batch_inference_results
            for document_stem, _ in batch_inference_result.failed_document_stems
        )


def get_bucket_paginator(config: Config, prefix: str):
    """Returns an s3 paginator for the pipeline cache bucket"""
    s3 = boto3.client("s3", region_name=config.bucket_region)
    paginator = s3.get_paginator("list_objects_v2")
    return paginator.paginate(
        Bucket=config.cache_bucket,
        Prefix=prefix,
    )


def list_bucket_file_stems(config: Config) -> list[DocumentStem]:
    """
    Scan configured bucket and return all file stems.

    Where a stem refers to a file name without the extension. Often, this is the same as
    the document id, but not always as we have translated documents.
    """
    page_iterator = get_bucket_paginator(config, config.document_source_prefix)
    file_stems = []

    for p in page_iterator:
        if "Contents" in p:
            for o in p["Contents"]:
                file_stem = Path(o["Key"]).stem
                file_stems.append(file_stem)

    return file_stems


def get_latest_ingest_documents(config: Config) -> Sequence[DocumentImportId]:
    """
    Get IDs of changed documents from the latest ingest run

    Retrieves the `new_and_updated_docs.json` file from the latest ingest.
    Extracts the ids from the file, and returns them as a single list.
    """
    page_iterator = get_bucket_paginator(config, config.pipeline_state_prefix)
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
            f"`{config.cache_bucket}/{config.pipeline_state_prefix}`"
        )

    # Sort by Key and get the last one
    latest = sorted(matching_files, key=lambda x: x["Key"])[-1]

    data = download_s3_file(config, latest["Key"])
    content = json.loads(data)
    updated = list(content["updated_documents"].keys())
    new = [d["import_id"] for d in content["new_documents"]]

    print(f"Retrieved {len(new)} new, and {len(updated)} updated from {latest['Key']}")
    return new + updated


def determine_file_stems(
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
        requested_document_ids = get_latest_ingest_documents(config)
    elif requested_document_ids is None:
        current_bucket_file_stems__filtered = filter_non_english_language_file_stems(
            file_stems=current_bucket_file_stems
        )
        return current_bucket_file_stems__filtered

    assert config.cache_bucket

    requested_document_stems = []
    for doc_id in requested_document_ids:
        document_key = os.path.join(config.document_source_prefix, f"{doc_id}.json")
        requested_document_stems += get_file_stems_for_document_id(
            doc_id, config.cache_bucket, document_key
        )

    missing_from_bucket = list(
        set(requested_document_stems) - set(current_bucket_file_stems)
    )
    if len(missing_from_bucket) > 0:
        raise ValueError(
            f"Requested document_ids not found in bucket: {missing_from_bucket}"
        )

    return requested_document_stems


def remove_sabin_file_stems(
    file_stems: Sequence[DocumentStem],
) -> Sequence[DocumentStem]:
    """
    Remove Sabin document file stems from the list of file stems.

    File stems of the Sabin source follow the below naming convention:
    - "Sabin.document.16944.17490"
    """
    return [
        stem for stem in file_stems if not stem.startswith(("Sabin", "sabin", "SABIN"))
    ]


def download_classifier_from_wandb_to_local(
    run: Run, config: Config, classifier_name: str, alias: str
) -> str:
    """
    Download a classifier from W&B to local.

    Models referenced by weights and biases are stored in s3. This
    means that to download the model via the W&B API, we need access
    to both the s3 bucket via iam in your environment and WanDB via
    the api key.
    """
    artifact = os.path.join(config.wandb_model_registry, f"{classifier_name}:{alias}")
    print(f"Downloading artifact from W&B: {artifact}")
    artifact = run.use_artifact(artifact, type="model")
    classifier = artifact.download()
    return classifier


async def load_classifier(
    run: Run, config: Config, classifier_name: str, alias: str
) -> Classifier:
    """
    Load a classifier into memory.

    If the classifier is available locally, this will be used. Otherwise the
    classifier will be downloaded from W&B (Once implemented)
    """
    async with concurrency("load_classifier", occupy=5):
        local_classifier_path: Path = config.local_classifier_dir / classifier_name

        if not local_classifier_path.exists():
            model_cache_dir = download_classifier_from_wandb_to_local(
                run, config, classifier_name, alias
            )
            local_classifier_path = Path(model_cache_dir) / "model.pickle"

        classifier = Classifier.load(local_classifier_path)

        return classifier


def download_s3_file(config: Config, key: str):
    """Retrieve an s3 file from the pipeline cache"""

    s3 = boto3.client("s3", region_name=config.bucket_region)
    response = s3.get_object(Bucket=config.cache_bucket, Key=key)
    content = response["Body"].read().decode("utf-8")
    return content


def generate_document_source_key(config: Config, document_stem: DocumentStem) -> S3Uri:
    return S3Uri(
        bucket=config.cache_bucket,  # pyright: ignore[reportArgumentType]
        key=os.path.join(
            config.document_source_prefix,
            f"{document_stem}.json",
        ),
    )


def load_document(config: Config, file_stem: DocumentStem) -> BaseParserOutput:
    """Download and opens a parser output based on a document ID."""
    file_key = generate_document_source_key(
        config=config,
        document_stem=file_stem,
    ).key
    content = download_s3_file(config=config, key=file_key)
    document = BaseParserOutput.model_validate_json(content)
    return document


def _stringify(text: list[str]) -> str:
    return " ".join([line.strip() for line in text])


def document_passages(
    document: BaseParserOutput,
) -> Generator[tuple[str, str], None, None]:
    """Yield the text block irrespective of content type."""
    match document.document_content_type:
        case "application/pdf":
            text_blocks = document.pdf_data.text_blocks  # type: ignore
        case "text/html":
            text_blocks = document.html_data.text_blocks  # type: ignore
        case _:
            text_blocks = []
            print(
                "Unsupported document content type: "
                f"{document.document_content_type}, for "
                f"document: {document.document_id}"
            )
    for text_block in text_blocks:
        if text_block.type not in BLOCKED_BLOCK_TYPES:
            yield _stringify(text_block.text), text_block.text_block_id


T = TypeVar("T", bound=BaseModel)


def serialise_pydantic_list_as_jsonl(models: Sequence[T]) -> BytesIO:
    """
    Serialize a list of Pydantic models as JSONL (JSON Lines) format.

    Each model is serialized on a separate line using model_dump_json().
    """
    jsonl_content = "\n".join(model.model_dump_json() for model in models)
    return BytesIO(jsonl_content.encode("utf-8"))


def deserialise_pydantic_list_from_jsonl(
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


def deserialise_pydantic_list_with_fallback(
    content: str, model_class: type[T]
) -> list[T]:
    """
    Deserialize content to a list of Pydantic models with fallback support.

    First tries JSONL format, then falls back to original format (JSON array of JSON strings).
    """
    from pydantic import ValidationError

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
    classifier_name: str
    classifier_alias: str


def generate_s3_uri_output(
    config: Config, inference: SingleDocumentInferenceResult
) -> S3Uri:
    return S3Uri(
        bucket=config.cache_bucket,  # pyright: ignore[reportArgumentType]
        key=os.path.join(
            config.document_target_prefix,
            inference.classifier_name,
            inference.classifier_alias,
            f"{inference.document_stem}.json",
        ),
    )


def generate_s3_uri_input(
    config: Config, inference: SingleDocumentInferenceResult
) -> Path:
    return config.local_classifier_dir / inference.classifier_name


@materialize(
    None,  # Asset key is not known yet
    retries=1,
    persist_result=False,
)
async def store_labels(
    config: Config,
    inferences: Sequence[SingleDocumentInferenceResult],
) -> tuple[
    list[SingleDocumentInferenceResult],
    list[tuple[DocumentStem, Exception]],
    list[BaseException],
]:
    """Store the labels in the cache bucket."""
    logger = get_run_logger()

    session = boto3.Session(region_name=config.bucket_region)

    s3 = session.client("s3")
    # Don't get rate-limited by AWS
    semaphore = asyncio.Semaphore(10)

    async def fn(inference) -> PutObjectOutputTypeDef:
        s3_uri = generate_s3_uri_output(config, inference)
        logger.info(
            f"Storing labels for document {inference.document_stem} at {s3_uri}"
        )

        body = serialise_pydantic_list_as_jsonl(inference.labelled_passages)

        response = s3.put_object(
            Bucket=s3_uri.bucket,
            Key=s3_uri.key,
            Body=body,
            ContentType="application/json",
        )

        return response

    tasks = [
        wait_for_semaphore(
            semaphore,
            return_with(
                inference,
                fn(inference),
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
                logger.exception(
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


async def run_classifier_inference_on_document(
    config: Config,
    file_stem: DocumentStem,
    classifier_name: str,
    classifier_alias: str,
    classifier: Classifier,
) -> SingleDocumentInferenceResult:
    """Run the classifier inference flow on a document."""
    print(f"Loading document with file stem {file_stem}")
    document = load_document(config, file_stem)
    print(f"Loaded document with file stem {file_stem}")

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
            classifier_name=classifier_name,
            classifier_alias=classifier_alias,
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
        classifier_name=classifier_name,
        classifier_alias=classifier_alias,
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

    await create_table_artifact(
        key=f"batch-inference-{flow_run_name}",
        table=document_details,
        description=overview_description,
    )


def generate_assets(
    config: Config,
    inferences: Sequence[SingleDocumentInferenceResult],
) -> Sequence[str]:
    return [str(generate_s3_uri_output(config, inference)) for inference in inferences]


def generate_asset_deps(
    config: Config,
    inferences: Sequence[SingleDocumentInferenceResult],
) -> Sequence[str]:
    return list(
        flatten(
            [
                (
                    f"wandb://{config.wandb_entity}/{config.wandb_model_registry}/{inference.classifier_name}:{inference.classifier_alias}",
                    str(
                        generate_document_source_key(
                            config=config,
                            document_stem=inference.document_stem,
                        )
                    ),
                )
                for inference in inferences
            ]
        )
    )


@flow(log_prints=True, result_storage=S3_BLOCK_RESULTS_CACHE)
async def inference_batch_of_documents(
    batch: list[DocumentStem],
    config_json: dict,
    classifier_name: str,
    classifier_alias: str,
) -> State:
    """
    Run classifier inference on a batch of documents.

    This reflects the unit of work that should be run in one of many paralellised
    docker containers.
    """
    logger = get_run_logger()

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

    logger.info(
        f"Loading classifier with name: {classifier_name}, and alias: {classifier_alias}"  # noqa: E501
    )
    classifier = await load_classifier(
        run,
        config,
        classifier_name,
        classifier_alias,
    )
    logger.info(
        f"Loaded classifier with name: {classifier_name}, and alias: {classifier_alias}"  # noqa: E501
    )

    tasks = [
        return_with(
            file_stem,
            run_classifier_inference_on_document(
                config=config,
                file_stem=file_stem,
                classifier_name=classifier_name,
                classifier_alias=classifier_alias,
                classifier=classifier,
            ),
        )
        for file_stem in batch
    ]

    results: list[
        tuple[DocumentStem, Exception | SingleDocumentInferenceResult] | BaseException
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
                logger.exception(f"Failed to process document {document_stem}: {value}")
                inferences_failures.append((document_stem, value))
            else:
                inferences_successes.append(value)

    (
        store_labels_successes,
        store_labels_failures,
        store_labels_unknown_failures,
    ) = await store_labels.with_options(  # pyright: ignore[reportFunctionMemberAccess]
        assets=generate_assets(config, inferences_successes),
        asset_deps=generate_asset_deps(config, inferences_successes),
    )(config=config, inferences=inferences_successes)

    # This doesn't need to be combined since successes are funnelled
    # through all steps.
    #
    # Failures are possibly reduced at each step.
    all_successes = store_labels_successes
    # Combine the multiple places that have reports
    all_failures = inferences_failures + store_labels_failures
    all_unknown_failures = inferences_unknown_failures + store_labels_unknown_failures

    # https://docs.prefect.io/v3/concepts/runtime-context#access-the-run-context-directly
    run_context = get_run_context()
    flow_run_name: str | None
    if run_context:
        flow_run_name = str(run_context.flow_run.name)
    else:
        flow_run_name = None

    await create_inference_on_batch_summary_artifact(
        all_successes,
        all_failures,
        all_unknown_failures,
        flow_run_name,
    )

    batch_inference_result = BatchInferenceResult(
        successful_document_stems=[i.document_stem for i in inferences_successes],
        failed_document_stems=inferences_failures,
        unknown_failures=inferences_unknown_failures,
        classifier_name=classifier_name,
        classifier_alias=classifier_alias,
    )

    if batch_inference_result.failed:
        message = (
            f"Failed to run inference on {len(inferences_failures) + len(inferences_unknown_failures)}/"
            f"{len(results)} documents."
        )
        return Failed(
            message=message,
            data=BatchInferenceException(
                message=message,
                data=batch_inference_result.model_dump(),
            ),
        )
    return Completed(
        message=f"Successfully ran inference on all ({len(results)}) documents in batch.",
        data=batch_inference_result.model_dump(),
    )


@Profiler(
    printer=print,
    name="processing results",
)
def group_inference_results_into_states(
    successes_in: Sequence[BatchInferenceResult],
    failures_in: Sequence[BaseException | FlowRun],
) -> tuple[
    list[FlowRun | BaseException],
    dict[ClassifierSpec, BatchInferenceResult],
]:
    """Group results of sub-runs into the different states of success and failure."""
    successes: dict[ClassifierSpec, BatchInferenceResult] = {}

    for success in successes_in:
        classifier_spec = ClassifierSpec(
            name=success.classifier_name,
            alias=success.classifier_alias,
        )
        successes[classifier_spec] = success

    return list(failures_in), successes


@flow(
    log_prints=True,
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
) -> State:
    """
    Flow to run inference on documents within a bucket prefix.

    Default behaviour is to run on everything, pass document_ids to
    limit to specific files.

    Iterates: classifiers > documents > passages. Loading output into s3

    params:
    - document_ids: List of document ids to run inference on
    - classifier_spec: List of classifier names and aliases (alias tag
      for the version) to run inference with
    - config: A Config object, uses the default if not given. Usually
      there is no need to change this outside of local dev
    """
    if not config:
        config = await Config.create()

    print(f"Running with config: {config}")

    current_bucket_file_stems = list_bucket_file_stems(config=config)
    validated_file_stems = determine_file_stems(
        config=config,
        use_new_and_updated=use_new_and_updated,
        requested_document_ids=document_ids,
        current_bucket_file_stems=current_bucket_file_stems,
    )
    filtered_file_stems = remove_sabin_file_stems(validated_file_stems)

    if classifier_specs is None:
        classifier_specs = parse_spec_file(config.aws_env)

    disallow_latest_alias(classifier_specs)

    print(
        f"Running with {len(filtered_file_stems)} documents and "
        f"{len(classifier_specs)} classifiers"
    )

    all_raw_successes = []
    all_raw_failures = []

    for classifier_spec in classifier_specs:
        batches = iterate_batch(filtered_file_stems, batch_size)

        def parameters(
            batch: Sequence[DocumentStem],
        ) -> dict[str, Any]:
            return {
                "batch": batch,
                "config_json": config.to_json(),
                "classifier_name": classifier_spec.name,
                "classifier_alias": classifier_spec.alias,
            }

        with Profiler(
            printer=print,
            name="running classifier inference with map_as_sub_flow",
        ):
            raw_successes, raw_failures = await map_as_sub_flow(
                fn=inference_batch_of_documents,
                aws_env=config.aws_env,
                counter=classifier_concurrency_limit,
                batches=batches,
                parameters=parameters,
                unwrap_result=True,
            )

            all_raw_successes.extend(raw_successes)
            all_raw_failures.extend(raw_failures)

    all_successes = [BatchInferenceResult(**result) for result in all_raw_successes]
    _, successes = group_inference_results_into_states(all_successes, all_raw_failures)
    failures_classifier_specs = list(set(classifier_specs) - set(successes.keys()))

    inference_result = InferenceResult(
        batch_inference_results=all_successes,
        unexpected_failures=all_raw_failures,
        successful_classifier_specs=successes.keys(),
        failed_classifier_specs=failures_classifier_specs,
    )

    await create_inference_summary_artifact(
        config=config,
        filtered_file_stems=filtered_file_stems,
        classifier_specs=classifier_specs,
        successes=successes,
        failures_classifier_specs=set(failures_classifier_specs),
    )

    if inference_result.failed:
        message = "Some inference batches had failures!"
        return Failed(
            message=message,
            data=InferenceException(
                message=message,
                data=inference_result.model_dump(),
            ),
        )
    return Completed(
        message="Successfully ran inference on all batches!",
        data=inference_result.model_dump(),
    )


async def create_inference_summary_artifact(
    config: Config,
    filtered_file_stems: Sequence[DocumentStem],
    classifier_specs: Sequence[ClassifierSpec],
    successes: dict[ClassifierSpec, FlowRun],
    failures_classifier_specs: set[ClassifierSpec],
) -> None:
    """Create an artifact with a summary about the inference run."""

    # Prepare summary data for the artifact
    total_documents = len(filtered_file_stems)
    total_classifiers = len(classifier_specs)
    successful_classifiers = len(successes)
    failed_classifiers = len(failures_classifier_specs)

    # Format the overview information as a string for the description
    overview_description = f"""# Classifier Inference Summary

## Overview
- **Environment**: {config.aws_env.value}
- **Total documents processed**: {total_documents}
- **Total classifiers**: {total_classifiers}
- **Successful classifiers**: {successful_classifiers}
- **Failed classifiers**: {failed_classifiers}
"""

    # Create classifier details table
    classifier_details = [
        {"Classifier": spec.name, "Alias": spec.alias, "Status": "✓"}
        for spec in successes.keys()
    ] + [
        {"Classifier": spec.name, "Alias": spec.alias, "Status": "✗"}
        for spec in failures_classifier_specs
    ]

    await create_table_artifact(
        key=f"classifier-inference-{config.aws_env.value}",
        table=classifier_details,
        description=overview_description,
    )
