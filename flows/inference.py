import asyncio
import json
import os
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from datetime import timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Final, Optional, TypeAlias

import boto3
import wandb
from cpr_sdk.parser_models import BaseParserOutput, BlockType
from cpr_sdk.ssm import get_aws_ssm_param
from prefect import flow
from prefect.artifacts import create_table_artifact
from prefect.client.schemas.objects import FlowRun
from prefect.concurrency.asyncio import concurrency
from prefect.context import get_run_context
from prefect.logging import get_run_logger
from prefect.utilities.names import generate_slug
from pydantic import PositiveInt, SecretStr
from wandb.sdk.wandb_run import Run

from flows.utils import (
    DocumentImportId,
    DocumentStem,
    Profiler,
    SlackNotify,
    filter_non_english_language_file_stems,
    get_file_stems_for_document_id,
    iterate_batch,
    map_as_sub_flow,
    return_with_id,
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


def load_document(config: Config, file_stem: DocumentStem) -> BaseParserOutput:
    """Download and opens a parser output based on a document ID."""
    file_key = os.path.join(
        config.document_source_prefix,
        f"{file_stem}.json",
    )
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


def serialise_labels(labels: list[LabelledPassage]) -> BytesIO:
    data = [label.model_dump_json() for label in labels]
    return BytesIO(json.dumps(data).encode("utf-8"))


def store_labels(
    config: Config,
    labels: list[LabelledPassage],
    file_stem: DocumentStem,
    classifier_name: str,
    classifier_alias: str,
) -> None:
    """Store the labels in the cache bucket."""
    key = os.path.join(
        config.document_target_prefix,
        classifier_name,
        classifier_alias,
        f"{file_stem}.json",
    )
    print(f"Storing labels for document {file_stem} at {key}")

    body = serialise_labels(labels)

    s3 = boto3.client("s3", region_name=config.bucket_region)
    s3.put_object(
        Bucket=config.cache_bucket,
        Key=key,
        Body=body,
        ContentType="application/json",
    )


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
) -> None:
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
        store_labels(
            config=config,
            labels=[],
            file_stem=file_stem,
            classifier_name=classifier_name,
            classifier_alias=classifier_alias,
        )

        return None

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

    store_labels(
        config=config,
        labels=doc_labels,
        file_stem=file_stem,
        classifier_name=classifier_name,
        classifier_alias=classifier_alias,
    )

    return None


async def create_inference_on_batch_summary_artifact(
    successes: list[DocumentStem],
    failures: list[tuple[DocumentStem, Exception]],
    unknown_failures: list[BaseException],
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
                "Document stem": document_stem,
                "Status": "✓",
                "Exception": "N/A",
            }
            for document_stem in successes
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


@flow(log_prints=True)
async def inference_batch_of_documents(
    batch: list[DocumentStem],
    config_json: dict,
    classifier_name: str,
    classifier_alias: str,
) -> None:
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
        return_with_id(
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
        tuple[DocumentStem, Exception | None] | BaseException
    ] = await asyncio.gather(*tasks, return_exceptions=True)

    successes: list[DocumentStem] = []
    failures: list[tuple[DocumentStem, Exception]] = []
    # We really don't expect these, since there's a try/catch handler
    # in `return_with_id`. It is technically possible though, for
    # there to be what I'm calling here an _unknown_ failure.
    unknown_failures: list[BaseException] = []

    for result in results:
        if isinstance(result, BaseException):
            unknown_failures.append(result)
        else:
            document_stem, value = result
            if isinstance(value, Exception):
                logger.exception(f"Failed to process document {document_stem}: {value}")
                failures.append((document_stem, value))
            else:
                successes.append(document_stem)

    # https://docs.prefect.io/v3/concepts/runtime-context#access-the-run-context-directly
    run_context = get_run_context()
    flow_run_name: str | None
    if run_context:
        flow_run_name = str(run_context.flow_run.name)
    else:
        flow_run_name = None

    await create_inference_on_batch_summary_artifact(
        successes,
        failures,
        unknown_failures,
        flow_run_name,
    )

    if len(failures) > 0:
        raise ValueError(
            f"Failed to process {len(failures) + len(unknown_failures)}/{len(results)} documents"
        )

    return None


@Profiler(
    printer=print,
    name="processing results",
)
def group_inference_results_into_states(
    successes_in: Sequence[FlowRun],
    failures_in: Sequence[BaseException | FlowRun],
) -> tuple[
    list[FlowRun | BaseException],
    dict[ClassifierSpec, FlowRun],
]:
    """Group results of sub-runs into the different states of success and failure."""
    successes: dict[ClassifierSpec, FlowRun] = {}

    for success in successes_in:
        classifier_spec = ClassifierSpec(
            name=success.parameters["classifier_name"],
            alias=success.parameters["classifier_alias"],
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
) -> Sequence[DocumentStem]:
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
                unwrap_result=False,
            )

            all_raw_successes.extend(raw_successes)
            all_raw_failures.extend(raw_failures)

    failures, successes = group_inference_results_into_states(
        all_raw_successes, all_raw_failures
    )
    failures_classifier_specs = set(classifier_specs) - set(successes.keys())

    await create_inference_summary_artifact(
        config=config,
        filtered_file_stems=filtered_file_stems,
        classifier_specs=classifier_specs,
        successes=successes,
        failures_classifier_specs=failures_classifier_specs,
    )

    if failures:
        raise ValueError(
            f"some classifier specs. had failures: {','.join(map(str, failures_classifier_specs))}"
        )

    return filtered_file_stems


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
