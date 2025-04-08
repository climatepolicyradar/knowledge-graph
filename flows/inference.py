import asyncio
import json
import os
from collections import defaultdict
from collections.abc import Generator
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Final, Optional, TypeAlias
from uuid import UUID

import boto3
import prefect.artifacts as artifacts
from cpr_sdk.parser_models import BaseParserOutput, BlockType
from cpr_sdk.ssm import get_aws_ssm_param
from prefect import flow
from prefect.client.schemas.objects import FlowRun, StateType
from prefect.concurrency.asyncio import concurrency
from prefect.deployments import run_deployment
from prefect.flow_runs import wait_for_flow_run
from prefect.logging import get_run_logger
from prefect.task_runners import ConcurrentTaskRunner
from pydantic import SecretStr

import wandb
from flows.utils import SlackNotify, get_file_stems_for_document_id
from scripts.cloud import (
    AwsEnv,
    ClassifierSpec,
    function_to_flow_name,
    generate_deployment_name,
    get_prefect_job_variable,
)
from scripts.update_classifier_spec import parse_spec_file
from src.classifier import Classifier
from src.labelled_passage import LabelledPassage
from src.span import Span
from wandb.sdk.wandb_run import Run

DOCUMENT_SOURCE_PREFIX_DEFAULT: str = "embeddings_input"
# NOTE: Comparable list being maintained at https://github.com/climatepolicyradar/navigator-search-indexer/blob/91e341b8a20affc38cd5ce90c7d5651f21a1fd7a/src/config.py#L13.
BLOCKED_BLOCK_TYPES: Final[set[BlockType]] = {
    BlockType.PAGE_NUMBER,
    BlockType.TABLE,
    BlockType.FIGURE,
}
DOCUMENT_TARGET_PREFIX_DEFAULT: str = "labelled_passages"

# Example: CCLW.executive.1813.2418
DocumentImportId: TypeAlias = str
DocumentRunIdentifier: TypeAlias = tuple[str, str, str]
DocumentStem: TypeAlias = str


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


def get_latest_ingest_documents(config: Config) -> list[str]:
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
    requested_document_ids: Optional[list[DocumentImportId]],
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
        return current_bucket_file_stems

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


def download_classifier_from_wandb_to_local(
    run: Run, config: Config, classifier_name: str, alias: str = "latest"
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


def text_block_inference(
    classifier: Classifier, block_id: str, text: str
) -> LabelledPassage:
    """Run predict on a single text block."""
    spans: list[Span] = classifier.predict(text)

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

    labelled_passage = LabelledPassage(
        id=block_id,
        text=text,
        spans=spans,
        metadata=metadata,
    )

    return labelled_passage


def _name_document_run_identifiers_set(
    documents: set[DocumentRunIdentifier],
    status: str,
) -> list[dict[str, str]]:
    """Convert a set of document run identifiers for table rows."""
    keys = ("document_id", "classifier_name", "classifier_alias", "status")
    return [dict(zip(keys, doc + (status,))) for doc in documents]


async def report_documents_runs(
    queued: set[DocumentRunIdentifier],
    completed: set[DocumentRunIdentifier],
    aws_env: AwsEnv,
) -> None:
    try:
        # Create rows for both queued and completed documents with status
        queued_rows = _name_document_run_identifiers_set(queued, "queued")
        completed_rows = _name_document_run_identifiers_set(completed, "completed")

        # Combine both sets of rows
        all_rows = queued_rows + completed_rows

        await artifacts.create_table_artifact(
            table=all_rows,
            description=f"# Document Processing Status ({aws_env.value})",
            key=f"classifier-inference-document-processing-status-{aws_env.value}",
        )
    except Exception:
        # Do nothing, not even log. It'll be too noisy.
        pass


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

    if document.languages != ["en"]:
        raise ValueError(
            f"Cannot run inference on {file_stem} as it has non-english language: "
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


def iterate_batch(data: list[str], batch_size: int = 400) -> Generator:
    """Generate batches from a list with a specified size."""
    for i in range(0, len(data), batch_size):
        yield data[i : i + batch_size]


@flow
async def run_classifier_inference_on_batch_of_documents(
    batch: list[str],
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
        run_classifier_inference_on_document(
            config=config,
            file_stem=file_stem,
            classifier_name=classifier_name,
            classifier_alias=classifier_alias,
            classifier=classifier,
        )
        for file_stem in batch
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    failures: list[Exception] = []

    for result in results:
        if isinstance(result, Exception):
            logger.exception(f"Failed to process document: {result}")
            failures.append(result)
        elif result is None:
            continue
        else:
            raise ValueError(
                f"Unexpected type of result. Type: `{type(result)}`, value: `{result}`"
            )

    if len(failures) > 0:
        raise ValueError(f"Failed to process {len(failures)}/{len(results)} documents")

    return None


@flow(
    log_prints=True,
    task_runner=ConcurrentTaskRunner(),
    on_failure=[SlackNotify.message],
    on_crashed=[SlackNotify.message],
)
async def classifier_inference(
    classifier_specs: Optional[list[ClassifierSpec]] = None,
    document_ids: Optional[list[str]] = None,
    use_new_and_updated: bool = False,
    config: Optional[Config] = None,
    batch_size: int = 1000,
):
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

    if classifier_specs is None:
        classifier_specs = parse_spec_file(config.aws_env)

    print(
        f"Running with {len(validated_file_stems)} documents and "
        f"{len(classifier_specs)} classifiers"
    )

    flow_name = function_to_flow_name(run_classifier_inference_on_batch_of_documents)
    deployment_name = generate_deployment_name(
        flow_name=flow_name, aws_env=config.aws_env
    )

    failures: dict[ClassifierSpec, list[str | UUID]] = defaultdict(list)

    for classifier_spec in classifier_specs:
        batches = iterate_batch(validated_file_stems, batch_size)

        ttl_per_task_s = 20 * 60

        async def run_and_wait_for_deployment(
            flow_name, deployment_name, batch, config, classifier_spec
        ):
            flow_run: FlowRun = await run_deployment(
                name=f"{flow_name}/{deployment_name}",
                parameters={
                    "batch": batch,
                    "config_json": config.to_json(),
                    "classifier_name": classifier_spec.name,
                    "classifier_alias": classifier_spec.alias,
                },
                # Return the metadata immediately
                timeout=0,
                as_subflow=True,
            )

            return await wait_for_flow_run(
                flow_run_id=flow_run.id,
                timeout=ttl_per_task_s,  # Seconds
            )

        tasks = [
            asyncio.wait_for(
                run_and_wait_for_deployment(
                    flow_name,
                    deployment_name,
                    batch,
                    config,
                    classifier_spec,
                ),
                # We could just rely on the timeout via Prefect, but,
                # this may also not be running on Prefect.
                #
                # Realistically, there is some spin-up time on
                # Prefect, for the task to actually be executing our
                # code, so be aware of that.
                timeout=ttl_per_task_s,  # Seconds
            )
            for batch in batches
        ]

        results: list[FlowRun] = await asyncio.gather(*tasks)

        for result in results:
            # We'd never expect the flow run name to be missing, but
            # Prefect does account for that in their class.
            name: str = result.name
            if result.state is None:
                failures[classifier_spec].append(name)
            else:
                if result.state.type != StateType.COMPLETED:
                    failures[classifier_spec].append(name)

    if failures:
        raise ValueError(
            f"some classifier specs. had failures: {', '.join(map(str, failures.keys()))}"
        )

    print("Finished running classifier inference.")
