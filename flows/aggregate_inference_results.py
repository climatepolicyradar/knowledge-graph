import asyncio
import json
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Sequence, TypeAlias

import boto3
from botocore.exceptions import ClientError
from prefect import flow, task
from prefect.artifacts import create_table_artifact
from prefect.context import get_run_context
from prefect.exceptions import MissingContextError
from prefect.task_runners import ConcurrentTaskRunner

from flows.boundary import (
    DocumentImportId,
    TextBlockId,
    convert_labelled_passage_to_concepts,
    s3_copy_file,
    s3_object_write_text,
)
from flows.inference import DOCUMENT_TARGET_PREFIX_DEFAULT
from flows.utils import (
    S3Uri,
    SlackNotify,
    collect_unique_file_stems_under_prefix,
    iterate_batch,
    wait_for_semaphore,
)
from scripts.cloud import (
    AwsEnv,
    ClassifierSpec,
    get_prefect_job_variable,
)
from scripts.update_classifier_spec import parse_spec_file
from src.labelled_passage import LabelledPassage

# Constant, s3 prefix for the aggregated results
INFERENCE_RESULTS_PREFIX = "inference_results"

# A unique identifier for the run output made from the run context
RunOutputIdentifier: TypeAlias = str

# A string representation of a classifier spec (i.e. Q123:v4)
SpecStr: TypeAlias = str

# A serialised vespa concept, see cpr_sdk.models.search.Concept
SerialisedVespaConcept: TypeAlias = list[dict[str, str]]


class AggregationFailure(Exception):
    """A document failure."""

    def __init__(
        self, document_id: DocumentImportId, exception: Exception, context: str
    ):
        self.document_id = document_id
        self.exception = exception
        self.context = context


@dataclass()
class Config:
    """Configuration used across flow runs."""

    _cache_bucket: str | None = None
    document_source_prefix: str = DOCUMENT_TARGET_PREFIX_DEFAULT
    aggregate_inference_results_prefix: str = INFERENCE_RESULTS_PREFIX
    bucket_region: str = "eu-west-1"
    aws_env: AwsEnv = AwsEnv(os.environ["AWS_ENV"])

    @property
    def cache_bucket(self) -> str:
        """Get the cache bucket name. Raises ValueError if not set."""
        if self._cache_bucket is None:
            raise ValueError("cache_bucket has not been set")
        return self._cache_bucket

    @classmethod
    async def create(cls) -> "Config":
        """Create a new Config instance with initialized values."""
        config = cls()
        if not config._cache_bucket:
            config._cache_bucket = await get_prefect_job_variable(
                "pipeline_cache_bucket_name"
            )

        return config

    @property
    def cache_bucket_str(self) -> str:
        """Return the cache bucket, raising an error if not set."""
        if not self.cache_bucket:
            raise ValueError(
                "Cache bucket is not set in config, consider calling the `create` method first."
            )
        return self.cache_bucket

    def to_json(self) -> dict[str, Any]:
        """Convert the Config instance to a dictionary, handling complex types."""
        result = asdict(self)
        result["aws_env"] = self.aws_env.value  # serialize AwsEnv manually
        return result


def build_run_output_identifier() -> RunOutputIdentifier:
    """Builds an identifier from the start time and name of the flow run."""
    run_context = get_run_context()
    if not run_context:
        raise MissingContextError()
    start_time = run_context.flow_run.start_time.replace(tzinfo=None).isoformat(
        timespec="minutes"
    )
    run_name = run_context.flow_run.name
    return f"{start_time}-{run_name}"


async def get_all_labelled_passages_for_one_document(
    document_id: DocumentImportId,
    classifier_specs: list[ClassifierSpec],
    config: Config,
) -> dict[SpecStr, list[LabelledPassage]]:
    """Get the labelled passages from s3."""
    s3 = boto3.client("s3")

    labelled_passages = defaultdict(list)

    for spec in classifier_specs:
        s3_uri = S3Uri(
            bucket=config.cache_bucket,
            key=os.path.join(
                config.document_source_prefix,
                spec.name,
                spec.alias,
                f"{document_id}.json",
            ),
        )
        response = s3.get_object(Bucket=s3_uri.bucket, Key=s3_uri.key)
        body = response["Body"].read().decode("utf-8")
        spec_doc = [
            LabelledPassage.model_validate_json(passage) for passage in json.loads(body)
        ]
        labelled_passages[str(spec)].extend(spec_doc)

    return labelled_passages


def check_all_values_are_the_same(values: list[Any]) -> bool:
    """Check if all values are the same."""
    return len(set(values)) == 1


def validate_passages_are_same_except_concepts(passages: list[LabelledPassage]) -> None:
    """Check if passages are the same (except for metadata & spans)."""
    properties = [
        "id",
        "text",
    ]
    for property in properties:
        values = [getattr(passage, property) for passage in passages]
        if not check_all_values_are_the_same(values):
            unique_values = set(values)
            raise ValueError(
                f"Found a discrepancy in passage for {property}: {unique_values}"
            )


def combine_labelled_passages(
    labelled_passages: dict[SpecStr, list[LabelledPassage]],
) -> dict[TextBlockId, SerialisedVespaConcept]:
    """Combine the labelled passages across the different classifier specs."""
    labelled_passages_lists = list(labelled_passages.values())
    if not check_all_values_are_the_same([len(lpl) for lpl in labelled_passages_lists]):
        raise ValueError(
            f"The length of the labelled passages are not the same across classifier "
            f"outputs: {labelled_passages.keys()}"
        )

    combined_passages = {}
    for passages in zip(*labelled_passages_lists):
        validate_passages_are_same_except_concepts(passages)
        passage_id = passages[0].id

        all_vespa_concepts = []
        for passage in passages:
            vespa_concepts = convert_labelled_passage_to_concepts(passage)
            serialised_vespa_concepts = [
                vc.model_dump(mode="json") for vc in vespa_concepts
            ]
            all_vespa_concepts.extend(serialised_vespa_concepts)

        combined_passages[passage_id] = all_vespa_concepts

    return combined_passages


@task()
async def process_single_document(
    document_id: DocumentImportId,
    classifier_specs: list[ClassifierSpec],
    config: Config,
    run_output_identifier: RunOutputIdentifier,
) -> DocumentImportId | AggregationFailure:
    """Process a single document and return its status."""
    try:
        all_labelled_passages = await get_all_labelled_passages_for_one_document(
            document_id, classifier_specs, config
        )
        print(
            f"Combining {len(all_labelled_passages)} labelled passages for {document_id}"
        )
        vespa_concepts = combine_labelled_passages(all_labelled_passages)

        # Write to s3
        s3_uri = S3Uri(
            bucket=config.cache_bucket,
            key=os.path.join(
                config.aggregate_inference_results_prefix,
                run_output_identifier,
                f"{document_id}.json",
            ),
        )
        await asyncio.to_thread(
            s3_object_write_text, str(s3_uri), json.dumps(vespa_concepts)
        )

        # Duplicate to latest
        await s3_copy_file(
            source=s3_uri,
            target=S3Uri(
                bucket=config.cache_bucket,
                key=os.path.join(
                    config.aggregate_inference_results_prefix,
                    "latest",
                    f"{document_id}.json",
                ),
            ),
        )
        return document_id
    except ClientError as e:
        print(e.response)
        raise AggregationFailure(
            document_id=document_id, exception=e, context=e.response
        )
    except Exception as e:
        raise AggregationFailure(
            document_id=document_id, exception=e, context="Unknown error"
        )


@task()
async def process_n_documents(
    document_ids_batch: Sequence[DocumentImportId],
    classifier_specs: list[ClassifierSpec],
    config: Config,
    run_output_identifier: RunOutputIdentifier,
) -> list[DocumentImportId | AggregationFailure | BaseException]:
    """Process a batch of documents."""
    return await asyncio.gather(
        *(
            process_single_document(
                document_id, classifier_specs, config, run_output_identifier
            )
            for document_id in document_ids_batch
        ),
        return_exceptions=True,
    )


def handle_results(
    batched_results: Sequence[Sequence[DocumentImportId | AggregationFailure]],
) -> tuple[list[DocumentImportId], list[AggregationFailure]]:
    success_ids: list[DocumentImportId] = []
    failures: list[AggregationFailure] = []

    for batch in batched_results:
        for result in batch:
            if isinstance(result, AggregationFailure):
                failures.append(result)
            elif isinstance(result, str):
                success_ids.append(DocumentImportId(result))
            else:
                raise ValueError(f"Unknown result type: {type(result)}")

    return success_ids, failures


async def create_aggregate_inference_summary_artifact(
    config: Config,
    success_ids: list[DocumentImportId],
    failures: list[AggregationFailure],
) -> None:
    """Create a summary artifact of the aggregated inference results."""

    overview_description = f"""# Aggregate Inference Summary

## Overview
- **Environment**: {config.aws_env.value}
- **Documents processed**: {len(success_ids)}
- **Failed documents**: {len(failures)}/{len(success_ids)}
"""

    details = [
        {
            "Failed document ID": failure.document_id,
            "Exception": str(failure.exception),
            "Context": failure.context,
        }
        for failure in failures
    ]

    await create_table_artifact(
        key=f"aggregate-inference-{config.aws_env.value}",
        table=details,
        description=overview_description,
    )


def collect_stems_by_specs(config: Config) -> list[DocumentImportId]:
    """Collect the stems for the given specs."""
    document_ids = []
    specs = parse_spec_file(config.aws_env)
    for spec in specs:
        prefix = os.path.join(config.document_source_prefix, spec.name, spec.alias)
        document_ids.extend(
            collect_unique_file_stems_under_prefix(
                bucket_name=config.cache_bucket,
                prefix=prefix,
            )
        )

    return list(set(document_ids))


@flow(
    on_failure=[SlackNotify.message],
    on_crashed=[SlackNotify.message],
    timeout_seconds=None,
    log_prints=True,
    task_runner=ConcurrentTaskRunner(),
)
async def aggregate_inference_results(
    document_ids: None | list[DocumentImportId] = None,
    config: Config | None = None,
    max_concurrent_tasks: int = 10,
    batch_size: int = 10,
) -> RunOutputIdentifier:
    """Aggregate the inference results for the given document ids."""
    if not config:
        print("no config provided, creating one")
        config = await Config.create()

    if not document_ids:
        print(
            "no document ids provided, collecting all available from s3 under prefix: "
            f"{config.document_source_prefix}"
        )
        document_ids = collect_stems_by_specs(config)

    run_output_identifier = build_run_output_identifier()
    classifier_specs = parse_spec_file(config.aws_env)

    print(
        f"Aggregating inference results for {len(document_ids)} documents, using "
        f"{len(classifier_specs)} classifiers, outputting to {run_output_identifier}"
    )

    semaphore = asyncio.Semaphore(max_concurrent_tasks)

    # Create tasks for each document
    tasks = [
        wait_for_semaphore(
            semaphore,
            process_n_documents(
                document_ids_batch,
                classifier_specs,
                config,
                run_output_identifier,
            ),
        )
        for document_ids_batch in iterate_batch(document_ids, batch_size=batch_size)
    ]

    batched_results = await asyncio.gather(*tasks)
    success_ids, failures = handle_results(batched_results)

    await create_aggregate_inference_summary_artifact(
        config=config,
        success_ids=success_ids,
        failures=failures,
    )

    if failures:
        raise ValueError(
            f"Saw {len(failures)} failures when aggregating inference results"
        )

    return run_output_identifier
