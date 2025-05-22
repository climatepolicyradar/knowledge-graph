import asyncio
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import boto3
from prefect import flow
from prefect.context import get_run_context
from prefect.exceptions import MissingContextError
from prefect.logging import get_run_logger

from flows.boundary import convert_labelled_passage_to_concepts, s3_object_write_text
from flows.inference import DOCUMENT_TARGET_PREFIX_DEFAULT
from flows.utils import (
    SlackNotify,
)
from scripts.cloud import (
    AwsEnv,
    ClassifierSpec,
    get_prefect_job_variable,
)
from scripts.update_classifier_spec import parse_spec_file
from src.labelled_passage import LabelledPassage

INFERENCE_RESULTS_PREFIX = "inference_results"


@dataclass()
class Config:
    """Configuration used across flow runs."""

    cache_bucket: str | None = None
    document_source_prefix: str = DOCUMENT_TARGET_PREFIX_DEFAULT
    bucket_region: str = "eu-west-1"
    aws_env: AwsEnv = AwsEnv(os.environ["AWS_ENV"])

    def __post_init__(self):
        """Create a new Config instance with initialized values."""
        logger = get_run_logger()

        if not self.cache_bucket:
            logger.info(
                "no cache bucket provided, getting it from Prefect job variable"
            )
            self.cache_bucket = asyncio.run(
                get_prefect_job_variable("pipeline_cache_bucket_name")
            )


def build_run_output_prefix() -> str:
    """Get the name of the flow run."""
    run_context = get_run_context()
    if not run_context:
        raise MissingContextError()
    start_time = run_context.flow_run.start_time.replace(tzinfo=None).isoformat(
        timespec="minutes"
    )
    run_name = run_context.flow_run.name
    return f"{INFERENCE_RESULTS_PREFIX}/{start_time}-{run_name}"


def get_all_labelled_passages_for_one_document(
    document_id: str, classifier_specs: list[ClassifierSpec], config: Config
) -> dict[str, list[LabelledPassage]]:
    """Get the labelled passages from s3."""
    s3 = boto3.client("s3")

    labelled_passages = defaultdict(list)
    for spec in classifier_specs:
        key = os.path.join(
            DOCUMENT_TARGET_PREFIX_DEFAULT, spec.name, spec.alias, f"{document_id}.json"
        )
        response = s3.get_object(Bucket=config.cache_bucket, Key=key)
        body = response["Body"].read().decode("utf-8")
        spec_doc = [
            LabelledPassage.model_validate_json(passage) for passage in json.loads(body)
        ]
        labelled_passages[f"{spec.name}:{spec.alias}"].extend(spec_doc)

    return labelled_passages


def check_all_values_are_the_same(values: list[Any]) -> bool:
    """Check if all values are the same."""
    if len(set(values)) == 1:
        return True
    return False


def validate_passages_are_same_except_concepts(passages: list[LabelledPassage]) -> None:
    """Check if two passages are the same except for the spans."""
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
    labelled_passages: dict[str, list[LabelledPassage]],
) -> dict[str, list[dict[str, str]]]:
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


@flow(
    on_failure=[SlackNotify.message],
    on_crashed=[SlackNotify.message],
    timeout_seconds=None,
    log_prints=True,
)
def aggregate_inference_results(document_ids: list[str], config: Config | None = None):
    """Aggregate the inference results for the given document ids."""
    if not config:
        print("no config provided, creating one")
        config = Config()

    run_output_prefix = build_run_output_prefix()
    classifier_specs = parse_spec_file(config.aws_env)

    print(
        f"Aggregating inference results for {len(document_ids)} documents, using"
        f"{len(classifier_specs)} classifiers, outputting to {run_output_prefix}"
    )
    for document_id in document_ids:
        all_labelled_passages = get_all_labelled_passages_for_one_document(
            document_id, classifier_specs, config
        )
        vespa_concepts = combine_labelled_passages(all_labelled_passages)

        # Write to s3
        s3_uri = f"s3://{config.cache_bucket}/{run_output_prefix}/{document_id}.json"
        s3_object_write_text(s3_uri, json.dumps(vespa_concepts))

    return run_output_prefix
