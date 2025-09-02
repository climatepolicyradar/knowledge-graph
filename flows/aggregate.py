import json
import os
from collections.abc import AsyncGenerator, Iterable, Sequence
from typing import Any, TypeAlias, TypeVar

import aioboto3
import prefect.tasks as tasks
from botocore.exceptions import ClientError
from prefect import flow
from prefect.artifacts import create_markdown_artifact, create_table_artifact
from prefect.assets import materialize
from prefect.client.schemas.objects import FlowRun
from prefect.context import TaskRunContext, get_run_context
from prefect.futures import PrefectFuture, PrefectFutureList
from prefect.task_runners import ThreadPoolTaskRunner
from prefect.utilities.names import generate_slug
from pydantic import PositiveInt
from types_aiobotocore_s3.client import S3Client

from flows.boundary import (
    TextBlockId,
    convert_labelled_passage_to_concepts,
    s3_copy_file,
    s3_object_write_text_async,
)
from flows.classifier_specs.spec_interface import (
    ClassifierSpec,
    load_classifier_specs,
    should_skip_doc,
)
from flows.config import Config
from flows.inference import (
    deserialise_pydantic_list_with_fallback,
)
from flows.utils import (
    DocumentStem,
    S3Uri,
    SlackNotify,
    collect_unique_file_stems_under_prefix,
    iterate_batch,
    map_as_sub_flow,
)
from src.cloud import AwsEnv
from src.labelled_passage import LabelledPassage

T = TypeVar("T")
R = TypeVar("R")


DEFAULT_N_DOCUMENTS_IN_BATCH: PositiveInt = 20
DEFAULT_N_BATCHES: PositiveInt = 5

# A unique identifier for the run output made from the run context
RunOutputIdentifier: TypeAlias = str

# A string representation of a classifier spec (i.e. Q123:v4)
SpecStr: TypeAlias = str

# A serialised vespa concept, see cpr_sdk.models.search.Concept
SerialisedVespaConcept: TypeAlias = list[dict[str, str]]


class AggregationFailure(Exception):
    """A document failure."""

    def __init__(self, document_stem: DocumentStem, exception: Exception, context: str):
        super().__init__(
            f"{document_stem} | exception: {str(exception)} | context: {context}"
        )
        self.document_stem = document_stem
        self.exception = exception
        self.context = context


class AllSkippedFailure(Exception):
    """Every classifier was skipped for this document."""


def build_run_output_identifier() -> RunOutputIdentifier:
    """Builds an identifier from the start time and name of the flow run."""
    run_context = get_run_context()
    if isinstance(run_context, TaskRunContext):
        raise ValueError("expected flow run context but got task run context")

    if run_context.flow_run is None:
        raise ValueError("run context is missing flow run")

    if run_context.flow_run.start_time is None:
        raise ValueError("flow run didn't have a start time.")

    start_time = run_context.flow_run.start_time.replace(tzinfo=None).isoformat(
        timespec="minutes"
    )
    run_name = run_context.flow_run.name
    return f"{start_time}-{run_name}"


async def get_all_labelled_passages_for_one_document(
    s3: S3Client,
    document_stem: DocumentStem,
    classifier_specs: Sequence[ClassifierSpec],
    config: Config,
) -> AsyncGenerator[tuple[ClassifierSpec, list[LabelledPassage]], None]:
    """Get the labelled passages from S3."""

    for spec in classifier_specs:
        if should_skip_doc(document_stem, spec):
            continue

        s3_uri = generate_s3_uri_input(
            cache_bucket=config.cache_bucket_str,
            document_source_prefix=config.aggregate_document_source_prefix,
            classifier_spec=spec,
            document_stem=document_stem,
        )
        response = await s3.get_object(Bucket=s3_uri.bucket, Key=s3_uri.key)
        body = await response["Body"].read()
        content = body.decode("utf-8")

        labelled_passages = deserialise_pydantic_list_with_fallback(
            content, LabelledPassage
        )
        yield spec, labelled_passages


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


def task_input_hash(
    remove: Sequence[str],
    context: TaskRunContext,
    arguments: dict[str, Any],
) -> str | None:
    """
    Remove arguments from a task's input.

    Sometimes we don't want to, or can't, serialise them.
    """

    return tasks.task_input_hash(
        context,
        {k: v for k, v in arguments.items() if k not in remove},
    )


def generate_s3_uri_input(
    cache_bucket: str,
    document_source_prefix: str,
    classifier_spec: ClassifierSpec,
    document_stem: DocumentStem,
) -> S3Uri:
    return S3Uri(
        bucket=cache_bucket,
        key=os.path.join(
            document_source_prefix,
            classifier_spec.wikibase_id,
            classifier_spec.classifier_id,
            f"{document_stem}.json",
        ),
    )


def generate_s3_uri_output(
    cache_bucket: str,
    aggregate_inference_results_prefix: str,
    run_output_identifier: RunOutputIdentifier,
    document_stem: DocumentStem,
) -> S3Uri:
    return S3Uri(
        bucket=cache_bucket,
        key=os.path.join(
            aggregate_inference_results_prefix,
            run_output_identifier,
            f"{document_stem}.json",
        ),
    )


def task_run_name(parameters: dict[str, Any]) -> str:
    document_id = parameters.get("document_id", "unknown")
    slug = generate_slug(2)
    return f"aggregate-single-document-{document_id}-{slug}"


@materialize(
    "foo://bar",  # Asset key is not known yet
    task_run_name=task_run_name,  # pyright: ignore[reportArgumentType]
    retries=1,
    persist_result=False,
)
async def process_document(
    document_stem: DocumentStem,
    classifier_specs: Sequence[ClassifierSpec],
    config: Config,
    run_output_identifier: RunOutputIdentifier,
) -> DocumentStem:
    """Process a single document and return its status."""
    try:
        session = aioboto3.Session(region_name=config.bucket_region)
        async with session.client("s3") as s3:
            print("Fetching labelled passages for", document_stem)

            concepts_for_vespa: dict[TextBlockId, SerialisedVespaConcept] = {}
            async for (
                spec,
                labelled_passages,
            ) in get_all_labelled_passages_for_one_document(
                s3, document_stem, classifier_specs, config
            ):
                # `concepts_for_vespa` starts empty so Validation not
                # needed initially
                if not concepts_for_vespa:
                    for passage in labelled_passages:
                        concepts_for_vespa[TextBlockId(passage.id)] = [
                            vc.model_dump(mode="json")
                            for vc in convert_labelled_passage_to_concepts(passage)
                        ]
                    continue

                if len(labelled_passages) != len(concepts_for_vespa.keys()):
                    raise ValueError(
                        f"The number of passages diverge when appending {spec}: "
                        + f"{len(labelled_passages)=} != {len(concepts_for_vespa)=}"
                    )

                for passage, text_block_id in zip(
                    labelled_passages, concepts_for_vespa.keys()
                ):
                    if passage.id != text_block_id:
                        raise ValueError(
                            f"The text_block id diverges for {spec} when "
                            + "compared with what has been collected so far:"
                            + f"{passage.id=} != {text_block_id=}"
                        )
                    serialised_concepts = [
                        vc.model_dump(mode="json")
                        for vc in convert_labelled_passage_to_concepts(passage)
                    ]
                    concepts_for_vespa[TextBlockId(passage.id)].extend(
                        serialised_concepts
                    )

            # Validation, checking if we skipped all concepts for a document
            if not concepts_for_vespa:
                raise AllSkippedFailure

            # Write to s3
            s3_uri = generate_s3_uri_output(
                cache_bucket=config.cache_bucket_str,
                aggregate_inference_results_prefix=config.aggregate_inference_results_prefix,
                run_output_identifier=run_output_identifier,
                document_stem=document_stem,
            )
            await s3_object_write_text_async(s3, s3_uri, json.dumps(concepts_for_vespa))

            # Duplicate to latest
            await s3_copy_file(
                s3,
                source=s3_uri,
                target=S3Uri(
                    bucket=config.cache_bucket_str,
                    key=os.path.join(
                        config.aggregate_inference_results_prefix,
                        "latest",
                        f"{document_stem}.json",
                    ),
                ),
            )
            return document_stem
    except AllSkippedFailure as e:
        raise AggregationFailure(
            document_stem=document_stem, exception=e, context=repr(e)
        )
    except ClientError as e:
        print(f"ClientError: {e.response}")
        raise AggregationFailure(
            document_stem=document_stem,
            exception=e,
            context=str(e.response),
        )
    except Exception as e:
        raise AggregationFailure(
            document_stem=document_stem, exception=e, context=repr(e)
        )


def handle_results(
    results: Sequence[DocumentStem | AggregationFailure],
) -> tuple[list[DocumentStem], list[AggregationFailure]]:
    success_stems: list[DocumentStem] = []
    failures: list[AggregationFailure] = []

    for result in results:
        if isinstance(result, AggregationFailure):
            failures.append(result)
        elif isinstance(result, str):
            success_stems.append(DocumentStem(result))
        else:
            raise ValueError(f"Unknown result type: {type(result)}")

    return success_stems, failures


async def create_aggregate_inference_summary_artifact(
    config: Config,
    success_stems: list[DocumentStem],
    failures: list[AggregationFailure],
) -> None:
    """Create a summary artifact of the aggregated inference results."""

    overview_description = f"""# Aggregate Inference Summary

## Overview
- **Environment**: {config.aws_env.value}
- **Documents processed**: {len(success_stems)}
- **Failed documents**: {len(failures)}/{len(success_stems)}
"""

    details = [
        {
            "Failed document Stem": failure.document_stem,
            "Exception": str(failure.exception),
            "Context": failure.context,
        }
        for failure in failures
    ]

    await create_table_artifact(  # pyright: ignore[reportGeneralTypeIssues]
        key=f"aggregate-inference-{config.aws_env.value}",
        table=details,
        description=overview_description,
    )


async def create_aggregate_inference_overall_summary_artifact(
    aws_env: AwsEnv,
    document_stems: Sequence[DocumentStem],
    classifier_specs: list[ClassifierSpec],
    run_output_identifier: RunOutputIdentifier,
    successes: Sequence[FlowRun],
    failures: Sequence[BaseException | FlowRun],
) -> None:
    """Create a summary artifact of the overall aggregated inference results."""
    markdown_content = f"""# Aggregate Inference Overall Summary

## Overview
- **Environment**: {aws_env.value}
- **Run Output Identifier**: `{run_output_identifier}`
- **Total classifier specs.**: {len(classifier_specs)}
- **Total documents**: {len(document_stems)}
- **Successful batches**: {len(successes)}
- **Failed batches**: {len(failures)}
"""

    await create_markdown_artifact(  # pyright: ignore[reportGeneralTypeIssues]
        key=f"aggregate-inference-overall-{aws_env.value}",
        markdown=markdown_content,
    )


def collect_stems_by_specs(config: Config) -> list[DocumentStem]:
    """Collect the stems for the given specs."""
    document_stems = []
    specs = load_classifier_specs(config.aws_env)
    for spec in specs:
        prefix = os.path.join(
            config.aggregate_document_source_prefix,
            spec.wikibase_id,
            spec.classifier_id,
        )
        document_stems.extend(
            collect_unique_file_stems_under_prefix(
                bucket_name=config.cache_bucket_str,
                prefix=prefix,
            )
        )

    return list(set(document_stems))


@flow(  # pyright: ignore[reportCallIssue]
    timeout_seconds=None,
    log_prints=True,
    task_runner=ThreadPoolTaskRunner(
        # This is valid, as per the docs[1].
        #
        # [1]: https://github.com/PrefectHQ/prefect/blob/01f9d5e7d5204f5b8760b431d72db52dd78e6aca/docs/v3/concepts/task-runners.mdx#L49
        max_workers=DEFAULT_N_DOCUMENTS_IN_BATCH  # pyright: ignore[reportArgumentType]
    ),
)
async def aggregate_batch_of_documents(
    document_stems: Sequence[DocumentStem],
    config_json: dict[str, Any],
    classifier_specs: Sequence[ClassifierSpec],
    run_output_identifier: RunOutputIdentifier,
) -> RunOutputIdentifier:
    """Aggregate the inference results for the given document ids."""
    config = Config.model_validate(config_json)

    tasks: list[PrefectFuture[DocumentStem]] = []

    print("submitting tasks")
    for document_stem in document_stems:
        assets: Sequence[str] = [
            str(
                generate_s3_uri_output(
                    cache_bucket=config.cache_bucket_str,
                    aggregate_inference_results_prefix=config.aggregate_inference_results_prefix,
                    run_output_identifier=run_output_identifier,
                    document_stem=document_stem,
                )
            )
        ]
        asset_deps: list[str] = [
            str(
                generate_s3_uri_input(
                    cache_bucket=config.cache_bucket_str,
                    document_source_prefix=config.aggregate_document_source_prefix,
                    classifier_spec=spec,
                    document_stem=document_stem,
                )
            )
            for spec in classifier_specs
        ]

        tasks.append(
            process_document.with_options(  # pyright: ignore[reportFunctionMemberAccess, reportArgumentType]
                assets=assets,
                asset_deps=asset_deps,  # pyright: ignore[reportArgumentType]
            ).submit(
                document_stem=document_stem,
                classifier_specs=classifier_specs,
                config=config,
                run_output_identifier=run_output_identifier,
            )
        )

    # This is valid, as per the source code[1].
    #
    # [1]: https://github.com/PrefectHQ/prefect/blob/01f9d5e7d5204f5b8760b431d72db52dd78e6aca/src/prefect/task_runners.py#L183-L213
    futures: PrefectFutureList[DocumentStem] = PrefectFutureList(tasks)  # pyright: ignore[reportAbstractUsage]

    print("getting results")
    results = futures.result(raise_on_failure=False)

    print("handling results")
    success_stems, failures = handle_results(results)

    print("creating summary artifact")
    await create_aggregate_inference_summary_artifact(
        config=config,
        success_stems=success_stems,
        failures=failures,
    )

    if failures:
        raise ValueError(
            f"Saw {len(failures)} failures when aggregating inference results"
        )

    return run_output_identifier


@flow(
    on_failure=[SlackNotify.message],
    on_crashed=[SlackNotify.message],
    timeout_seconds=None,
    log_prints=True,
)
async def aggregate(
    document_stems: None | Sequence[DocumentStem] = None,
    config: Config | None = None,
    n_documents_in_batch: PositiveInt = DEFAULT_N_DOCUMENTS_IN_BATCH,
    n_batches: PositiveInt = DEFAULT_N_BATCHES,
) -> RunOutputIdentifier:
    """Aggregate the inference results for the given document ids."""
    if not config:
        print("no config provided, creating one")
        config = await Config.create()

    if not document_stems:
        print(
            "no document stems provided, collecting all available from S3 under prefix: "
            + f"{config.aggregate_document_source_prefix}"
        )
        document_stems = collect_stems_by_specs(config)

    run_output_identifier = build_run_output_identifier()
    classifier_specs = load_classifier_specs(config.aws_env)

    print(
        f"Aggregating inference results for {len(document_stems)} documents, using "
        + f"{len(classifier_specs)} classifiers, outputting to {run_output_identifier}"
    )

    batches = iterate_batch(
        document_stems,
        n_documents_in_batch,
    )

    def parameters(batch: Sequence[DocumentStem]) -> dict[str, Any]:
        return {
            "document_stems": batch,
            "config_json": config.model_dump(),
            "classifier_specs": classifier_specs,
            "run_output_identifier": run_output_identifier,
        }

    parameterised_batches: Iterable[dict[str, Any]] = (
        parameters(batch) for batch in batches
    )

    successes, failures = await map_as_sub_flow(  # pyright: ignore[reportCallIssue]
        # The typing doesn't pick up the Flow decorator
        fn=aggregate_batch_of_documents,  # pyright: ignore[reportArgumentType]
        aws_env=config.aws_env,
        counter=n_batches,
        parameterised_batches=parameterised_batches,
        unwrap_result=False,
    )

    await create_aggregate_inference_overall_summary_artifact(
        aws_env=config.aws_env,
        document_stems=document_stems,
        classifier_specs=classifier_specs,
        run_output_identifier=run_output_identifier,
        successes=successes,
        failures=failures,
    )

    if failures:
        raise ValueError(f"{len(failures)}/{len(failures) + len(successes)} failed")

    return run_output_identifier
