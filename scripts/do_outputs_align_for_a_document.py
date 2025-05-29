import asyncio
import json
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import typer
import yaml  # pyright: ignore[reportMissingModuleSource]
from cpr_sdk.models.search import Document as VespaDocument
from cpr_sdk.models.search import Passage
from cpr_sdk.search_adaptors import VespaSearchAdapter
from rich.console import Console
from rich.table import Table
from vespa.application import Vespa
from vespa.exceptions import VespaError

from flows.boundary import (
    DocumentImportId,
    get_document_passages_from_vespa__generator,
)
from scripts.cloud import AwsEnv

app = typer.Typer()
console = Console()


VESPA_INSTANCE_URL = os.environ["VESPA_INSTANCE_URL"]
INFERENCE_PREFIX = "labelled_passages"
AGGREGATED_RESULTS_PREFIX = "inference_results"
YAML_FILES_MAP = {
    "prod": "flows/classifier_specs/prod.yaml",
    "staging": "flows/classifier_specs/staging.yaml",
    "sandbox": "flows/classifier_specs/sandbox.yaml",
    "labs": "flows/classifier_specs/labs.yaml",
}


class Profiler:
    """Context manager for profiling and printing the duration"""

    def __init__(self, should_profile: bool = False):
        self.start_time = time.perf_counter()
        self.end_time = None
        self.duration = None
        self.should_profile = should_profile

    def __enter__(self):
        """Start the timer."""
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Stop the timer and print the duration."""
        self.end_time = time.perf_counter()
        self.duration = self.end_time - self.start_time

        if self.should_profile:
            typer.secho(f"> Done in: {self.duration:.2f} seconds", fg="white", dim=True)


def get_document_from_vespa(client: Vespa, document_id: str) -> VespaDocument:
    response = client.get_data(
        namespace="doc_search",
        schema="family_document",
        data_id=document_id,
    )

    if not response.is_successful():
        raise VespaError(f"Failed to get document from vespa: {response.json}")

    return VespaDocument.from_vespa_response(response.json)


async def get_passages_from_vespa(
    vespa: Vespa, document_id: str, max_workers: int
) -> list[Passage]:
    passages = []
    async with vespa.asyncio(connections=max_workers) as vespa_connection_pool:
        async for batch in get_document_passages_from_vespa__generator(
            document_import_id=DocumentImportId(document_id),
            vespa_connection_pool=vespa_connection_pool,
        ):
            batch_passages = [p[1] for p in batch.values()]
            passages.extend(batch_passages)
    return passages


def count_passage_concepts(passages: list[Passage]) -> dict:
    all_concepts = []
    for passage in passages:
        if passage.concepts:
            all_concepts.extend(
                [f"{concept.id}:{concept.name}" for concept in passage.concepts]
            )
    return {concept: count for concept, count in Counter(all_concepts).items()}


def determine_version_from_spec(specs, id_from_vespa):
    for spec in specs:
        wikibase_id, version = spec.split(":")
        if id_from_vespa == wikibase_id:
            return version

    raise ValueError(f"Couldn't find {id_from_vespa} in specs file")


def build_s3_path(id_from_vespa, spec_version, document_id):
    return os.path.join(
        INFERENCE_PREFIX, id_from_vespa, spec_version, f"{document_id}.json"
    )


def get_labelled_passages(bucket_name: str, s3_path: str) -> list[dict]:
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket_name, Key=s3_path)
    body = response["Body"].read().decode("utf-8")
    return [json.loads(passage) for passage in json.loads(body)]


def parse_concept_name_from_labelled_passage(s: dict) -> str:
    matches = set()
    for labeller in s["labellers"]:
        match = re.search(r'\w+\("([^"]+)"\)', labeller)
        if match:
            matches.add(match.group(1))
    if len(matches) != 1:
        raise ValueError(
            f"Found multiple name matches for {s['concept_id']}: {matches}"
        )
    return matches.pop()


def load_specs(yaml_file: str) -> list[str]:
    with open(yaml_file, "r") as file:
        specs = yaml.safe_load(file)
    return specs


def count_concepts_in_s3_labelled_passages(
    s3_labelled_passages: list[dict],
) -> list[str]:
    concept_list = []
    for passage in s3_labelled_passages:
        if len(passage["spans"]) > 0:
            for span in passage["spans"]:
                concept_list.append(
                    f"{span['concept_id']}:{parse_concept_name_from_labelled_passage(span)}"
                )
    return concept_list


def get_s3_count_for_one_spec(
    bucket_name: str, spec: str, document_id: str
) -> list[str]:
    classifier_id, spec_version = spec.split(":")
    s3_path = build_s3_path(classifier_id, spec_version, document_id)
    try:
        s3_labelled_passages = get_labelled_passages(bucket_name, s3_path)
    except Exception as e:
        typer.secho("x", fg="red", nl=False)
        typer.secho(f"Error getting {s3_path}: {e}", fg="red")
        return []
    counts = count_concepts_in_s3_labelled_passages(s3_labelled_passages)
    typer.secho(".", fg="green", nl=False)
    return counts


def numeric_ordering(wikibase_id: str) -> int:
    return int(re.sub(r"\D", "", wikibase_id))


def get_s3_concept_counts(
    specs: list[str], bucket_name: str, document_id: str, max_workers: int
) -> dict:
    all_s3_concept_counts = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_spec = {
            executor.submit(
                get_s3_count_for_one_spec, bucket_name, spec, document_id
            ): spec
            for spec in specs
        }
        for future in as_completed(future_to_spec):
            s3_concept_counts = future.result()
            all_s3_concept_counts.extend(s3_concept_counts)

    s3_concept_counts = {
        concept: count
        for concept, count in sorted(
            Counter(all_s3_concept_counts).items(), key=lambda x: numeric_ordering(x[0])
        )
    }
    typer.secho("", fg="green", nl=True)
    return s3_concept_counts


def count_s3_aggregated_concepts(
    bucket_name: str, document_id: str, aggregator_run_identifier: str
) -> dict:
    s3_path = os.path.join(
        AGGREGATED_RESULTS_PREFIX, aggregator_run_identifier, f"{document_id}.json"
    )
    s3 = boto3.client("s3")
    try:
        response = s3.get_object(Bucket=bucket_name, Key=s3_path)
    except Exception as e:
        typer.secho(f"Error getting {s3_path}: {e}", fg="red")
        return {}
    body = response["Body"].read().decode("utf-8")
    s3_aggregated_concepts = json.loads(body)

    concepts = []
    for c in s3_aggregated_concepts.values():
        concepts.extend([f"{i['id']}:{i['name']}" for i in c])

    return Counter(concepts)


def create_results_table(
    doc: VespaDocument,
    s3_concept_counts: dict,
    passage_concept_counts: dict,
    aggregated_concepts: None | dict[str, int],
) -> Table:
    table = Table(title="Concept Counts")
    table.add_column(
        "Concept", justify="left", style="cyan"
    )  # The name an ID of the concept
    table.add_column(
        "Inference Count", justify="right", style="green"
    )  # Count of that concept in the raw s3 inference output
    table.add_column(
        "Aggregated", justify="right", style="green"
    )  # Count of that concept in the s3 aggregated output
    table.add_column(
        "Passage Count", justify="right", style="green"
    )  # Count of that concept in vespa on passages
    table.add_column(
        "Document Count", justify="right", style="green"
    )  # Count from concepts_counts in the vespa document
    table.add_column(
        "Aligned", justify="right", style="magenta"
    )  # Whether the counts are aligned

    # TODO: Use classifier specs
    for concept, count in s3_concept_counts.items():
        document_count = doc.concept_counts.get(concept, 0) if doc.concept_counts else 0
        passage_count = passage_concept_counts.get(concept, 0)

        if aggregated_concepts:
            aggregated_count = aggregated_concepts.get(concept, 0)
            aligned = count == aggregated_count == document_count == passage_count
        else:
            aggregated_count = "/"
            aligned = count == document_count == passage_count
        table.add_row(
            concept,
            str(count),
            str(aggregated_count),
            str(passage_count),
            str(document_count),
            "✅" if aligned else "❌",
        )
    return table


def highlight_spans(passage: Passage) -> str:
    """
    Add a highlight to the text object of a passage

    Doesnt handle overlapping spans very well.
    """
    text = passage.text_block
    highlight_start = "\033[1;4;37m"  # bold, underline, white
    highlight_end = "\033[0m\033[32m"  # reset to green

    if passage.concepts:
        offset = 0
        for concept in passage.concepts:
            start = concept.start + offset
            end = concept.end + offset
            label = f"({concept.id}:{concept.name})"
            text = (
                text[:start]
                + f"{highlight_start}{text[start:end]}{label}{highlight_end}"
                + text[end:]
            )
            offset += len(highlight_start) + len(highlight_end) + len(label)
    return text


@app.command()
def main(
    document_id: str = typer.Argument(
        help="the document id of the document to show spans for"
    ),
    aws_env: AwsEnv = typer.Argument(
        help="Which aws environment to look for results in. Determines which spec file"
        "to use",
        default=AwsEnv.sandbox,
    ),
    bucket_name: str = typer.Argument(
        help=(
            "Name of the s3 bucket, should be the root without protocol or prefix"
            "i.e. my-bucket-name"
        )
    ),
    aggregator_run_identifier: str = typer.Option(
        default=None,
        help="The identifier of the aggregator run to use",
    ),
    print_vespa_passages: bool = typer.Option(
        default=False,
        help="Whether to print the whole text of the vespa passages",
    ),
    max_workers: int = typer.Option(
        default=8,
        help="Maximum number of parallel workers to use for checking specs",
    ),
    profile: bool = typer.Option(
        default=False,
        help="Whether to profile the code, intended to help with choosing worker count",
    ),
) -> None:
    vespa = VespaSearchAdapter(VESPA_INSTANCE_URL).client

    typer.secho("Collecting data from vespa document", fg="green")
    with Profiler(should_profile=profile):
        doc = get_document_from_vespa(vespa, document_id)

    typer.secho("Collecting data from vespa passages", fg="green")
    with Profiler(should_profile=profile):
        passages = asyncio.run(get_passages_from_vespa(vespa, document_id, max_workers))
    if print_vespa_passages:
        table = Table(title="Vespa Passages", show_lines=True)
        table.add_column("id", justify="left", style="cyan")
        table.add_column("text", justify="left", style="green")
        for passage in passages:
            table.add_row(
                passage.text_block_id,
                highlight_spans(passage),
            )
        console.print(table)

    passage_concept_counts = count_passage_concepts(passages)

    typer.secho("Collecting data from s3 inference output files", fg="green")
    specs = load_specs(YAML_FILES_MAP[aws_env])
    with Profiler(should_profile=profile):
        s3_concept_counts = get_s3_concept_counts(
            specs, bucket_name, document_id, max_workers
        )

    s3_aggregated_concepts = None
    if aggregator_run_identifier:
        with Profiler(should_profile=profile):
            s3_aggregated_concepts = count_s3_aggregated_concepts(
                bucket_name, document_id, aggregator_run_identifier
            )

    # Output in a table
    table = create_results_table(
        doc, s3_concept_counts, passage_concept_counts, s3_aggregated_concepts
    )
    typer.secho(
        f"Spans found for {document_id}, across {len(passages):,} passages:", fg="green"
    )
    console.print(table)


if __name__ == "__main__":
    app()
