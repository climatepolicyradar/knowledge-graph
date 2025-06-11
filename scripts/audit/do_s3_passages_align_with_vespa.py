"""
Do Passages for Documents in S3 Match Passages in Vespa

=================================================

A script to check whether the passages for documents in a given S3 prefix
are present in a Vespa database. It compares the number of passages in S3 with those
in Vespa and generates a report of any discrepancies.

We retrieve all the passage counts relating to document ids in vespa via a query,
we then identify the passage count in each s3 object relating to the document and create
a result after comparing the two.

Set your environment variables for VESPA_INSTANCE_URL and authenticate your AWS CLI
before running the script. If providing the `vespa_cert_directory` argument,
ensure it points to the directory containing your Vespa certificates. These must be
named key.pem and cert.pem.

python -m scripts.audit.do_s3_passages_align_with_vespa
   "cpr-staging-data-pipeline-cache" \
   "indexer_input" \
   "~./.vespa/climate-policy-radar.navigator_dev.default"
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import typer
import vespa.querybuilder as qb
from cpr_sdk.parser_models import ParserOutput
from cpr_sdk.search_adaptors import VespaSearchAdapter
from cpr_sdk.utils import dig
from tenacity import Retrying, stop_after_attempt
from vespa.package import Document, Schema
from vespa.querybuilder import Grouping as G

from flows.boundary import DocumentStem
from flows.index_from_aggregate_results import load_json_data_from_s3
from flows.utils import (
    DocumentImportId,
    remove_translated_suffix,
)
from scripts.audit.do_classifier_specs_have_results import collect_file_names

app = typer.Typer()


VESPA_INSTANCE_URL = os.environ.get("VESPA_INSTANCE_URL", None)


@dataclass
class Result:
    """Result of checking a single document in s3 against vespa."""

    document_id: DocumentImportId
    document_file_name: str
    passages_mismatch: bool
    passages_count_in_vespa: int
    passages_count_in_s3: int
    failed_to_load: bool = False


def document_passages_in_s3(bucket_name: str, s3_key: str) -> int:
    """Check how many passages we have in s3 relating to a document id."""

    data = load_json_data_from_s3(
        bucket=bucket_name,
        key=s3_key,
    )

    parser_output = ParserOutput.model_validate(data)

    return len(parser_output.text_blocks)


def check_document_passages(
    document_id: DocumentImportId,
    document_file_name: str,
    bucket_name: str,
    s3_prefix: str,
    vespa_passage_counts: dict[DocumentImportId, int],
) -> Result:
    """Check if the number of passages in S3 matches those in Vespa."""

    failed_to_load: bool = False
    s3_passage_count: int = 0

    try:
        for attempt in Retrying(stop=stop_after_attempt(3)):
            with attempt:
                s3_passage_count: int = document_passages_in_s3(
                    bucket_name=bucket_name,
                    s3_key=os.path.join(s3_prefix, document_file_name),
                )
    except Exception as e:
        typer.echo(f"Failed to load document {document_id} from S3: {e}.")
        failed_to_load = True

    vespa_passage_count: int = vespa_passage_counts.get(document_id, 0)

    return Result(
        document_id=document_id,
        document_file_name=document_file_name,
        passages_mismatch=(vespa_passage_count != s3_passage_count),
        passages_count_in_vespa=vespa_passage_count,
        passages_count_in_s3=s3_passage_count,
        failed_to_load=failed_to_load,
    )


def get_vespa_passage_counts(vespa: VespaSearchAdapter) -> dict[DocumentImportId, int]:
    """
    Get the number of passages for each document in Vespa.

    We make one query to vespa to get the count of passages for each document via
    grouping.
    """

    grouping = G.all(G.group("document_import_id"), G.each(G.output(G.count())))

    query: qb.Query = (
        qb.select("*")  # pyright: ignore[reportAttributeAccessIssue]
        .from_(
            Schema(name="document_passage", document=Document()),
        )
        .where(True)
        .set_limit(0)
        .groupby(grouping)
    )

    vespa_query_response = vespa.client.query(yql=query)

    vespa_group_hits: list[dict[str, Any]] = dig(
        vespa_query_response.get_json(),
        "root",
        "children",
        0,
        "children",
        0,
        "children",
    )

    vespa_passage_counts: dict[DocumentImportId, int] = {}
    for hit in vespa_group_hits:
        if hit["value"] != "":
            document_id = DocumentImportId(hit["value"])
            count = hit["fields"]["count()"]
            vespa_passage_counts[document_id] = count

    return vespa_passage_counts


@app.command()
def check_passages(
    bucket_name: str = typer.Argument(
        help=(
            "Name of the s3 bucket, should be the root without protocol or prefix"
            "i.e. my-bucket-name"
        )
    ),
    s3_prefix: str = typer.Argument(
        default="indexer_input",
        help="The S3 prefix to check for documents, e.g., 'indexer_input",
    ),
    vespa_cert_directory: str = typer.Argument(
        default=None,
        help="Directory containing Vespa certificates for secure connection",
    ),
    report_file_path: str = typer.Argument(
        default="missing_passages.csv",
        help="Path to save the report of missing passages",
    ),
    max_workers: int = typer.Argument(
        default=10,
        help="MMaximum number of concurrent tasks to run",
    ),
) -> None:
    """
    Compare documents in an s3 prefix against the relating data in the vespa database.

    The environent to run against (prod, sandbox, etc.) is determined by the vespa
    instance url that is set as well as the aws environment that your terminal is
    authenticated against.
    """

    if not VESPA_INSTANCE_URL:
        typer.echo(
            "Please set the VESPA_INSTANCE_URL environment variable to the Vespa instance URL."
        )
        raise typer.Exit(code=1)

    start_time = time.time()

    typer.echo(
        f"Running with the following parameters:\n"
        f"Bucket Name: {bucket_name}\n"
        f"S3 Prefix: {s3_prefix}\n"
        f"Vespa Cert Directory: {vespa_cert_directory}\n"
        f"Report File Path: {report_file_path}\n"
        f"Max workers: {max_workers}\n"
    )

    typer.echo("Retrieving passage counts from vespa...")
    vespa = VespaSearchAdapter(
        instance_url=VESPA_INSTANCE_URL,
        cert_directory=vespa_cert_directory,
    )
    vespa_passage_counts = get_vespa_passage_counts(vespa)
    typer.echo(f"Found {len(vespa_passage_counts)} documents with passages in Vespa.")

    typer.echo("Retrieving file names from s3...")
    s3_file_names: list[str] = [
        f for f in collect_file_names(bucket_name, s3_prefix) if f.endswith(".json")
    ]
    typer.echo(f"Found {len(s3_file_names)} file names in s3 prefix {s3_prefix}")

    typer.echo("Constructing document ids from file names...")
    s3_document_ids: list[tuple[DocumentImportId, str]] = list(
        (
            DocumentImportId(remove_translated_suffix(DocumentStem(Path(file).stem))),
            file,
        )
        for file in s3_file_names
    )
    typer.echo(
        f"Constructed {len(s3_document_ids)} doc ids from {len(s3_file_names)} file names."
    )
    missing_passage_documents = []
    total_documents = len(s3_document_ids)
    completed = 0

    typer.echo(f"Checking passages for {len(s3_document_ids)} documents...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                check_document_passages,
                document_id,
                document_file_name,
                bucket_name,
                s3_prefix,
                vespa_passage_counts,
            ): document_id
            for document_id, document_file_name in s3_document_ids
        }

        for future in as_completed(futures.keys()):
            completed += 1
            if completed % 100 == 0 or completed == total_documents:
                typer.echo(
                    f"Progress: {completed}/{total_documents} documents checked "
                    f"({(completed / total_documents) * 100:.1f}%)"
                )

            result = future.result()
            if result.passages_mismatch or result.failed_to_load:
                missing_passage_documents.append(result)
    typer.echo(f"Completed checking all {total_documents} documents.")

    typer.echo("Writing results to CSV...")
    missing_df = pd.DataFrame(
        [
            {
                "document_id": result.document_id,
                "document_file_name": result.document_file_name,
                "passages_count_in_s3": result.passages_count_in_s3,
                "passages_count_in_vespa": result.passages_count_in_vespa,
                "passages_mismatch": result.passages_mismatch,
                "failed_to_load": result.failed_to_load,
            }
            for result in missing_passage_documents
        ]
    )
    missing_df.to_csv(report_file_path, index=False)
    typer.echo(f"Results written to {report_file_path}")

    elapsed_time = time.time() - start_time
    typer.echo(
        f"Completed in {elapsed_time:.2f} seconds. "
        f"Found {len(missing_passage_documents)} documents with mismatched passages "
        "or that failed to load."
    )


if __name__ == "__main__":
    app()
