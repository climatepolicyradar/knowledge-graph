"""
Do Pasages for Documents in S3 Match Passages in Vespa

=================================================

A script to check whether the passages for documents in a given S3 prefix
are present in a Vespa database. It compares the number of passages in S3 with those
in Vespa and generates a report of any discrepancies.

Set your environment variables for VESPA_INSTANCE_URL and authenticate your AWS CLI
before running the script. If providing the `vespa_cert_directory` argument,
ensure it points to the directory containing your Vespa certificates. These must be
named key.pem and cert.pem.

python -m scripts.audit.do_s3_prefix_docs_have_passages_in_vespa 
   "cpr-staging-data-pipeline-cache" \
   "indexer_input" \
   "~./.vespa/climate-policy-radar.navigator_dev.default"
"""

import asyncio
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from functools import wraps
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from cpr_sdk.parser_models import ParserOutput
from cpr_sdk.search_adaptors import VespaSearchAdapter
from cpr_sdk.utils import dig

from flows.boundary import (
    DocumentStem,
)
from flows.index_from_aggregate_results import load_json_data_from_s3
from flows.utils import (
    DocumentImportId,
    remove_translated_suffix,
)
from scripts.audit.do_classifier_specs_have_results import collect_file_names

app = typer.Typer()


VESPA_INSTANCE_URL = os.environ["VESPA_INSTANCE_URL"]


@dataclass
class Result:
    """Result of checking a single document in s3 against vespa."""

    document_id: DocumentImportId
    passages_mismatch: bool
    passages_count_in_vespa: int
    passages_count_in_s3: int


def document_passages_in_s3(bucket_name: str, s3_key: str) -> int:
    """Check how many passages we have in s3 relating to a document id."""

    data = load_json_data_from_s3(
        bucket=bucket_name,
        key=s3_key,
    )

    parser_output = ParserOutput.model_validate(data)

    return len(parser_output.text_blocks)


def check_document_passages(
    document: tuple[DocumentImportId, DocumentStem],
    bucket_name: str,
    s3_prefix: str,
    vespa_passage_counts: dict[DocumentImportId, int],
) -> Result:
    """Check if the number of passages in S3 matches those in Vespa for a document."""

    document_id: DocumentImportId = document[0]
    document_stem: DocumentStem = document[1]

    s3_passages_count: int = document_passages_in_s3(
        bucket_name=bucket_name, s3_key=os.path.join(s3_prefix, f"{document_stem}.json")
    )

    # TODO: Think whether default is right here.
    vespa_passage_count_for_document: int = vespa_passage_counts.get(document_id, 0)

    return Result(
        document_id=document_id,
        passages_mismatch=(vespa_passage_count_for_document != s3_passages_count),
        passages_count_in_vespa=vespa_passage_count_for_document,
        passages_count_in_s3=s3_passages_count,
    )


def get_vespa_passage_counts(vespa: VespaSearchAdapter) -> dict[DocumentImportId, int]:
    """
    Get the number of passages for each document in Vespa.

    We make one query to vespa to get the count of passages for each document via
    grouping.
    """

    # TODO: Update to the vespa query builder
    vespa_query_response = vespa.client.query(
        yql=(
            "select * from document_passage where true limit 0 "
            "| all(group(document_import_id) each(output(count())))"
        )
    )

    vespa_group_hits: list[dict[str, Any]] = dig(
        vespa_query_response.get_json(),
        "root",
        "children",
        0,
        "children",
        0,
        "children",
    )

    vespa_passage_counts: dict[DocumentImportId, int] = {
        DocumentImportId(hit["value"]): hit["fields"]["count()"]
        for hit in vespa_group_hits
        if hit["value"] != ""
    }

    return vespa_passage_counts


def coro(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(f(*args, **kwargs))

    return wrapper


@app.command()
@coro
async def check_passages(
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

    start_time = time.time()

    typer.echo(
        f"Running with the following parameters:\n"
        f"Bucket Name: {bucket_name}\n"
        f"S3 Prefix: {s3_prefix}\n"
        f"Vespa Cert Directory: {vespa_cert_directory}\n"
        f"Report File Path: {report_file_path}\n"
        f"Max workers: {max_workers}\n"
        f"Start Time: {start_time}\n"
    )

    typer.echo("Retrieving passage counts from vespa...")
    vespa = VespaSearchAdapter(
        instance_url=VESPA_INSTANCE_URL,
        cert_directory=vespa_cert_directory,
    )
    vespa_passage_counts = get_vespa_passage_counts(vespa)
    typer.echo(f"Found {len(vespa_passage_counts)} documents with passages in Vespa.")

    typer.echo("Retrieving file names from s3...")
    s3_file_names: list[str] = collect_file_names(bucket_name, s3_prefix)
    typer.echo(f"Found {len(s3_file_names)} file names in s3 prefix {s3_prefix}")

    typer.echo("Constructing document ids from file names...")
    s3_document_ids: list[tuple[DocumentImportId, DocumentStem]] = list(
        (
            DocumentImportId(Path(remove_translated_suffix(DocumentStem(file))).stem),
            DocumentStem(file),
        )
        for file in s3_file_names
        if file.endswith(".json")
    )
    typer.echo(
        f"Constructed {len(s3_document_ids)} document ids from {len(s3_file_names)} file names."
    )

    s3_document_ids_not_in_vespa = []
    s3_document_ids_in_vespa = []
    for doc_id, stem in s3_document_ids:
        if doc_id in vespa_passage_counts:
            s3_document_ids_in_vespa.append((doc_id, stem))
        else:
            s3_document_ids_not_in_vespa.append((doc_id, stem))

    if s3_document_ids_not_in_vespa:
        typer.echo(
            f"Found {len(s3_document_ids_not_in_vespa)} document ids in s3 prefix "
            f"{s3_prefix} that are not in Vespa. These will be skipped."
        )
        missing_in_vespa_data = [
            {"document_id": str(doc_id), "document_stem": stem}
            for doc_id, stem in s3_document_ids_not_in_vespa
        ]
        # TODO: Make name comfigurable or add to report
        with open("missing_in_vespa.json", "w") as f:
            json.dump(missing_in_vespa_data, f, indent=2)

    # TEMP
    s3_document_ids_in_vespa = s3_document_ids_in_vespa[:100]
    # TEMP

    missing_passage_documents = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                check_document_passages,
                doc_id,
                bucket_name,
                s3_prefix,
                vespa_passage_counts,
            ): doc_id
            for doc_id, _ in s3_document_ids_in_vespa
        }

        for future in as_completed(futures.keys()):
            result = future.result()
            if result.passages_mismatch:
                missing_passage_documents.append(result)

    typer.echo("Writing results to CSV...")
    missing_df = pd.DataFrame(
        [
            {
                "document_id": result.document_id,
                "passages_count_in_s3": result.passages_count_in_s3,
                "passages_count_in_vespa": result.passages_count_in_vespa,
                "passages_mismatch": result.passages_mismatch,
            }
            for result in missing_passage_documents
        ]
    )
    missing_df.to_csv(report_file_path, index=False)
    typer.echo(f"Results written to {report_file_path}")

    elapsed_time = time.time() - start_time
    typer.echo(
        f"Completed in {elapsed_time:.2f} seconds. "
        f"Found {len(missing_passage_documents)} documents with mismatched passages."
    )


if __name__ == "__main__":
    app()
