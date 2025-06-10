"""
Do S3 Prefix Docs Have Passages in Vespa

=================================================

A script to check whether the passages for documents in a given S3 prefix
are present in a Vespa database. It compares the number of passages in S3 with those
in Vespa and generates a report of any discrepancies.

See an example run command below:

Set your environment variables for VESPA_INSTANCE_URL and authenticate your AWS CLI
before running the script. If providing the `vespa_cert_directory` argument,
ensure it points to the directory containing your Vespa certificates. These must be
named key.pem and cert.pem.

python -m scripts.audit.do_s3_prefix_docs_have_passages_in_vespa
    "cpr-staging-data-pipeline-cache"
    "indexer_input"
    "~./.vespa/climate-policy-radar.navigator_dev.default"
    "missing_passages.csv"
    50
    200
"""

import asyncio
import os
import time
from dataclasses import dataclass
from functools import wraps
from pathlib import Path

import pandas as pd
import typer
from cpr_sdk.parser_models import ParserOutput
from cpr_sdk.search_adaptors import VespaSearchAdapter
from vespa.application import VespaAsync

from flows.boundary import TextBlockId, get_document_passages_from_vespa__generator
from flows.index_from_aggregate_results import load_json_data_from_s3
from flows.utils import DocumentImportId, iterate_batch, wait_for_semaphore
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


async def document_passages_in_vespa(
    document_id: DocumentImportId, vespa_connection_pool: VespaAsync
) -> int:
    """Check how many passages we have in vespa relating to a document id."""

    passages_generator = get_document_passages_from_vespa__generator(
        document_import_id=document_id,
        vespa_connection_pool=vespa_connection_pool,
    )

    passages_in_vespa: set[TextBlockId] = set()
    async for passage_batch in passages_generator:
        passages_in_vespa.update(passage_batch.keys())

    return len(passages_in_vespa)


async def check_document_passages(
    document_id: DocumentImportId,
    bucket_name: str,
    s3_prefix: str,
    vespa_connection_pool: VespaAsync,
) -> Result:
    """Check if the number of passages in S3 matches those in Vespa for a document."""

    s3_passages_count: int = document_passages_in_s3(
        bucket_name=bucket_name, s3_key=os.path.join(s3_prefix, f"{document_id}.json")
    )

    vespa_passages_count: int = await document_passages_in_vespa(
        document_id=document_id, vespa_connection_pool=vespa_connection_pool
    )

    return Result(
        document_id=document_id,
        passages_mismatch=(vespa_passages_count != s3_passages_count),
        passages_count_in_vespa=vespa_passages_count,
        passages_count_in_s3=s3_passages_count,
    )


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
    max_concurrent_tasks: int = typer.Argument(
        default=10,
        help="Maximum number of concurrent tasks to run when checking documents",
    ),
    batch_size: int = typer.Argument(
        default=500,
        help="Size of each batch of document ids to process concurrently",
    ),
) -> None:
    """
    Compare documents in an s3 prefix against the relating data in the vespa database.

    The environent to run against (prod, sandbox, etc.) is determined by the vespa
    instance url that is set as well as the aws environment that your terminal is
    authenticated against.
    """

    typer.echo(
        f"Running with the following parameters:\n"
        f"Bucket Name: {bucket_name}\n"
        f"S3 Prefix: {s3_prefix}\n"
        f"Vespa Cert Directory: {vespa_cert_directory}\n"
        f"Report File Path: {report_file_path}\n"
        f"Max Concurrent Tasks: {max_concurrent_tasks}\n"
        f"Batch Size: {batch_size}\n"
    )

    vespa = VespaSearchAdapter(
        instance_url=VESPA_INSTANCE_URL, cert_directory=vespa_cert_directory
    )
    async with vespa.client.asyncio() as vespa_connection_pool:
        file_names: list[str] = collect_file_names(bucket_name, s3_prefix)
        document_ids: set[DocumentImportId] = set(
            DocumentImportId(Path(file).stem)
            for file in file_names
            if file.endswith(".json")
        )
        typer.echo(f"Found {len(document_ids)} document ids in s3 prefix {s3_prefix}")

        batches = iterate_batch(list(document_ids), batch_size=batch_size)

        for i, batch in enumerate(batches):
            start_time = time.time()
            typer.echo(
                f"Checking s3 against state in vespa for each document id in batch {i}..."
            )
            missing_passage_documents = []

            semaphore = asyncio.Semaphore(max_concurrent_tasks)
            tasks = [
                wait_for_semaphore(
                    semaphore,
                    check_document_passages(
                        document_id=document_id,
                        bucket_name=bucket_name,
                        s3_prefix=s3_prefix,
                        vespa_connection_pool=vespa_connection_pool,
                    ),
                )
                for document_id in batch
            ]
            results: list[Result | BaseException] = await asyncio.gather(
                *tasks, return_exceptions=True
            )
            for result in results:
                if isinstance(result, BaseException):
                    typer.echo(f"Error processing document: {result}")
                    continue
                if result.passages_mismatch:
                    missing_passage_documents.append(result)
            typer.echo(
                f"Found {len(missing_passage_documents)} documents with mismatching passages in batch {i}."
            )

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
            stem_root = Path(report_file_path).stem
            missing_df.to_csv(
                Path(report_file_path).with_name(stem_root + f"_batch_{i}"), index=False
            )
            typer.echo(f"Results written to {report_file_path}")
            elapsed_time = time.time() - start_time
            typer.echo(
                f"Batch {i} completed in {elapsed_time:.2f} seconds. "
                f"Processed {len(batch)} documents."
            )


if __name__ == "__main__":
    app()
