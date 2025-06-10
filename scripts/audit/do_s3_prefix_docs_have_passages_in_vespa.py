import asyncio
import os
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
from flows.utils import DocumentImportId
from scripts.audit.do_classifier_specs_have_results import collect_file_names
from scripts.cloud import AwsEnv

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

    # It might be overkill to parse the data into the model but we can rely on the
    # text blocks property to count.
    parser_output = ParserOutput.model_validate(data)

    return len(parser_output.text_blocks)


async def document_passages_in_vespa(
    document_id: DocumentImportId, vespa_connection_pool: VespaAsync
) -> int:
    """Check what data we have in vespa relating to a document id."""

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
    # No. of passages in s3 object
    s3_passages_count: int = document_passages_in_s3(
        bucket_name=bucket_name, s3_key=os.path.join(s3_prefix, f"{document_id}.json")
    )

    # No. of passages in vespa
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
async def check_s3_prefix_documents_match_vespa(
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
    s3_prefix: str = typer.Argument(
        help="The S3 prefix to check for documents, e.g., 'indexer_input"
    ),
    vespa_cert_directory: str = typer.Argument(
        default=None,
        help="Directory containing Vespa certificates for secure connection",
    ),
) -> None:
    """Compare documents in an s3 prefix against the relating data in the vespa database."""

    vespa = VespaSearchAdapter(
        instance_url=VESPA_INSTANCE_URL, cert_directory=vespa_cert_directory
    )
    async with vespa.client.asyncio() as vespa_connection_pool:
        typer.echo(f"Getting document ids from {aws_env} {bucket_name}/{s3_prefix}")
        file_names: list[str] = collect_file_names(bucket_name, s3_prefix)
        document_ids: set[DocumentImportId] = set(
            DocumentImportId(Path(file).stem)
            for file in file_names
            if file.endswith(".json")
        )
        typer.echo(f"Found {len(document_ids)} document ids in s3 prefix {s3_prefix}")

        typer.echo("Checking s3 against state in vespa for each document id...")
        missing_passage_documents = []
        for document_id in document_ids:
            result = await check_document_passages(
                document_id=document_id,
                bucket_name=bucket_name,
                s3_prefix=s3_prefix,
                vespa_connection_pool=vespa_connection_pool,
            )
            if result.passages_mismatch:
                typer.echo(
                    f"Document {document_id} has a mismatch: "
                    f"{result.passages_count_in_s3} passages in S3, "
                    f"{result.passages_count_in_vespa} passages in Vespa."
                )
                missing_passage_documents.append(result)

        typer.echo(
            f"Found {len(missing_passage_documents)} documents with no passages in Vespa."
        )
        # Convert results to DataFrame
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

        # Write to CSV
        csv_path = "missing_passages.csv"
        missing_df.to_csv(csv_path, index=False)
        typer.echo(f"Results written to {csv_path}")


if __name__ == "__main__":
    app()
