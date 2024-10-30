from collections import Counter
from typing import Annotated

import boto3
import typer
from cpr_sdk.parser_models import BaseParserOutput
from keybert import KeyBERT
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer()


def stringify(text: list[str]) -> str:
    return " ".join([line.strip() for line in text])


def get_document_text_blocks(doc_id: str) -> list[str]:
    region = "eu-west-1"
    bucket_name = "cpr-prod-data-pipeline-cache"
    file_name = f"{doc_id}.json"
    s3_file_path = f"embeddings_input/{file_name}"

    console.log(f"Downloading {s3_file_path} from {bucket_name}")

    s3 = boto3.client("s3", region_name=region)
    response = s3.get_object(Bucket=bucket_name, Key=s3_file_path)
    content = response["Body"].read().decode("utf-8")
    document = BaseParserOutput.model_validate_json(content)

    match document.document_content_type:
        case "application/pdf":
            text_blocks = [stringify(t.text) for t in document.pdf_data.text_blocks]
        case "text/html":
            text_blocks = [stringify(t.text) for t in document.html_data.text_blocks]
        case _:
            raise ValueError(
                f"Invalid document content type: {document.document_content_type}, for "
                f"document: {document.document_id}"
            )
    return text_blocks


def extract_keywords(text_blocks: list[str]) -> list[tuple[str, float]]:
    document_keywords = []
    model = KeyBERT()  # https://maartengr.github.io/KeyBERT/api/keybert.html

    for i, block in enumerate(text_blocks):
        console.log(f"Running on block index {i}/{len(text_blocks)}")

        block_keywords = model.extract_keywords(
            block,
            keyphrase_ngram_range=(1, 2),  # Consider substrings of one or two words
            top_n=15,
            seed_keywords=None,
        )
        document_keywords.extend(block_keywords)
    return document_keywords


def identify_top_key_phrases(
    key_phrases: list[tuple[str, float]],
) -> list[tuple[str, int]]:
    """
    Find the most popular keyphrases in the whole document

    Each keyword/phrase has a score of how relevant it is to a passage,
    but here we just count the most frequent occuring across all passages
    """
    phrases = [p[0] for p in key_phrases]
    count = Counter(phrases)
    return count.most_common(15)


@app.command()
def main(
    docid: Annotated[
        str,
        typer.Option(help="The document id of a parser output"),
    ],
):
    console.log(f"Extracting keywords from {docid}")
    text_blocks = get_document_text_blocks(docid)
    key_phrases = extract_keywords(text_blocks)
    top_key_phrases = identify_top_key_phrases(key_phrases)

    # Display results
    table = Table("Keyphrase", "Count")
    for row in top_key_phrases:
        table.add_row(row[0], str(row[1]))
    console.log(table)


if __name__ == "__main__":
    app()
