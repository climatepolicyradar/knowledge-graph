import json
import logging
from datetime import datetime
from typing import Annotated

import boto3
import typer
from botocore.exceptions import ClientError
from neomodel.sync_.core import Database
from rich.logging import RichHandler

from knowledge_graph.graph_models import (
    ConceptNode,
    DocumentNode,
    PassageNode,
    get_neo4j_session,
)
from knowledge_graph.wikibase import WikibaseSession

logger = logging.getLogger(__name__)
logger.addHandler(RichHandler())
logger.setLevel(logging.INFO)

app = typer.Typer()


def update_concepts(db: Database, clear_concepts: bool = False):
    """
    Add concepts to the knowledge graph in neo4j.

    :clear_concepts: If True, clear the concept nodes from the database before adding new ones.
    """
    if clear_concepts:
        logger.info(
            "Clearing concept nodes and their relationships from the database..."
        )
        db.cypher_query("MATCH (n:ConceptNode) DETACH DELETE n")

    wikibase = WikibaseSession()
    logger.info("Fetching concepts from Wikibase...")
    all_concepts = wikibase.get_concepts()
    logger.info(f"Found {len(all_concepts)} concepts")

    logger.info("Adding concept nodes to Neo4j...")
    for concept in all_concepts:
        concept_node = ConceptNode.from_concept(concept)
        concept_node.save()
        logger.info(f"Added concept node {concept}")

    logger.info("Updating concept relationships...")
    for concept in all_concepts:
        concept_node = ConceptNode.nodes.get_or_none(
            wikibase_id=str(concept.wikibase_id)
        )
        if not concept_node:
            logger.warning(f"Concept node {concept.wikibase_id} not found")
            continue
        related_concept_wikibase_ids = [
            wikibase._resolve_redirect(related_concept_id)
            for related_concept_id in concept.related_concepts
        ]
        for related_concept_wikibase_id in related_concept_wikibase_ids:
            related_concept_node = ConceptNode.nodes.get_or_none(
                wikibase_id=str(related_concept_wikibase_id)
            )
            if not related_concept_node:
                logger.warning(
                    f"Related concept {related_concept_wikibase_id} not found for concept {concept.wikibase_id}"
                )
                continue
            concept_node.related_to.connect(related_concept_node)


def update_documents(db: Database, clear_documents: bool = False):
    """
    Add documents to the knowledge graph in neo4j.

    :clear_documents: If True, clear the document nodes from the database before adding new ones.
    """
    if clear_documents:
        logger.info(
            "Clearing document nodes and their relationships from the database..."
        )
        db.cypher_query("MATCH (n:DocumentNode) DETACH DELETE n")
        db.cypher_query("MATCH (n:PassageNode) DETACH DELETE n")

    print("Updating documents...")
    session = boto3.Session(region_name="eu-west-1", profile_name="prod")
    s3 = session.client("s3")
    bucket_name = "cpr-prod-data-pipeline-cache"

    # Use paginator to handle large numbers of objects
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix="inference_results/latest/")

    for page in pages:
        if "Contents" not in page:
            continue
        for obj in page["Contents"]:
            # Skip very small files (likely empty placeholders)
            if obj.get("Size", 0) < 10:
                logger.warning(
                    f"Skipping small file: {obj['Key']} (Size: {obj.get('Size', 0)})"
                )
                continue

            document_id_key = obj["Key"]

            # Strip the prefix to get just the filename
            # e.g., "inference_results/latest/AF.document.002MMUCR.n0001.json"
            # -> "AF.document.002MMUCR.n0001.json"
            filename = document_id_key.replace("inference_results/latest/", "")

            try:
                aggregated_inference_object = s3.get_object(
                    Bucket="cpr-prod-data-pipeline-cache", Key=document_id_key
                )
                aggregated_inference_content = (
                    aggregated_inference_object["Body"].read().decode("utf-8")
                )
                aggregated_inference_json = json.loads(aggregated_inference_content)

                # Skip if the JSON is empty
                if not aggregated_inference_json:
                    logger.warning(f"Skipping empty inference file: {document_id_key}")
                    continue
            except Exception as e:
                logger.error(f"Error reading inference file {document_id_key}: {e}")
                continue

            try:
                document_data_object = s3.get_object(
                    Bucket=bucket_name, Key=f"embeddings_input/{filename}"
                )
                document_data_string = (
                    document_data_object["Body"].read().decode("utf-8")
                )
                document_data_json = json.loads(document_data_string)
            except ClientError as e:
                if e.response["Error"]["Code"] == "NoSuchKey":
                    logger.warning(f"Document data not found for {filename}, skipping")
                    continue
                else:
                    raise
            except Exception as e:
                logger.error(f"Error reading document data for {filename}: {e}")
                continue

            # Parse the publication timestamp from ISO format string to datetime
            publication_ts_str = document_data_json["document_metadata"][
                "publication_ts"
            ]
            # Handle ISO format with 'Z' (UTC) - replace Z with +00:00 for fromisoformat
            if publication_ts_str and publication_ts_str.endswith("Z"):
                publication_ts_str = publication_ts_str.replace("Z", "+00:00")
            publication_ts = (
                datetime.fromisoformat(publication_ts_str)
                if publication_ts_str
                else None
            )

            document_node = DocumentNode(
                document_id=document_data_json["document_id"],
                title=document_data_json["document_metadata"]["document_title"],
                document_slug=document_data_json["document_metadata"]["slug"],
                family_id=document_data_json["document_metadata"]["family_import_id"],
                family_slug=document_data_json["document_metadata"]["family_slug"],
                publication_ts=publication_ts,
                translated=document_data_json["translated"],
                geography_ids=document_data_json["document_metadata"]["geographies"],
                corpus_id=document_data_json["document_metadata"]["corpus_import_id"],
            )
            document_node.save()
            logger.info(f"Added document node {document_data_json['document_id']}")

            for text_block_id, identified_concepts in aggregated_inference_json.items():
                # Skip passages with no concept mentions
                if not identified_concepts:
                    continue

                # Find the matching text block in the list
                text_block_data = None
                for block in document_data_json["pdf_data"]["text_blocks"]:
                    if block["text_block_id"] == text_block_id:
                        text_block_data = block
                        break

                if not text_block_data:
                    logger.warning(
                        f"Text block {text_block_id} not found for document {document_data_json['document_id']}"
                    )
                    continue

                # Join text array into a single string
                text_content = (
                    " ".join(text_block_data["text"])
                    if isinstance(text_block_data["text"], list)
                    else text_block_data["text"]
                )

                # add the passage node to the document node
                passage_node = PassageNode(
                    document_passage_id=f"{document_data_json['document_id']}_{text_block_id}",
                    text=text_content,
                    text_block_id=text_block_data["text_block_id"],
                    text_block_language=text_block_data["language"],
                    text_block_type=text_block_data["type"],
                    page_number=text_block_data["page_number"],
                )
                passage_node.save()
                document_node.passages.connect(passage_node)
                logger.info(
                    f"Added passage node {passage_node.document_passage_id} to document {document_node.document_id}"
                )

                # relate the passage nodes to the concepts that they mention
                for concept_data in identified_concepts:
                    wikibase_id = concept_data["id"]
                    concept_node = ConceptNode.nodes.get_or_none(
                        wikibase_id=str(wikibase_id)
                    )
                    if not concept_node:
                        logger.warning(f"Concept node {wikibase_id} not found")
                        continue
                    passage_node.concepts.connect(concept_node)
                    logger.info(
                        f"Added concept node {wikibase_id} to passage {passage_node.document_passage_id}"
                    )


@app.command()
def main(
    update_concepts_flag: Annotated[
        bool,
        typer.Option(
            "--update-concepts", help="If True, update the concepts in the database"
        ),
    ] = False,
    update_documents_flag: Annotated[
        bool,
        typer.Option(
            "--update-documents", help="If True, update the documents in the database"
        ),
    ] = False,
    clear_concepts_flag: Annotated[
        bool,
        typer.Option(
            "--clear-concepts",
            help="Clear the concept nodes and their relationships from the database before adding new ones",
        ),
    ] = False,
    clear_documents_flag: Annotated[
        bool,
        typer.Option(
            "--clear-documents",
            help="Clear the document nodes and their relationships from the database before adding new ones",
        ),
    ] = False,
):
    """Update Neo4j with the latest version of the knowledge graph."""
    db = get_neo4j_session(clear=False)
    logger.info("Connected to Neo4j")

    if update_concepts_flag:
        update_concepts(db=db, clear_concepts=clear_concepts_flag)

    if update_documents_flag:
        update_documents(db=db, clear_documents=clear_documents_flag)


if __name__ == "__main__":
    app()
