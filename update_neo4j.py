import json
import logging
import time
from datetime import datetime
from typing import Annotated

import boto3
import typer
from botocore.exceptions import ClientError
from neo4j.exceptions import DatabaseError
from neomodel.sync_.core import Database
from rich.logging import RichHandler

from knowledge_graph.graph_models import get_neo4j_session
from knowledge_graph.wikibase import WikibaseSession

logger = logging.getLogger(__name__)
logger.addHandler(RichHandler())
logger.setLevel(logging.INFO)

app = typer.Typer()


def _check_storage_usage(db: Database):
    """
    Check database storage usage if possible (works for some Neo4j versions).

    :param db: Database connection
    :returns: Storage info dict or None if not available
    """
    try:
        # Try to get storage information (may not work on all Neo4j versions/configurations)
        result, _ = db.cypher_query(
            """
            CALL dbms.queryJmx("org.neo4j:instance=kernel#0,name=Store file sizes")
            YIELD attributes
            RETURN attributes
            """
        )
        if result:
            logger.info("Storage information retrieved")
            return result
    except Exception:
        # Storage query not available or not supported
        pass

    # Try alternative: count nodes/relationships as a rough indicator
    try:
        node_count_result, _ = db.cypher_query("MATCH (n) RETURN count(n) as count")
        rel_count_result, _ = db.cypher_query(
            "MATCH ()-[r]->() RETURN count(r) as count"
        )

        node_count = node_count_result[0][0] if node_count_result else 0
        rel_count = rel_count_result[0][0] if rel_count_result else 0

        logger.info(
            f"Database contains approximately {node_count:,} nodes and {rel_count:,} relationships"
        )
        return {"nodes": node_count, "relationships": rel_count}
    except Exception:
        pass

    return None


def _check_database_writable(db: Database):
    """
    Check if the database is writable by attempting a simple write operation.

    :param db: Database connection
    :raises DatabaseError: If the database is in read-only mode
    """
    try:
        # Try to create a temporary node to test write access
        db.cypher_query(
            """
            CREATE (t:__WriteTest__ {test: true})
            WITH t
            DELETE t
            RETURN count(t) as deleted
            """
        )
        logger.debug("Database is writable")
    except DatabaseError as e:
        error_message = str(e)
        if (
            "read-only" in error_message.lower()
            or "WriteOnReadOnlyAccessDbException" in error_message
        ):
            # Try to get storage info before showing error
            _check_storage_usage(db)

            logger.error(
                "❌ Database is in READ-ONLY mode. Cannot perform write operations.\n\n"
                "⚠️  MOST COMMON CAUSE: Neo4j Aura storage is at 100% capacity.\n"
                "   When Aura reaches 100% disk storage, it automatically switches to\n"
                "   read-only mode to protect data integrity.\n\n"
                "Solutions:\n"
                "  1. Check your Aura console for storage usage\n"
                "  2. Delete unnecessary data to free up space\n"
                "  3. Upgrade your Aura plan for more storage\n\n"
                "Other possible causes:\n"
                "  • Connected to a read replica/secondary database\n"
                "  • Database user has read-only permissions\n"
                "  • Database is configured as read-only (dbms.read_only=true)\n"
                "  • Database is in maintenance mode\n\n"
                "Please check your NEO4J_CONNECTION_URI and Aura storage dashboard."
            )
            raise
        else:
            # Re-raise if it's a different database error
            raise


def _delete_nodes_in_batches(
    db: Database, node_label: str, batch_size: int = 1000, delay: float = 0.1
):
    """
    Delete all nodes of a given label in small batches to avoid memory issues.

    :param db: Database connection
    :param node_label: Label of nodes to delete (e.g., 'PassageNode', 'DocumentNode')
    :param batch_size: Number of nodes to delete per batch
    :param delay: Delay in seconds between batches to avoid overwhelming the database
    """
    deleted_count = 0
    while True:
        # Delete a batch and return how many were deleted
        result, _ = db.cypher_query(
            f"""
            MATCH (n:{node_label})
            WITH n LIMIT $batch_size
            DETACH DELETE n
            RETURN count(n) as deleted
            """,
            {"batch_size": batch_size},
        )

        batch_deleted = result[0][0] if result else 0
        deleted_count += batch_deleted

        if batch_deleted == 0:
            # No more nodes to delete
            break

        # Add delay between batches to avoid overwhelming the database
        if delay > 0:
            time.sleep(delay)

        if deleted_count % 10000 == 0:
            logger.info(f"Deleted {deleted_count} {node_label} nodes so far...")

    logger.info(f"Deleted {deleted_count} {node_label} nodes total")


def update_concepts(db: Database, clear_concepts: bool = False):
    """
    Add concepts to the knowledge graph in neo4j using batch operations.

    :clear_concepts: If True, clear the concept nodes from the database before adding new ones.
    """
    _check_database_writable(db)

    if clear_concepts:
        logger.info(
            "Clearing concept nodes and their relationships from the database..."
        )
        _delete_nodes_in_batches(db, "ConceptNode", batch_size=1000)

    wikibase = WikibaseSession()
    logger.info("Fetching concepts from Wikibase...")
    all_concepts = wikibase.get_concepts()
    logger.info(f"Found {len(all_concepts)} concepts")

    # Batch create concept nodes using UNWIND
    logger.info("Adding concept nodes to Neo4j in batch...")
    concept_data = [
        {
            "wikibase_id": str(concept.wikibase_id),
            "preferred_label": concept.preferred_label,
            "alternative_labels": [str(label) for label in concept.alternative_labels],
            "negative_labels": [str(label) for label in concept.negative_labels],
        }
        for concept in all_concepts
    ]

    db.cypher_query(
        """
        UNWIND $concepts AS concept
        MERGE (c:ConceptNode {wikibase_id: concept.wikibase_id})
        SET c.preferred_label = concept.preferred_label,
            c.alternative_labels = concept.alternative_labels,
            c.negative_labels = concept.negative_labels
        """,
        {"concepts": concept_data},
    )
    logger.info(f"Added {len(concept_data)} concept nodes")

    # Batch create relationships using UNWIND
    logger.info("Creating concept relationships in batch...")
    relationship_data = []
    for concept in all_concepts:
        related_concept_wikibase_ids = [
            str(wikibase._resolve_redirect(related_concept_id))
            for related_concept_id in concept.related_concepts
        ]
        for related_id in related_concept_wikibase_ids:
            relationship_data.append(
                {"from_id": str(concept.wikibase_id), "to_id": related_id}
            )

    if relationship_data:
        db.cypher_query(
            """
            UNWIND $relationships AS rel
            MATCH (from:ConceptNode {wikibase_id: rel.from_id})
            MATCH (to:ConceptNode {wikibase_id: rel.to_id})
            MERGE (from)-[:RELATED_TO]->(to)
            """,
            {"relationships": relationship_data},
        )
        logger.info(f"Created {len(relationship_data)} concept relationships")


def update_documents(
    db: Database,
    clear_documents: bool = False,
    batch_size: int = 50,
    batch_delay: float = 0.5,
):
    """
    Add documents to the knowledge graph in neo4j using batch operations.

    :clear_documents: If True, clear the document nodes from the database before adding new ones.
    :batch_size: Number of documents to process in each batch (reduced from 100 to avoid overwhelming DB).
    :batch_delay: Delay in seconds between batches to avoid overwhelming the database.
    """
    _check_database_writable(db)

    logger.info(
        f"Processing documents with batch_size={batch_size}, "
        f"batch_delay={batch_delay}s to avoid overwhelming the database"
    )

    if clear_documents:
        logger.info(
            "Clearing document nodes and their relationships from the database..."
        )
        # Delete passages first (they reference documents)
        logger.info("Deleting PassageNodes in batches...")
        _delete_nodes_in_batches(db, "PassageNode", batch_size=1000)

        # Then delete documents
        logger.info("Deleting DocumentNodes in batches...")
        _delete_nodes_in_batches(db, "DocumentNode", batch_size=1000)

    logger.info("Updating documents...")
    session = boto3.Session(region_name="eu-west-1", profile_name="prod")
    s3 = session.client("s3")
    bucket_name = "cpr-prod-data-pipeline-cache"

    # Use paginator to handle large numbers of objects
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket_name, Prefix="inference_results/latest/")

    # Collect batch data
    document_batch = []
    passage_batch = []
    doc_passage_relationships = []
    passage_concept_relationships = []

    batch_count = 0
    total_files_processed = 0

    for page in pages:
        if "Contents" not in page:
            continue
        for obj in page["Contents"]:
            total_files_processed += 1

            # Skip very small files (likely empty placeholders)
            if obj.get("Size", 0) < 10:
                logger.warning(
                    f"Skipping small file: {obj['Key']} (Size: {obj.get('Size', 0)})"
                )
                continue

            document_id_key = obj["Key"]

            # Strip the prefix to get just the filename
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

            # Parse the publication timestamp - use .get() for safety
            document_metadata = document_data_json.get("document_metadata", {})
            publication_ts_str = document_metadata.get("publication_ts")
            if publication_ts_str and publication_ts_str.endswith("Z"):
                publication_ts_str = publication_ts_str.replace("Z", "+00:00")
            publication_ts = (
                datetime.fromisoformat(publication_ts_str)
                if publication_ts_str
                else None
            )

            # Add document data to batch - use .get() for safety
            document_id = document_data_json.get("document_id")
            if not document_id:
                logger.warning("Document missing document_id, skipping")
                continue

            document_batch.append(
                {
                    "document_id": document_id,
                    "title": document_metadata.get("document_title", ""),
                    "document_slug": document_metadata.get("slug", ""),
                    "family_id": document_metadata.get("family_import_id"),
                    "family_slug": document_metadata.get("family_slug", ""),
                    "publication_ts": publication_ts.isoformat()
                    if publication_ts
                    else None,
                    "translated": document_data_json.get("translated", False),
                    "geography_ids": document_metadata.get("geographies", []),
                    "corpus_id": document_metadata.get("corpus_import_id"),
                }
            )
            logger.info(f"Processing document {document_id}")

            # Process passages - check for both pdf_data and html_data
            content_data = document_data_json.get("pdf_data") or document_data_json.get(
                "html_data"
            )

            if not content_data or not content_data.get("text_blocks"):
                logger.warning(
                    f"No pdf_data/html_data or text_blocks found for document "
                    f"{document_id}, skipping passages"
                )
            else:
                for (
                    text_block_id,
                    identified_concepts,
                ) in aggregated_inference_json.items():
                    # Skip passages with no concept mentions
                    if not identified_concepts:
                        continue

                    # Find the matching text block
                    text_block_data = None
                    for block in content_data["text_blocks"]:
                        if block.get("text_block_id") == text_block_id:
                            text_block_data = block
                            break

                    if not text_block_data:
                        logger.warning(
                            f"Text block {text_block_id} not found for document "
                            f"{document_id}"
                        )
                        continue

                    # Join text array into a single string
                    text_field = text_block_data.get("text", "")
                    text_content = (
                        " ".join(text_field)
                        if isinstance(text_field, list)
                        else text_field
                    )

                    passage_id = f"{document_id}_{text_block_id}"

                    # Add passage data to batch - use .get() with defaults for safety
                    passage_batch.append(
                        {
                            "document_passage_id": passage_id,
                            "text": text_content,
                            "text_block_id": text_block_data.get(
                                "text_block_id", text_block_id
                            ),
                            "text_block_language": text_block_data.get(
                                "language", "unknown"
                            ),
                            "text_block_type": text_block_data.get("type", "Text"),
                            "page_number": text_block_data.get("page_number"),
                        }
                    )

                    # Add document-passage relationship
                    doc_passage_relationships.append(
                        {
                            "document_id": document_id,
                            "passage_id": passage_id,
                        }
                    )

                    # Add passage-concept relationships
                    for concept_data in identified_concepts:
                        passage_concept_relationships.append(
                            {
                                "passage_id": passage_id,
                                "wikibase_id": str(concept_data["id"]),
                            }
                        )

            # Process batch when it reaches the batch size
            if len(document_batch) >= batch_size:
                _write_document_batch(
                    db,
                    document_batch,
                    passage_batch,
                    doc_passage_relationships,
                    passage_concept_relationships,
                    delay=batch_delay,
                )
                batch_count += len(document_batch)
                logger.info(
                    f"Processed {batch_count} documents so far "
                    f"({total_files_processed} files checked)..."
                )

                # Clear batches
                document_batch = []
                passage_batch = []
                doc_passage_relationships = []
                passage_concept_relationships = []

    # Process remaining items in batch
    if document_batch:
        _write_document_batch(
            db,
            document_batch,
            passage_batch,
            doc_passage_relationships,
            passage_concept_relationships,
            delay=batch_delay,
        )
        batch_count += len(document_batch)

    logger.info(
        f"Completed processing {batch_count} documents "
        f"(checked {total_files_processed} files total)"
    )


def _write_document_batch(
    db: Database,
    documents: list,
    passages: list,
    doc_passage_rels: list,
    passage_concept_rels: list,
    delay: float = 0.5,
):
    """
    Write a batch of documents, passages, and relationships to Neo4j.

    :param delay: Delay in seconds after writing the batch to avoid overwhelming the database
    """

    # Create document nodes
    if documents:
        db.cypher_query(
            """
            UNWIND $documents AS doc
            MERGE (d:DocumentNode {document_id: doc.document_id})
            SET d.title = doc.title,
                d.document_slug = doc.document_slug,
                d.family_id = doc.family_id,
                d.family_slug = doc.family_slug,
                d.publication_ts = CASE
                    WHEN doc.publication_ts IS NOT NULL
                    THEN datetime(doc.publication_ts)
                    ELSE NULL
                END,
                d.translated = doc.translated,
                d.geography_ids = doc.geography_ids,
                d.corpus_id = doc.corpus_id
            """,
            {"documents": documents},
        )

    # Create passage nodes
    if passages:
        db.cypher_query(
            """
            UNWIND $passages AS passage
            MERGE (p:PassageNode {document_passage_id: passage.document_passage_id})
            SET p.text = passage.text,
                p.text_block_id = passage.text_block_id,
                p.text_block_language = passage.text_block_language,
                p.text_block_type = passage.text_block_type,
                p.page_number = passage.page_number
            """,
            {"passages": passages},
        )

    # Create document-passage relationships
    if doc_passage_rels:
        db.cypher_query(
            """
            UNWIND $rels AS rel
            MATCH (d:DocumentNode {document_id: rel.document_id})
            MATCH (p:PassageNode {document_passage_id: rel.passage_id})
            MERGE (d)-[:HAS_PASSAGE]->(p)
            """,
            {"rels": doc_passage_rels},
        )

    # Create passage-concept relationships
    if passage_concept_rels:
        db.cypher_query(
            """
            UNWIND $rels AS rel
            MATCH (p:PassageNode {document_passage_id: rel.passage_id})
            MATCH (c:ConceptNode {wikibase_id: rel.wikibase_id})
            MERGE (p)-[:MENTIONS_CONCEPT]->(c)
            """,
            {"rels": passage_concept_rels},
        )

    # Add delay after writing batch to avoid overwhelming the database
    if delay > 0:
        time.sleep(delay)


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
