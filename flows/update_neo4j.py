"""
Prefect flows to update our Neo4j db with the latest knowledge graph data.

This module contains two flows:
- `update_concepts`: Updates concept nodes and relationships from Wikibase
- `update_documents`: Updates document and passage nodes from S3

`deployments.py` manages a deployment for nightly runs of the `update_concepts` flow,
which keep the Neo4j database in sync with the concept store. The `update_documents`
flow is not deployed, but can be run manually from the command line.

See `knowledge_graph/graph_models.py` for the schema which describes the nodes, their
properties, and the relationships between them.
"""

import asyncio
import json
import os
import time
from datetime import datetime
from typing import Annotated, Any, Literal, Sequence

import boto3
import botocore.exceptions
import typer
from botocore.exceptions import ClientError
from cpr_sdk.ssm import get_aws_ssm_param
from neomodel import db
from prefect import flow

from flows.utils import get_logger, iterate_batch
from knowledge_graph.graph_models import get_neo4j_session
from knowledge_graph.wikibase import WikibaseSession

CONCEPT_BATCH_SIZE = 5000
RELATIONSHIP_BATCH_SIZE = 1000


def _setup_env_from_ssm() -> None:
    """
    Source required secrets from SSM

    Expected SSM parameter names:
    - /Neo4j/ConnectionURI -> NEO4J_CONNECTION_URI
    - /Wikibase/Cloud/ServiceAccount/Username -> WIKIBASE_USERNAME
    - /Wikibase/Cloud/ServiceAccount/Password -> WIKIBASE_PASSWORD
    - /Wikibase/Cloud/URL -> WIKIBASE_URL
    """
    logger = get_logger()

    def _set_env_var_from_ssm(ssm_name: str, env_var: str) -> None:
        try:
            value = get_aws_ssm_param(ssm_name)
            os.environ[env_var] = value
            logger.info("Loaded %s from SSM (%s)", env_var, ssm_name)
        except (
            botocore.exceptions.BotoCoreError,
            botocore.exceptions.ClientError,
            ValueError,
        ) as e:
            # Keep going; downstream code will fail fast if the secret is actually required
            logger.warning("Could not load %s from SSM %s: %s", env_var, ssm_name, e)

    _set_env_var_from_ssm(
        ssm_name="/Neo4j/ConnectionURI", env_var="NEO4J_CONNECTION_URI"
    )
    _set_env_var_from_ssm(
        ssm_name="/Wikibase/Cloud/ServiceAccount/Username", env_var="WIKIBASE_USERNAME"
    )
    _set_env_var_from_ssm(
        ssm_name="/Wikibase/Cloud/ServiceAccount/Password", env_var="WIKIBASE_PASSWORD"
    )
    _set_env_var_from_ssm(ssm_name="/Wikibase/Cloud/URL", env_var="WIKIBASE_URL")


def process_in_batches(
    items: Sequence[Any],
    batch_size: int,
    short_label_for_logging: str,
    process_fn,
) -> None:
    """Run a side-effecting operation over a sequence in batches."""
    logger = get_logger()
    if not items:
        logger.info(f"{short_label_for_logging}: nothing to process")
        return

    logger.info(
        f"{short_label_for_logging}: {len(items)} items in batches of {batch_size}"
    )
    for batch in iterate_batch(data=items, batch_size=batch_size):
        process_fn(batch)
    logger.info(f"{short_label_for_logging}: done ({len(items)})")


def get_existing_concept_ids_from_neo4j() -> set[str]:
    """Get the list of all wikibase_ids currently in the Neo4j database."""
    result = execute_cypher(
        "MATCH (c:ConceptNode) RETURN c.wikibase_id as wikibase_id",
        None,
        dry_run=False,
        return_results=True,
    )
    return {row[0] for row in result} if result else set()


def create_or_update_concept_nodes(
    batch: Sequence[dict[str, Any]], *, dry_run: bool
) -> None:
    """Upsert ConceptNode nodes using `wikibase_id` as the stable key."""
    execute_cypher(
        """
        UNWIND $batch AS concept
        MERGE (c:ConceptNode {wikibase_id: concept.wikibase_id})
        SET c.preferred_label = concept.preferred_label,
            c.description = concept.description,
            c.definition = concept.definition,
            c.alternative_labels = concept.alternative_labels,
            c.negative_labels = concept.negative_labels
        """,
        {"batch": list(batch)},
        dry_run=dry_run,
    )


def delete_concept_relationships(*, dry_run: bool) -> None:
    """
    Remove only concept-to-concept edges prior to rebuild.

    Specifically deletes:
    - `(:ConceptNode)-[:SUBCONCEPT_OF]->(:ConceptNode)`
    - `(:ConceptNode)-[:RELATED_TO]->(:ConceptNode)`

    Document-related nodes/edges are untouched.
    """
    logger = get_logger()
    logger.info("Deleting existing concept-to-concept relationships...")

    # Delete SUBCONCEPT_OF relationships
    execute_cypher(
        "MATCH ()-[r:SUBCONCEPT_OF]->() DELETE r",
        None,
        dry_run=dry_run,
    )

    # Delete RELATED_TO relationships
    execute_cypher(
        "MATCH ()-[r:RELATED_TO]->() DELETE r",
        None,
        dry_run=dry_run,
    )
    logger.info("Deleted RELATED_TO and SUBCONCEPT_OF relationships")


def _create_relationships(
    batch: Sequence[dict[str, str]],
    relationship_type: Literal["RELATED_TO", "SUBCONCEPT_OF"],
    dry_run: bool,
) -> None:
    """
    Create directed relationships of a specified type between ConceptNodes.

    Each relationship in the batch should have 'from_id' and 'to_id' keys specifying
    the source and target concept IDs respectively.

    :param batch: Sequence of relationship dictionaries, each containing 'from_id' and 'to_id'
    :param relationship_type: The type of relationship to create (e.g., 'RELATED_TO', 'SUBCONCEPT_OF')
    :param dry_run: If True, skip actual writes to Neo4j
    """
    cypher_query = f"""
    UNWIND $batch AS rel
    MATCH (from:ConceptNode {{wikibase_id: rel.from_id}})
    MATCH (to:ConceptNode {{wikibase_id: rel.to_id}})
    MERGE (from)-[:{relationship_type}]->(to)
    """
    execute_cypher(cypher_query, {"batch": list(batch)}, dry_run=dry_run)


def execute_cypher(
    query: str, params: dict | None, *, dry_run: bool, return_results: bool = False
) -> list | None:
    """
    Run a Cypher statement against Neo4j, or log it when in dry-run mode.

    :param query: The Cypher query to execute
    :param params: Optional parameters for the query
    :param dry_run: If True, skip actual execution and log the query
    :param return_results: If True, return the query results (first element of the tuple)
    :return: Query results if return_results=True, otherwise None
    """
    logger = get_logger()
    if dry_run:
        logger.info("DRY RUN — skipping Cypher execution: %s", " ".join(query.split()))
        return None if not return_results else []
    if params is None:
        result, _ = db.cypher_query(query)
    else:
        result, _ = db.cypher_query(query, params)
    return result if return_results else None


def _delete_nodes_in_batches(
    node_label: str, batch_size: int = 1000, delay: float = 0.1, *, dry_run: bool
) -> None:
    """
    Delete all nodes with a given label.

    Deletion is batched to avoid memory issues on the neo4j database.

    :param node_label: Label of nodes to delete (e.g., 'PassageNode', 'DocumentNode')
    :param batch_size: Number of nodes to delete per batch
    :param delay: Delay in seconds between batches to avoid overwhelming the database
    :param dry_run: If True, skip actual deletion
    """
    logger = get_logger()
    deleted_count = 0
    while True:
        # Delete a batch and return how many were deleted
        query = f"""
            MATCH (n:{node_label})
            WITH n LIMIT $batch_size
            DETACH DELETE n
            RETURN count(n) as deleted
        """
        result = execute_cypher(
            query, {"batch_size": batch_size}, dry_run=dry_run, return_results=True
        )
        if dry_run:
            logger.info(f"DRY RUN — would delete batch of {node_label} nodes")
            break

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


def _write_document_batch(
    documents: list,
    passages: list,
    doc_passage_rels: list,
    passage_concept_rels: list,
    delay: float = 0.5,
    *,
    dry_run: bool,
) -> None:
    """
    Write a batch of documents, passages, and relationships to Neo4j.

    :param delay: Delay in seconds after writing the batch to avoid overwhelming the database
    :param dry_run: If True, skip actual writes
    """
    # Create document nodes
    if documents:
        execute_cypher(
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
            dry_run=dry_run,
        )

    # Create passage nodes
    if passages:
        execute_cypher(
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
            dry_run=dry_run,
        )

    # Create document-passage relationships
    if doc_passage_rels:
        execute_cypher(
            """
            UNWIND $rels AS rel
            MATCH (d:DocumentNode {document_id: rel.document_id})
            MATCH (p:PassageNode {document_passage_id: rel.passage_id})
            MERGE (d)-[:HAS_PASSAGE]->(p)
            """,
            {"rels": doc_passage_rels},
            dry_run=dry_run,
        )

    # Create passage-concept relationships
    if passage_concept_rels:
        execute_cypher(
            """
            UNWIND $rels AS rel
            MATCH (p:PassageNode {document_passage_id: rel.passage_id})
            MATCH (c:ConceptNode {wikibase_id: rel.wikibase_id})
            MERGE (p)-[:MENTIONS_CONCEPT]->(c)
            """,
            {"rels": passage_concept_rels},
            dry_run=dry_run,
        )

    # Add delay after writing batch to avoid overwhelming the database
    if delay > 0 and not dry_run:
        time.sleep(delay)


@flow()
async def update_concepts(*, dry_run: bool = False) -> None:
    """Synchronise Neo4j with the concept graph from Wikibase"""

    logger = get_logger()
    logger.info("Starting concept graph update")

    # Ensure required secrets are set before establishing connections to Neo4j and Wikibase
    _setup_env_from_ssm()

    # Connect to Neo4j
    get_neo4j_session(clear=False)
    logger.info("Connected to Neo4j")

    existing_concept_ids = get_existing_concept_ids_from_neo4j()
    logger.info("Existing concepts in Neo4j: %s", len(existing_concept_ids))

    # Connect to Wikibase and fetch all concepts
    async with WikibaseSession() as wikibase:
        logger.info("Fetching all concepts from Wikibase...")
        all_concepts = await wikibase.get_concepts_async()
        logger.info("Fetched %d concepts from Wikibase", len(all_concepts))

        concepts_data = [
            {
                "wikibase_id": str(c.wikibase_id),
                "preferred_label": c.preferred_label,
                "description": c.description,
                "definition": c.definition,
                "alternative_labels": [str(label) for label in c.alternative_labels],
                "negative_labels": [str(label) for label in c.negative_labels],
            }
            for c in all_concepts
        ]

        # Create or update concept nodes
        process_in_batches(
            concepts_data,
            CONCEPT_BATCH_SIZE,
            "Upserting concept nodes...",
            lambda batch: create_or_update_concept_nodes(batch, dry_run=dry_run),
        )

        new_concepts = len(
            [c for c in all_concepts if str(c.wikibase_id) not in existing_concept_ids]
        )
        updated_concepts = len(
            [c for c in all_concepts if str(c.wikibase_id) in existing_concept_ids]
        )
        logger.info(
            "Concept nodes upserted — new: %s, updated: %s",
            new_concepts,
            updated_concepts,
        )

        # Find all related concept IDs that might need nodes created
        all_known_ids = {str(c.wikibase_id) for c in all_concepts}
        newly_discovered_ids: set[str] = set()

        logger.info("Scanning for referenced-but-missing concept nodes...")
        for concept in all_concepts:
            for related_id in concept.related_concepts:
                if str(related_id) not in all_known_ids:
                    newly_discovered_ids.add(str(related_id))
            for parent_id in concept.subconcept_of:
                if str(parent_id) not in all_known_ids:
                    newly_discovered_ids.add(str(parent_id))
            for child_id in concept.has_subconcept:
                if str(child_id) not in all_known_ids:
                    newly_discovered_ids.add(str(child_id))

        # Create any missing concept nodes (for concepts referenced but not in main list)
        if newly_discovered_ids:
            missing_concepts = [
                {
                    "wikibase_id": id,
                    "preferred_label": f"Concept {id}",
                    "alternative_labels": [],
                    "negative_labels": [],
                }
                for id in newly_discovered_ids
            ]

            process_in_batches(
                missing_concepts,
                CONCEPT_BATCH_SIZE,
                f"Creating missing concept nodes ({len(missing_concepts)})",
                lambda batch: create_or_update_concept_nodes(batch, dry_run=dry_run),
            )
            logger.info("Missing concept nodes created: %s", len(missing_concepts))

        # Delete all existing concept-to-concept relationships (only)
        delete_concept_relationships(dry_run=dry_run)

        # Prepare all relationships
        all_relationships: dict[str, list[dict[str, str]]] = {
            "related_to": [],
            "subconcept_of": [],
            "has_subconcept": [],
        }
        logger.info("Preparing concept-to-concept relationships...")
        for concept in all_concepts:
            for related_id in concept.related_concepts:
                all_relationships["related_to"].append(
                    {"from_id": str(concept.wikibase_id), "to_id": str(related_id)}
                )
            for parent_id in concept.subconcept_of:
                all_relationships["subconcept_of"].append(
                    {"from_id": str(concept.wikibase_id), "to_id": str(parent_id)}
                )
            for child_id in concept.has_subconcept:
                # Swap IDs so relationship goes from child -> parent (same direction as subconcept_of)
                all_relationships["has_subconcept"].append(
                    {"from_id": str(child_id), "to_id": str(concept.wikibase_id)}
                )

        # Create new relationships
        if all_relationships["related_to"]:
            process_in_batches(
                all_relationships["related_to"],
                RELATIONSHIP_BATCH_SIZE,
                f"Creating RELATED_TO relationships ({len(all_relationships['related_to'])})",
                lambda batch: _create_relationships(
                    batch, relationship_type="RELATED_TO", dry_run=dry_run
                ),
            )
            logger.info(
                "Relationships created — RELATED_TO: %s",
                len(all_relationships["related_to"]),
            )

        if all_relationships["subconcept_of"]:
            process_in_batches(
                all_relationships["subconcept_of"],
                RELATIONSHIP_BATCH_SIZE,
                f"Creating SUBCONCEPT_OF relationships ({len(all_relationships['subconcept_of'])})",
                lambda batch: _create_relationships(
                    batch, relationship_type="SUBCONCEPT_OF", dry_run=dry_run
                ),
            )
            logger.info(
                "Relationships created — SUBCONCEPT_OF: %s",
                len(all_relationships["subconcept_of"]),
            )

        if all_relationships["has_subconcept"]:
            process_in_batches(
                all_relationships["has_subconcept"],
                RELATIONSHIP_BATCH_SIZE,
                f"Creating HAS_SUBCONCEPT relationships ({len(all_relationships['has_subconcept'])})",
                lambda batch: _create_relationships(
                    batch, relationship_type="SUBCONCEPT_OF", dry_run=dry_run
                ),
            )
            logger.info(
                "Relationships created — HAS_SUBCONCEPT: %s",
                len(all_relationships["has_subconcept"]),
            )

        # Give the user a summary of the update
        total_relationships = sum(len(rels) for rels in all_relationships.values())
        logger.info("Concept graph update complete")
        logger.info(
            "Summary — new concepts: %s, updated concepts: %s, missing nodes created: %s, total relationships: %s",
            new_concepts,
            updated_concepts,
            len(newly_discovered_ids),
            total_relationships,
        )
        logger.info(
            "Breakdown — RELATED_TO: %s, SUBCONCEPT_OF: %s",
            len(all_relationships["related_to"]),
            len(all_relationships["subconcept_of"])
            + len(all_relationships["has_subconcept"]),
        )

        logger.info("Concept update finished")


@flow()
async def update_documents(
    *,
    clear_documents: bool = False,
    batch_size: int = 50,
    batch_delay: float = 0.5,
    dry_run: bool = False,
) -> None:
    """
    Add documents to the knowledge graph in neo4j using batch operations.

    Fetches documents directly from S3 with rich metadata including titles, slugs,
    publication timestamps, geography IDs, and corpus information.

    :param clear_documents: If True, clear the document nodes from the database before adding new ones.
    :param batch_size: Number of documents to process in each batch (reduced from 100 to avoid overwhelming DB).
    :param batch_delay: Delay in seconds between batches to avoid overwhelming the database.
    :param dry_run: If True, skip actual writes to Neo4j.
    """
    logger = get_logger()
    logger.info("Starting document update")

    # Ensure required secrets are set before establishing connections to Neo4j
    _setup_env_from_ssm()

    # Connect to Neo4j
    get_neo4j_session(clear=False)
    logger.info("Connected to Neo4j")

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
        _delete_nodes_in_batches("PassageNode", batch_size=1000, dry_run=dry_run)

        # Then delete documents
        logger.info("Deleting DocumentNodes in batches...")
        _delete_nodes_in_batches("DocumentNode", batch_size=1000, dry_run=dry_run)

    logger.info("Updating documents from S3...")
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
            document_id_key = obj.get("Key")
            if not document_id_key:
                continue

            if obj.get("Size", 0) < 10:
                logger.warning(
                    f"Skipping small file: {document_id_key} (Size: {obj.get('Size', 0)})"
                )
                continue

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
                error_code = e.response.get("Error", {}).get("Code", "")
                if error_code == "NoSuchKey":
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
                    document_batch,
                    passage_batch,
                    doc_passage_relationships,
                    passage_concept_relationships,
                    delay=batch_delay,
                    dry_run=dry_run,
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
            document_batch,
            passage_batch,
            doc_passage_relationships,
            passage_concept_relationships,
            delay=batch_delay,
            dry_run=dry_run,
        )
        batch_count += len(document_batch)

    logger.info(
        f"Completed processing {batch_count} documents "
        f"(checked {total_files_processed} files total)"
    )


app = typer.Typer()


@app.command()
def concepts(
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Run without making changes to Neo4j"),
    ] = False,
) -> None:
    """Update concept nodes and relationships from Wikibase."""
    asyncio.run(update_concepts(dry_run=dry_run))


@app.command()
def documents(
    clear_documents: Annotated[
        bool,
        typer.Option(
            "--clear-documents", help="Clear existing documents before updating"
        ),
    ] = False,
    batch_size: Annotated[
        int,
        typer.Option("--batch-size", help="Number of documents per batch"),
    ] = 50,
    batch_delay: Annotated[
        float,
        typer.Option("--batch-delay", help="Delay in seconds between batches"),
    ] = 0.5,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Run without making changes to Neo4j"),
    ] = False,
) -> None:
    """Update document and passage nodes from S3."""
    asyncio.run(
        update_documents(
            clear_documents=clear_documents,
            batch_size=batch_size,
            batch_delay=batch_delay,
            dry_run=dry_run,
        )
    )


if __name__ == "__main__":
    app()
