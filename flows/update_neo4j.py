"""
A prefect flow to update our Neo4j database with the latest version of the knowledge graph.

Nightly runs of this flow keep the Neo4j database in sync with our Concept Store.

The flow creates ConceptNode nodes (by `wikibase_id`) and rebuilds the graph of 
concept-to-concept relationships (`RELATED_TO` and `SUBCONCEPT_OF`).

When a user sets `refresh_documents=True`, the flow will also create DocumentNode
and PassageNode nodes and (re)creates a set of extra relationships, establishing a 
complete knowledge graph:
- `(:DocumentNode)-[:HAS_PASSAGE]->(:PassageNode)`
- `(:PassageNode)-[:MENTIONS]->(:ConceptNode)`

The document refresh will not run automatically, it must be enabled by the user running
the flow manually, either locally or in the prefect console.

Refreshing document nodes and relationships requires a local set of aggregated inference
results from S3, under `data/processed/aggregated/*.json`. To download those files, run
the following command:
    aws s3 sync \
    s3://cpr-prod-data-pipeline-cache/inference_results/latest/ \
    data/processed/aggregated \
    --profile=prod

Command line usage
------------------
Refresh concepts only (default):
    python -m flows.update_neo4j

Refresh concepts and documents:
    python -m flows.update_neo4j -- --refresh-documents true
"""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Sequence

import botocore.exceptions
from cpr_sdk.ssm import get_aws_ssm_param
from neomodel import db
from prefect import flow, get_run_logger

from flows.utils import iterate_batch
from knowledge_graph.config import processed_data_dir
from knowledge_graph.neo4j import get_neo4j_session
from knowledge_graph.wikibase import WikibaseSession

CONCEPT_BATCH_SIZE = 5000
RELATIONSHIP_BATCH_SIZE = 1000
DOCUMENT_BATCH_SIZE = 100


def _setup_env_from_ssm() -> None:
    """
    Source required secrets from SSM

    Expected SSM parameter names:
    - /Neo4j/ConnectionURI -> NEO4J_CONNECTION_URI
    - /Wikibase/Cloud/ServiceAccount/Username -> WIKIBASE_USERNAME
    - /Wikibase/Cloud/ServiceAccount/Password -> WIKIBASE_PASSWORD
    - /Wikibase/Cloud/URL -> WIKIBASE_URL
    """
    logger = get_run_logger()

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
    logger = get_run_logger()
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
    result, _ = db.cypher_query(
        "MATCH (c:ConceptNode) RETURN c.wikibase_id as wikibase_id"
    )
    return {row[0] for row in result}


def create_or_update_concept_nodes(
    batch: Sequence[dict[str, Any]], *, dry_run: bool
) -> None:
    """Upsert ConceptNode nodes using `wikibase_id` as the stable key."""
    execute_cypher(
        """
        UNWIND $batch AS concept
        MERGE (c:ConceptNode {wikibase_id: concept.wikibase_id})
        SET c.preferred_label = concept.preferred_label
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
    logger = get_run_logger()
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


def create_relationships(
    query: str, batch: Sequence[dict[str, str]], *, dry_run: bool
) -> None:
    """Execute a parameterised `UNWIND` Cypher to create relationships."""
    execute_cypher(query, {"batch": list(batch)}, dry_run=dry_run)


def process_document_batch(doc_paths_batch: Sequence[str], *, dry_run: bool) -> None:
    """
    Upsert document/passages and link passages to concepts for a batch of files.

    Each file under `data/processed/aggregated/*.json` should be a JSON object
    mapping `text_block_id` to a list of concept mentions, where each mention contains
    an `id` field (Wikibase ID), e.g.:

        {
          "18593": [{"id": "Q123"}, {"id": "Q456"}],
          "18610": [{"id": "Q789"}]
        }

    The filename stem is treated as the `document_id`. For each file we:
    - MERGE a `(:DocumentNode {document_id})`
    - MERGE `(:PassageNode {document_passage_id})` per text block
    - MERGE `(:DocumentNode)-[:HAS_PASSAGE]->(:PassageNode)` for each text block
    - MERGE `(:PassageNode)-[:MENTIONS]->(:ConceptNode)` for each concept mention
    """
    logger = get_run_logger()
    all_documents: list[dict[str, str]] = []
    all_passages: list[dict[str, str]] = []
    all_doc_passage_rels: list[dict[str, str]] = []
    all_passage_concept_rels: list[dict[str, str]] = []

    for doc_path in doc_paths_batch:
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                document = json.load(f)

            document_id = Path(doc_path).stem
            all_documents.append({"document_id": document_id})

            for passage_id, concepts in document.items():
                if concepts:
                    document_passage_id = document_id + "_" + passage_id
                    all_passages.append({"document_passage_id": document_passage_id})
                    all_doc_passage_rels.append(
                        {"doc_id": document_id, "passage_id": document_passage_id}
                    )

                    for concept in concepts:
                        wikibase_id = concept["id"]
                        all_passage_concept_rels.append(
                            {
                                "passage_id": document_passage_id,
                                "concept_id": wikibase_id,
                            }
                        )
        except Exception as e:
            logger.error(f"Error processing {doc_path}: {e}")
            continue

    if all_documents:
        logger.info(
            "Documents batch — docs: %s, passages: %s, concept mentions: %s",
            len(all_documents),
            len(all_passages),
            len(all_passage_concept_rels),
        )

        execute_cypher(
            """
            UNWIND $docs AS doc
            MERGE (d:DocumentNode {document_id: doc.document_id})
            """,
            {"docs": all_documents},
            dry_run=dry_run,
        )

        if all_passages:
            unique_passages = [
                dict(t) for t in {tuple(d.items()) for d in all_passages}
            ]
            execute_cypher(
                """
                UNWIND $passages AS passage
                MERGE (p:PassageNode {document_passage_id: passage.document_passage_id})
                """,
                {"passages": unique_passages},
                dry_run=dry_run,
            )

        if all_doc_passage_rels:
            execute_cypher(
                """
                UNWIND $rels AS rel
                MATCH (d:DocumentNode {document_id: rel.doc_id})
                MATCH (p:PassageNode {document_passage_id: rel.passage_id})
                MERGE (d)-[:HAS_PASSAGE]->(p)
                """,
                {"rels": all_doc_passage_rels},
                dry_run=dry_run,
            )

        if all_passage_concept_rels:
            for i in range(0, len(all_passage_concept_rels), 10000):
                sub_batch = all_passage_concept_rels[i : i + 10000]
                execute_cypher(
                    """
                    UNWIND $rels AS rel
                    MATCH (p:PassageNode {document_passage_id: rel.passage_id})
                    MATCH (c:ConceptNode {wikibase_id: rel.concept_id})
                    MERGE (p)-[:MENTIONS]->(c)
                    """,
                    {"rels": sub_batch},
                    dry_run=dry_run,
                )


def execute_cypher(query: str, params: dict | None, *, dry_run: bool) -> None:
    """Run a Cypher statement against Neo4j, or log it when in dry-run mode"""
    logger = get_run_logger()
    if dry_run:
        logger.info("DRY RUN — skipping Cypher execution: %s", " ".join(query.split()))
        return
    if params is None:
        db.cypher_query(query)
    else:
        db.cypher_query(query, params)


@flow()
async def update_concepts(*, dry_run: bool = False) -> None:
    """Synchronise Neo4j with the concept graph from Wikibase"""

    logger = get_run_logger()
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
        logger.info("Connecting to Wikibase...")
        all_concepts = await wikibase.get_concepts_async()
        logger.info("Fetched concepts from Wikibase: %s", len(all_concepts))

        concepts_data = [
            {"wikibase_id": str(c.wikibase_id), "preferred_label": c.preferred_label}
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
                {"wikibase_id": id, "preferred_label": f"Concept {id}"}
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
                all_relationships["has_subconcept"].append(
                    {"from_id": str(concept.wikibase_id), "to_id": str(child_id)}
                )

        rel_queries = {
            "related_to": (
                """
                UNWIND $batch AS rel
                MATCH (from:ConceptNode {wikibase_id: rel.from_id})
                MATCH (to:ConceptNode {wikibase_id: rel.to_id})
                MERGE (from)-[:RELATED_TO]->(to)
                """
            ),
            "subconcept_of": (
                """
                UNWIND $batch AS rel
                MATCH (from:ConceptNode {wikibase_id: rel.from_id})
                MATCH (to:ConceptNode {wikibase_id: rel.to_id})
                MERGE (from)-[:SUBCONCEPT_OF]->(to)
                """
            ),
            "has_subconcept": (
                """
                UNWIND $batch AS rel
                MATCH (from:ConceptNode {wikibase_id: rel.from_id})
                MATCH (to:ConceptNode {wikibase_id: rel.to_id})
                MERGE (to)-[:SUBCONCEPT_OF]->(from)
                """
            ),
        }

        # Create new relationships
        for rel_type, relationships in all_relationships.items():
            if relationships:
                process_in_batches(
                    relationships,
                    RELATIONSHIP_BATCH_SIZE,
                    f"Creating {rel_type} relationships ({len(relationships)})",
                    lambda batch: create_relationships(
                        rel_queries[rel_type], batch, dry_run=dry_run
                    ),
                )
                logger.info(
                    "Relationships created — %s: %s", rel_type, len(relationships)
                )

        # Summary
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
async def update_documents(*, dry_run: bool = False) -> None:
    """Refresh document/passages and MENTIONS relationships using local data."""
    logger = get_run_logger()
    logger.info("Starting document link refresh from local aggregated results...")
    document_paths = [
        str(p) for p in (processed_data_dir / "aggregated").glob("*.json")
    ]
    logger.info("Found %s aggregated documents", len(document_paths))
    for i in range(0, len(document_paths), DOCUMENT_BATCH_SIZE):
        batch_paths = document_paths[i : i + DOCUMENT_BATCH_SIZE]
        process_document_batch(batch_paths, dry_run=dry_run)
    logger.info("Document link refresh finished")


@flow()
async def update_neo4j(
    refresh_documents: bool = False,
    dry_run: bool = False,
) -> None:
    """Refresh the Neo4j database with the latest version of the knowledge graph."""
    logger = get_run_logger()
    # Ensure required secrets are set before establishing connections to Neo4j and Wikibase
    _setup_env_from_ssm()
    # Connect to Neo4j here, for both subflows
    get_neo4j_session(clear=False)
    logger.info("Connected to Neo4j")

    # Always refresh the concept graph
    await update_concepts(dry_run=dry_run)

    # Optionally refresh the document graph
    if refresh_documents:
        await update_documents(dry_run=dry_run)


if __name__ == "__main__":
    asyncio.run(update_neo4j())
