import json
from functools import partial

from neomodel import db
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
    track,
)

from scripts.config import processed_data_dir
from src.neo4j import get_neo4j_session
from src.wikibase import WikibaseSession

console = Console()

# First check whether the user has downloaded the aggregated results from s3, and raise an
# error if not
aggregated_dir = processed_data_dir / "aggregated"
assert aggregated_dir.exists() and len(list(aggregated_dir.glob("*.json"))) > 0, (
    "It looks like you haven't downloaded the aggregated results from s3 yet.\n"
    "To download the necessary files, run the following from the root of this repo:\n"
    "  aws s3 sync s3://cpr-prod-data-pipeline-cache/inference_results/latest/ "
    "data/processed/aggregated --profile=prod"
)


DOCUMENT_BATCH_SIZE = 100
CONCEPT_BATCH_SIZE = 5000

with console.status("Connecting to neo4j and optimising settings for batch uploads"):
    db_session = get_neo4j_session(clear=True)
    try:
        db.cypher_query("CALL dbms.setConfigValue('db.transaction.timeout', '0')")
        console.log("Set unlimited transaction timeout")
    except Exception as e:
        console.log(f"Could not set timeout: {e}")

with console.status("Connecting to wikibase..."):
    wikibase = WikibaseSession()
console.log("Connected to Wikibase")


limit = None
with console.status(f"Fetching {limit or 'all'} concepts from Wikibase..."):
    all_concepts = wikibase.get_concepts(limit=limit)
console.log(f"Fetched {len(all_concepts)} concepts from Wikibase")


def process_in_batches_with_progress(items, batch_size, description, process_fn):
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TextColumn("({task.completed}/{task.total})"),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(description, total=len(items))
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            process_fn(batch)
            progress.update(task, advance=len(batch))


def create_concept_nodes(batch):
    db.cypher_query(
        """
        UNWIND $batch AS concept
        MERGE (c:ConceptNode {wikibase_id: concept.wikibase_id})
        SET c.preferred_label = concept.preferred_label
        """,
        {"batch": batch},
    )


concepts_data = [
    {"wikibase_id": c.wikibase_id, "preferred_label": c.preferred_label}
    for c in all_concepts
]
process_in_batches_with_progress(
    concepts_data, CONCEPT_BATCH_SIZE, "Creating concept nodes", create_concept_nodes
)
console.log(f"Created {len(all_concepts)} concept nodes in Neo4j")


all_known_ids = {c.wikibase_id for c in all_concepts}
newly_discovered_ids = set()

for concept in track(all_concepts, description="Preparing concept relationships..."):
    for related_id in concept.related_concepts:
        if related_id not in all_known_ids:
            newly_discovered_ids.add(related_id)
    for parent_id in concept.subconcept_of:
        if parent_id not in all_known_ids:
            newly_discovered_ids.add(parent_id)
    for child_id in concept.has_subconcept:
        if child_id not in all_known_ids:
            newly_discovered_ids.add(child_id)

# Create any missing concept nodes
if newly_discovered_ids:
    missing_concepts = [{"wikibase_id": id} for id in newly_discovered_ids]

    def create_missing_concept_nodes(batch):
        db.cypher_query(
            """
            UNWIND $batch AS concept
            MERGE (c:ConceptNode {wikibase_id: concept.wikibase_id})
            """,
            {"batch": batch},
        )

    process_in_batches_with_progress(
        missing_concepts,
        CONCEPT_BATCH_SIZE,
        f"Creating {len(missing_concepts)} missing concept nodes",
        create_missing_concept_nodes,
    )
    console.log(f"Created {len(missing_concepts)} missing concept nodes.")
    all_known_ids.update(newly_discovered_ids)


all_relationships = {"related_to": [], "subconcept_of": [], "has_subconcept": []}
for concept in track(all_concepts, description="Preparing relationships..."):
    for related_id in concept.related_concepts:
        all_relationships["related_to"].append(
            {"from_id": concept.wikibase_id, "to_id": related_id}
        )
    for parent_id in concept.subconcept_of:
        all_relationships["subconcept_of"].append(
            {"from_id": concept.wikibase_id, "to_id": parent_id}
        )
    for child_id in concept.has_subconcept:
        all_relationships["has_subconcept"].append(
            {"from_id": concept.wikibase_id, "to_id": child_id}
        )

rel_queries = {
    "related_to": """
        UNWIND $batch AS rel
        MATCH (from:ConceptNode {wikibase_id: rel.from_id})
        MATCH (to:ConceptNode {wikibase_id: rel.to_id})
        MERGE (from)-[:RELATED_TO]->(to)
    """,
    "subconcept_of": """
        UNWIND $batch AS rel
        MATCH (from:ConceptNode {wikibase_id: rel.from_id})
        MATCH (to:ConceptNode {wikibase_id: rel.to_id})
        MERGE (from)-[:SUBCONCEPT_OF]->(to)
    """,
    "has_subconcept": """
        UNWIND $batch AS rel
        MATCH (from:ConceptNode {wikibase_id: rel.from_id})
        MATCH (to:ConceptNode {wikibase_id: rel.to_id})
        MERGE (to)-[:SUBCONCEPT_OF]->(from)
    """,
}


def create_relationships(rel_type, query, batch):
    db.cypher_query(query, {"batch": batch})


for rel_type, relationships in all_relationships.items():
    if relationships:
        process_in_batches_with_progress(
            relationships,
            CONCEPT_BATCH_SIZE,
            f"Creating {rel_type} relationships",
            partial(create_relationships, rel_type, rel_queries[rel_type]),
        )
console.log("Finished creating concept graph")


document_paths = list((processed_data_dir / "aggregated").glob("*.json"))


def process_document_batch(doc_paths_batch, progress=None, main_task=None):
    """Process documents in large batches with minimal memory usage"""
    all_documents = []
    all_passages = []
    all_doc_passage_rels = []
    all_passage_concept_rels = []

    for doc_path in doc_paths_batch:
        try:
            with open(doc_path, "r", encoding="utf-8") as f:
                document = json.load(f)

            document_id = doc_path.stem
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
            console.log(f"Error processing {doc_path}: {e}")
            continue

    if all_documents:
        console.log(
            f"Processing batch: {len(all_documents)} docs, {len(all_passages)} passages, {len(all_passage_concept_rels)} concept links"
        )

        db.cypher_query(
            """
            UNWIND $docs AS doc
            MERGE (d:DocumentNode {document_id: doc.document_id})
        """,
            {"docs": all_documents},
        )

        if all_passages:
            unique_passages = [
                dict(t) for t in {tuple(d.items()) for d in all_passages}
            ]
            db.cypher_query(
                """
                UNWIND $passages AS passage
                MERGE (p:PassageNode {document_passage_id: passage.document_passage_id})
            """,
                {"passages": unique_passages},
            )

        # Create all doc-passage relationships
        if all_doc_passage_rels:
            db.cypher_query(
                """
                UNWIND $rels AS rel
                MATCH (d:DocumentNode {document_id: rel.doc_id})
                MATCH (p:PassageNode {document_passage_id: rel.passage_id})
                MERGE (d)-[:HAS_PASSAGE]->(p)
            """,
                {"rels": all_doc_passage_rels},
            )

        # Create all passage-concept relationships in sub-batches
        if all_passage_concept_rels:
            for i in range(
                0, len(all_passage_concept_rels), 10000
            ):  # Split very large batches
                sub_batch = all_passage_concept_rels[i : i + 10000]
                db.cypher_query(
                    """
                    UNWIND $rels AS rel
                    MATCH (p:PassageNode {document_passage_id: rel.passage_id})
                    MATCH (c:ConceptNode {wikibase_id: rel.concept_id})
                    MERGE (p)-[:MENTIONS]->(c)
                """,
                    {"rels": sub_batch},
                )

        if progress and main_task:
            progress.update(main_task, advance=len(doc_paths_batch))
        console.log(
            f"✅ Batch complete: {len(all_doc_passage_rels)} doc-passage, {len(all_passage_concept_rels)} passage-concept relationships"
        )


with Progress(
    TextColumn("[bold blue]{task.description}"),
    BarColumn(),
    TaskProgressColumn(),
    TextColumn("({task.completed}/{task.total})"),
    TimeRemainingColumn(),
    console=console,
) as progress:
    main_task = progress.add_task(
        f"Processing {len(document_paths)} documents in batches of {DOCUMENT_BATCH_SIZE}",
        total=len(document_paths),
    )
    for i in range(0, len(document_paths), DOCUMENT_BATCH_SIZE):
        batch_paths = document_paths[i : i + DOCUMENT_BATCH_SIZE]
        process_document_batch(batch_paths, progress=progress, main_task=main_task)

console.log("🎉 Finished indexing all predictions!")
