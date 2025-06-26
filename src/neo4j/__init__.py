import os
from time import sleep

from neomodel import config, db, install_all_labels
from rich.console import Console
from rich.progress import Progress, TaskID

from neo4j.exceptions import ServiceUnavailable
from src.neo4j.models import ConceptNode, DocumentNode, PassageNode

console = Console()


def clear_neo4j_database_in_batches(batch_size=50_000, console: Console = console):
    """Clear Neo4j database in batches to avoid memory issues"""
    console.log("Clearing neo4j database in batches...")

    # Count total relationships and nodes first
    with console.status("Counting existing data..."):
        rel_result = db.cypher_query("MATCH ()-[r]->() RETURN count(r) as total")
        total_relationships = rel_result[0][0][0] if rel_result[0] else 0

        node_result = db.cypher_query("MATCH (n) RETURN count(n) as total")
        total_nodes = node_result[0][0][0] if node_result[0] else 0

    console.log(
        f"Found {total_relationships:,} relationships and {total_nodes:,} nodes to delete"
    )

    with Progress(console=console) as progress:
        # First, delete all relationships in batches
        if total_relationships > 0:
            rel_task = progress.add_task(
                "Deleting relationships...", total=total_relationships
            )
            deleted_rels = 0

            while True:
                result = db.cypher_query(f"""
                    MATCH ()-[r]->()
                    WITH r LIMIT {batch_size}
                    DELETE r
                    RETURN count(r) as deleted
                """)
                deleted_count = result[0][0][0] if result[0] else 0
                if deleted_count == 0:
                    break

                deleted_rels += deleted_count
                progress.update(rel_task, completed=deleted_rels)

        # Then delete all nodes in batches
        if total_nodes > 0:
            node_task = progress.add_task("Deleting nodes...", total=total_nodes)
            deleted_nodes = 0

            while True:
                result = db.cypher_query(f"""
                    MATCH (n)
                    WITH n LIMIT {batch_size}
                    DELETE n
                    RETURN count(n) as deleted
                """)
                deleted_count = result[0][0][0] if result[0] else 0
                if deleted_count == 0:
                    break

                deleted_nodes += deleted_count
                progress.update(node_task, completed=deleted_nodes)

    console.log("✅ Database cleared successfully")


def get_neo4j_session(clear=False, console: Console = console):
    neo4j_connection_uri = os.environ.get("NEO4J_CONNECTION_URI")
    config.KEEP_ALIVE = True
    config.MAX_CONNECTION_LIFETIME = 300
    db.set_connection(neo4j_connection_uri)  # type: ignore
    wait_until_neo4j_is_live(console=console)

    if clear:
        clear_neo4j_database_in_batches()

    # we don't care about the output of install_all_labels so we redirect it to /dev/null
    with open(os.devnull, "w", encoding="utf-8") as dev_null:
        db.install_all_labels(dev_null)

    return db


def wait_until_neo4j_is_live(console: Console = console):
    while True:
        try:
            # run a simple query to check whether neo4j is live yet
            db.cypher_query("MATCH (n) RETURN n LIMIT 1")
            break
        except ServiceUnavailable:
            sleep(1)
            console.log("Connecting to neo4j...")
