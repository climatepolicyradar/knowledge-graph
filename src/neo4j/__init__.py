import os
from time import sleep

from neo4j.exceptions import ServiceUnavailable
from neomodel import config, db, install_all_labels
from rich.console import Console

from src.neo4j.models import ConceptNode, DocumentNode, PassageNode

console = Console()


def get_neo4j_session(clear=False):
    neo4j_username, neo4j_password = os.environ["NEO4J_AUTH"].split("/", maxsplit=1)
    neo4j_connection_uri = f"bolt://{neo4j_username}:{neo4j_password}@localhost:7687"
    config.DATABASE_URL = neo4j_connection_uri
    db.set_connection(neo4j_connection_uri)
    wait_until_neo4j_is_live()

    if clear:
        console.log("Clearing neo4j database")
        db.clear_neo4j_database(db)

    # we don't care about the output of install_all_labels so we redirect it to /dev/null
    with open(os.devnull, "w", encoding="utf-8") as dev_null:
        db.install_all_labels(dev_null)

    return db


def wait_until_neo4j_is_live():
    while True:
        try:
            # run a simple query to check whether neo4j is live yet
            db.cypher_query("MATCH (n) RETURN n LIMIT 1")
            break
        except ServiceUnavailable:
            sleep(1)
            console.log("Connecting to neo4j...")
