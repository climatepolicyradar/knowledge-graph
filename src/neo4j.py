import time

import neomodel
from neomodel import (
    ArrayProperty,
    Relationship,
    RelationshipFrom,
    RelationshipTo,
    StringProperty,
    StructuredNode,
)
from rich.console import Console

from src.concept import Concept


def clear_neo4j():
    neomodel.db.cypher_query("MATCH (n) DETACH DELETE n")


def ping():
    try:
        neomodel.db.cypher_query("MATCH (n) RETURN n LIMIT 1")
        return True
    except Exception:
        return False


def wait_for_neo4j():
    with Console().status("Waiting for neo4j to start..."):
        while not ping():
            time.sleep(1)


class ConceptNode(StructuredNode):
    """A concept interface for Neo4j"""

    preferred_label = StringProperty(required=True)
    alternative_labels = ArrayProperty(StringProperty())
    wikibase_id = StringProperty(unique_index=True)
    subconcept_of = RelationshipFrom("ConceptNode", "SUBCONCEPT_OF")
    subconcepts = RelationshipTo("ConceptNode", "HAS_SUBCONCEPT")
    related_to = Relationship("ConceptNode", "RELATED_TO")

    def to_concept(self) -> "Concept":
        # TODO make this work
        """Convert the ConceptNode to a Concept"""

        return Concept(
            preferred_label=self.preferred_label,
            alternative_labels=self.alternative_labels,
            wikibase_id=self.wikibase_id,
            subconcepts=[
                subconcept.to_concept() for subconcept in self.subconcepts.all()
            ],
        )

    @classmethod
    def from_concept(cls, concept: Concept) -> "ConceptNode":
        """Create a ConceptNode from a Concept"""
        return cls(
            preferred_label=concept.preferred_label,
            alternative_labels=list(concept.alternative_labels),
            wikibase_id=concept.wikibase_id,
        )
