from neomodel import RelationshipFrom, RelationshipTo, StringProperty, StructuredNode
from neomodel.sync_.cardinality import One, ZeroOrMore


class DocumentNode(StructuredNode):
    """Neo4j node representing a document."""

    title = StringProperty(required=True)
    document_id = StringProperty(required=True, unique_index=True)
    passages = RelationshipTo("PassageNode", "HAS_PASSAGE")


class PassageNode(StructuredNode):
    """Neo4j node representing a labelled passage of text."""

    text = StringProperty(required=True)
    concepts = RelationshipTo("ConceptNode", "MENTIONS_CONCEPT", cardinality=ZeroOrMore)
    source_document = RelationshipFrom("DocumentNode", "HAS_PASSAGE", cardinality=One)


class ConceptNode(StructuredNode):
    """Neo4j node representing a concept."""

    wikibase_id = StringProperty(required=True, unique_index=True)
    preferred_label = StringProperty()
    passages = RelationshipFrom(
        "PassageNode", "MENTIONS_CONCEPT", cardinality=ZeroOrMore
    )
    subconcept_of = RelationshipTo(
        "ConceptNode", "SUBCONCEPT_OF", cardinality=ZeroOrMore
    )
    has_subconcept = RelationshipFrom(
        "ConceptNode", "SUBCONCEPT_OF", cardinality=ZeroOrMore
    )
    related_to = RelationshipTo("ConceptNode", "RELATED_TO", cardinality=ZeroOrMore)
