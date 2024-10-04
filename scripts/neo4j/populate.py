# pylint: disable=no-member
from rich.console import Console
from rich.progress import track

from src.neo4j import get_neo4j_session
from src.neo4j.models import ConceptNode
from src.wikibase import WikibaseSession

console = Console()

session = get_neo4j_session(clear=True)

with console.status("Connecting to wikibase..."):
    wikibase = WikibaseSession()
console.log("Connected to Wikibase")

# get concepts from wikibase
limit = None
with console.status(f"Fetching {limit or 'all'} concepts from Wikibase..."):
    all_concepts = wikibase.get_concepts(limit=limit)
console.log(f"Fetched {len(all_concepts)} concepts from Wikibase")

# create nodes for each concept
for concept in track(
    all_concepts,
    console=console,
    description="Creating Neo4j nodes",
    total=len(all_concepts),
    transient=True,
):
    concept_node = ConceptNode.nodes.get_or_none(wikibase_id=concept.wikibase_id)
    if concept_node:
        console.log(f"Node for concept {concept} already exists")
        continue
    else:
        concept_node = ConceptNode(wikibase_id=concept.wikibase_id).save()

    for related_concept_wikibase_id in concept.related_concepts:
        related_concept_node = ConceptNode.nodes.get_or_none(
            wikibase_id=related_concept_wikibase_id
        )
        if related_concept_node:
            console.log(
                f"Node for related concept {related_concept_wikibase_id} already exists"
            )
        else:
            related_concept_node = ConceptNode(
                wikibase_id=related_concept_wikibase_id
            ).save()
        concept_node.related_to.connect(related_concept_node)

    for parent_concept_wikibase_id in concept.subconcept_of:
        parent_concept_node = ConceptNode.nodes.get_or_none(
            wikibase_id=parent_concept_wikibase_id
        )
        if parent_concept_node:
            console.log(
                f"Node for parent concept {parent_concept_wikibase_id} already exists"
            )
        else:
            parent_concept_node = ConceptNode(
                wikibase_id=parent_concept_wikibase_id
            ).save()
        concept_node.subconcept_of.connect(parent_concept_node)

    for has_subconcept_wikibase_id in concept.has_subconcept:
        subconcept_node = ConceptNode.nodes.get_or_none(
            wikibase_id=has_subconcept_wikibase_id
        )
        if subconcept_node:
            console.log(
                f"Node for subconcept {has_subconcept_wikibase_id} already exists"
            )
        else:
            subconcept_node = ConceptNode(wikibase_id=has_subconcept_wikibase_id).save()
        concept_node.has_subconcept.connect(subconcept_node)

    concept_node.save()
    console.log(f"Created node for concept {concept}")

console.log("Finished creating concept nodes")
