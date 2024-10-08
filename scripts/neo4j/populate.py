# pylint: disable=no-member
from rich.console import Console
from rich.progress import track

from scripts.config import processed_data_dir
from src.labelled_passage import LabelledPassage
from src.neo4j import get_neo4j_session
from src.neo4j.models import ConceptNode, DocumentNode, PassageNode
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
    description="Creating concept nodes in Neo4j",
    total=len(all_concepts),
    transient=True,
):
    concept_node = ConceptNode(
        wikibase_id=concept.wikibase_id, preferred_label=concept.preferred_label
    ).save()
    console.log(f'Created concept node for "{concept}"')


for concept in track(
    all_concepts,
    console=console,
    description="Creating relationships between nodes",
    total=len(all_concepts),
    transient=True,
):
    concept_node = ConceptNode.nodes.first(wikibase_id=concept.wikibase_id)
    for related_concept_wikibase_id in concept.related_concepts:
        related_concept_node = ConceptNode.nodes.first_or_none(
            wikibase_id=related_concept_wikibase_id
        )
        if not related_concept_node:
            related_concept_node = ConceptNode(
                wikibase_id=related_concept_wikibase_id
            ).save()
            console.log(f'Created concept node for "{related_concept_node}"')
        concept_node.related_to.connect(related_concept_node)

    for parent_concept_wikibase_id in concept.subconcept_of:
        parent_concept_node = ConceptNode.nodes.first_or_none(
            wikibase_id=parent_concept_wikibase_id
        )
        if not parent_concept_node:
            parent_concept_node = ConceptNode(
                wikibase_id=parent_concept_wikibase_id
            ).save()
            console.log(f'Created concept node for "{parent_concept_node}"')
        concept_node.subconcept_of.connect(parent_concept_node)

    for has_subconcept_wikibase_id in concept.has_subconcept:
        subconcept_node = ConceptNode.nodes.first_or_none(
            wikibase_id=has_subconcept_wikibase_id
        )
        if not subconcept_node:
            subconcept_node = ConceptNode(wikibase_id=has_subconcept_wikibase_id).save()
            console.log(f'Created concept node for "{subconcept_node}"')
        concept_node.has_subconcept.connect(subconcept_node)

    console.log(f'Created relationships for "{concept}"')

console.log("Finished creating concept graph")

console.log("Loading passage predictions...")

predictions_path = processed_data_dir / "predictions"
labelled_passages: list[LabelledPassage] = []
for predictions_file in predictions_path.glob("*.jsonl"):
    wikibase_id = predictions_file.stem
    concept_node = ConceptNode.nodes.first(wikibase_id=wikibase_id)
    with open(predictions_file, "r", encoding="utf-8") as f:
        labelled_passages.extend(
            [LabelledPassage.model_validate_json(line) for line in f]
        )

console.log(f"Loaded {len(labelled_passages)} labelled passages")

unique_documents = set(
    (
        labelled_passage.metadata["document_id"],
        labelled_passage.metadata["document_name"],
    )
    for labelled_passage in labelled_passages
)
console.log(f"Loaded {len(unique_documents)} unique documents")

for document_id, document_name in track(
    unique_documents,
    console=console,
    description="Creating document nodes",
    total=len(unique_documents),
    transient=True,
):
    document_node = DocumentNode.nodes.first_or_none(document_id=document_id)
    if not document_node:
        document_node = DocumentNode(
            document_id=document_id, title=document_name
        ).save()
        console.log(f'Created document node for "{document_node}"')
console.log("Finished creating document nodes")

for labelled_passage in track(labelled_passages, console=console):
    document_node = DocumentNode.nodes.first(
        document_id=labelled_passage.metadata["document_id"]
    )

    passage_node = PassageNode(text=labelled_passage.text).save()
    console.log(f'Created passage node for "{labelled_passage.id}"')

    document_node.passages.connect(passage_node)

    for span in labelled_passage.spans:
        concept_node = ConceptNode.nodes.first(wikibase_id=span.concept_id)
        if not concept_node:
            console.log(f'Concept node for "{span.concept_id}" not found')
            continue
        concept_node.passages.connect(passage_node)

console.log("Finished loading passage predictions")
