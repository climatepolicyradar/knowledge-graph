# pylint: disable=no-member
import pandas as pd
from rich.console import Console
from rich.progress import track

from scripts.config import classifier_dir, processed_data_dir
from src.classifier import Classifier
from src.models.labelled_passage import LabelledPassage
from src.neo4j import get_neo4j_session
from src.neo4j.models import ConceptNode, DocumentNode, PassageNode
from src.wikibase import WikibaseSession

console = Console()

wikibase_ids = [
    "Q368",
    "Q374",
    "Q404",
    "Q412",
    "Q757",
    "Q760",
    "Q761",
    "Q762",
    "Q763",
    "Q764",
    "Q765",
    "Q766",
    "Q767",
    "Q768",
    "Q769",
    "Q774",
    "Q775",
    "Q777",
    "Q778",
    "Q779",
    "Q786",
    "Q787",
    "Q788",
    "Q818",
    "Q856",
    "Q954",
    "Q955",
    "Q956",
    "Q973",
    "Q983",
    "Q986",
]

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

console.log("Creating classifiers...")
classifiers: list[Classifier] = []
for wikibase_id in track(
    wikibase_ids,
    console=console,
    description="Creating classifiers",
    total=len(wikibase_ids),
    transient=True,
):
    classifier = Classifier.load(classifier_dir / wikibase_id)
    classifiers.append(classifier)
    console.log(f"Created {classifier}")

console.log(f"Created {len(classifiers)} classifiers")

labelled_passages: list[LabelledPassage] = []
passages_dataset_path = processed_data_dir / "balanced_dataset_for_sampling.feather"
passages_df = pd.read_feather(passages_dataset_path)
console.log(f"Loaded document dataset from {passages_dataset_path}")

for classifier in classifiers:
    concept_labelled_passages: list[LabelledPassage] = []
    for _, row in track(
        passages_df.iterrows(),
        console=console,
        transient=True,
        total=len(passages_df),
        description=f"Running {classifier} on {len(passages_df)} passages",
    ):
        if text := row.get("text_block.text", ""):
            spans = classifier.predict(text)
            if spans:
                concept_labelled_passages.append(
                    LabelledPassage(
                        text=text, spans=spans, metadata=row.astype(str).to_dict()
                    )
                )

    labelled_passages.extend(concept_labelled_passages)
    n_passages = len(concept_labelled_passages)
    n_spans = sum([len(entry.spans) for entry in concept_labelled_passages])
    console.log(
        f"Processed {len(passages_df)} passages with {classifier}. "
        f"Found {n_passages} positive passages with {n_spans} individual spans"
    )

unique_documents = set(
    (
        labelled_passage.metadata["document_id"],
        labelled_passage.metadata["document_name"],
    )
    for labelled_passage in labelled_passages
)

n_unique_spans = sum([len(entry.spans) for entry in labelled_passages])
n_unique_passages = len(set([entry.id for entry in labelled_passages]))
n_unique_documents = len(unique_documents)
console.log(
    f"Found {n_unique_spans} spans across {n_unique_passages} passages "
    f"from {n_unique_documents} unique documents"
)


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

    passage_node = PassageNode.nodes.first_or_none(text=labelled_passage.text)
    if not passage_node:
        passage_node = PassageNode(text=labelled_passage.text).save()
        console.log(f'Created new passage node for "{labelled_passage.id}"')
    else:
        console.log(f'Found existing passage node for "{labelled_passage.id}"')

    document_node.passages.connect(passage_node)

    for span in labelled_passage.spans:
        concept_node = ConceptNode.nodes.first(wikibase_id=span.concept_id)
        if not concept_node:
            console.log(f'Concept node for "{span.concept_id}" not found')
            continue
        concept_node.passages.connect(passage_node)

console.log("Finished indexing passage predictions")
