import os

import pandas as pd
from rich import box
from rich.console import Console
from rich.progress import track
from rich.table import Table

import argilla as rg
from scripts.config import concept_dir, metrics_dir
from src.classifier import ClassifierFactory
from src.labelled_passage import LabelledPassage, create_gold_standard_labelled_passages
from src.metrics import ConfusionMatrix
from src.wikibase import WikibaseSession

console = Console()

# init argilla client
rg.init(
    api_url=os.getenv("ARGILLA_API_URL"),
    api_key=os.getenv("ARGILLA_API_KEY"),
    workspace="knowledge-graph",
)

# we're interested in evaluating any/all of the concepts for which we have labelled data
# in argilla. datasets are named according to the corresponding concept's wikibase id.
datasets = [dataset for dataset in rg.list_datasets(workspace="knowledge-graph")]
wikibase_ids = [dataset.name for dataset in datasets]

# get the concepts from wikibase
wikibase = WikibaseSession()
with console.status("🔍 Getting concepts from Wikibase..."):
    concepts = wikibase.get_concepts(wikibase_ids=wikibase_ids)
wikibase_id_to_concept = {concept.wikibase_id: concept for concept in concepts}

# attach labelled passages to concepts
for dataset in track(
    datasets, description="Attaching labelled passages to concepts", transient=True
):
    wikibase_id = dataset.name
    wikibase_id_to_concept[wikibase_id].labelled_passages = [
        LabelledPassage.from_argilla_record(record) for record in dataset.records
    ]
concepts = list(wikibase_id_to_concept.values())
console.print(f"🔍 Got {len(concepts)} concepts from Wikibase")

# persist the concepts to disk
for concept in concepts:
    concept.save(concept_dir / f"{concept.wikibase_id}.json")
console.print("💾 Persisted concepts to disk")


metrics = []
for concept in concepts:
    metrics_dir.mkdir(parents=True, exist_ok=True)

    console.log(f'📚 Loaded concept "{concept}" from {concept_dir}')

    console.log("🥇 Creating a list of gold-standard labelled passages")
    gold_standard_labelled_passages = create_gold_standard_labelled_passages(concept)
    n_annotations = sum([len(entry.spans) for entry in gold_standard_labelled_passages])
    console.log(
        f"🚚 Loaded {len(gold_standard_labelled_passages)} labelled passages "
        f"with {n_annotations} individual annotations"
    )

    classifier = ClassifierFactory.create(concept)
    console.log(f"🤖 Created a {classifier}")

    model_labelled_passages = [
        labelled_passage.model_copy(
            update={"spans": classifier.predict(labelled_passage.text)},
            deep=True,
        )
        for labelled_passage in track(
            gold_standard_labelled_passages,
            description=f"Labelling passages with {classifier}",
            transient=True,
        )
    ]
    n_annotations = sum([len(entry.spans) for entry in model_labelled_passages])
    console.log(
        f"✅ Labelled {len(model_labelled_passages)} passages "
        f"with {n_annotations} individual annotations"
    )
    console.log(f"📊 Calculating performance metrics for {concept}")

    confusion_matrix = ConfusionMatrix.at_passage_level(
        ground_truth_passages=gold_standard_labelled_passages,
        predicted_passages=model_labelled_passages,
    )

    metrics.append(
        {
            "concept": concept.wikibase_id,
            "classifier": classifier.name,
            "precision": confusion_matrix.precision(),
            "recall": confusion_matrix.recall(),
            "f1 score": confusion_matrix.f1_score(),
            "support": confusion_matrix.support(),
        }
    )

# format all of the metrics as a table
df = pd.DataFrame(metrics)
table = Table(box=box.SIMPLE, show_header=True)
for column in df.columns:
    table.add_column(column)
for _, row in df.iterrows():
    formatted_row = [
        f"{value:.2f}" if isinstance(value, float) else str(value) for value in row
    ]
    table.add_row(*formatted_row)


console.print(table)
