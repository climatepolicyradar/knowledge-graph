from pathlib import Path

import yaml
from rich.console import Console
from rich.progress import track

from src.classifier import ClassifierFactory
from src.concept import Concept
from src.wikibase import WikibaseSession

console = Console()

# Load the sampling config
console.log("Loading config")
config_path = Path("scripts/sampling_for_sectors_classifier/config/sectors.yaml")
sampling_config = yaml.safe_load(config_path.read_text())

# Set up the output directory
classifier_directory = Path("data/processed/classifiers")
classifier_directory.mkdir(parents=True, exist_ok=True)

wikibase = WikibaseSession()

for wikibase_id in track(
    sampling_config.get("wikibase_ids", []),
    description="Training classifiers",
    console=console,
    transient=True,
):
    console.log(f"Fetching concept ID: {wikibase_id}")
    concept = wikibase.get_concept(wikibase_id)

    # Fetch all of its subconcepts recursively
    subconcepts = wikibase.get_subconcepts(wikibase_id, recursive=True)

    # fetch all of the all_labels for all of the subconcepts and the concept itself
    all_labels = set(concept.all_labels)
    for subconcept in subconcepts:
        all_labels.update(subconcept.all_labels)

    classifier = ClassifierFactory.create(
        concept=Concept(
            preferred_label=concept.preferred_label,
            alternative_labels=list(all_labels),
        )
    )

    # until we have more sophisticated classifier implementations in the factory,
    # this is effectively a no-op
    classifier.fit()

    # Save the classifier to a file with the concept ID in the name
    classifier_path = classifier_directory / wikibase_id
    classifier.save(classifier_path)
    console.log(f"Saved classifier to {classifier_path}")
