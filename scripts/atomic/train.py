from typing import Annotated

import typer
from rich.console import Console

from scripts.config import classifier_dir, concept_dir
from src.classifier import ClassifierFactory
from src.concept import Concept
from src.identifiers import WikibaseID
from src.wikibase import WikibaseSession

console = Console()
app = typer.Typer()


@app.command()
def main(
    wikibase_id: Annotated[
        WikibaseID,
        typer.Option(
            ...,
            help="The Wikibase ID of the concept classifier to train",
            parser=WikibaseID,
        ),
    ],
):
    # Set up the output directory
    classifier_dir.mkdir(parents=True, exist_ok=True)

    console.log(f"Loading concept {wikibase_id} from {concept_dir}")
    try:
        concept = Concept.load(concept_dir / f"{wikibase_id}.json")
    except FileNotFoundError as e:
        raise typer.BadParameter(
            f"Data for {wikibase_id} not found. \n"
            "If you haven't already, you should run:\n"
            f"  just get-concept {wikibase_id}\n"
        ) from e

    wikibase = WikibaseSession()

    # Fetch all of its subconcepts recursively
    subconcepts = wikibase.get_subconcepts(wikibase_id, recursive=True)

    # fetch all of the labels and negative_labels for all of the subconcepts
    # and the concept itself
    all_positive_labels = set(concept.all_labels)
    all_negative_labels = set(concept.negative_labels)
    for subconcept in subconcepts:
        all_positive_labels.update(subconcept.all_labels)
        all_negative_labels.update(subconcept.negative_labels)

    concept.alternative_labels = list(all_positive_labels)
    concept.negative_labels = list(all_negative_labels)

    # Create a classifier instance
    classifier = ClassifierFactory.create(concept=concept)

    # until we have more sophisticated classifier implementations in the factory,
    # this is effectively a no-op
    classifier.fit()

    # Save the classifier to a file with the concept ID in the name
    classifier_path = classifier_dir / wikibase_id
    classifier.save(classifier_path)
    console.log(f"Saved {classifier} to {classifier_path}")


if __name__ == "__main__":
    app()
