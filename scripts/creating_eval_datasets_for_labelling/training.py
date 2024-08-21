import typer
from rich.console import Console
from rich.progress import track

from scripts.config import classifier_dir
from src.classifier import ClassifierFactory
from src.concept import Concept
from src.sampling import SamplingConfig
from src.wikibase import WikibaseSession

console = Console()
app = typer.Typer()


@app.command()
def main(config_path: str):
    # Load the sampling config
    console.log(f"Loading config from {config_path}")
    sampling_config = SamplingConfig.load(config_path)
    console.log(f"Config loaded: {sampling_config}")

    # Set up the output directory
    classifier_dir.mkdir(parents=True, exist_ok=True)

    wikibase = WikibaseSession()

    for wikibase_id in track(
        sampling_config.wikibase_ids,
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
        classifier_path = classifier_dir / wikibase_id
        classifier.save(classifier_path)
        console.log(f"Saved classifier to {classifier_path}")


if __name__ == "__main__":
    app()
