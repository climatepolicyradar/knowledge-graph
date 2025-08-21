from typing import Annotated

import typer
from rich.console import Console

from src.classifier import Classifier
from src.config import classifier_dir
from src.identifiers import WikibaseID
from src.labelled_passage import LabelledPassage

console = Console()

app = typer.Typer()


@app.command()
def main(
    wikibase_id: Annotated[
        WikibaseID,
        typer.Option(
            ...,
            help="The Wikibase ID of the concept classifier to run",
            parser=WikibaseID,
        ),
    ],
    input_string: Annotated[str, typer.Option()],
):
    """Run a classifier on a supplied string"""
    try:
        classifier = Classifier.load(classifier_dir / wikibase_id)
        console.log(f"Loaded {classifier} from {classifier_dir}")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Classifier for {wikibase_id} not found. \n"
            "If you haven't already, you should run:\n"
            f"  just train {wikibase_id}"
        ) from e

    spans = classifier.predict(input_string)
    labelled_passage = LabelledPassage(
        text=input_string,
        spans=spans,
    )
    console.log(labelled_passage.get_highlighted_text())

    console.log("Spans:")
    console.log(spans)


if __name__ == "__main__":
    app()
