from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.progress import track

from scripts.config import classifier_dir, processed_data_dir
from src.classifier import Classifier
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
):
    """
    Run classifiers on the documents in the combined dataset, and save the results

    This script runs inference of all the classifiers specified in the config on all of the
    documents in the combined dataset, and saves the resulting positive passages for each
    concept to a file.
    """
    console.log("🚚 Loading combined dataset")
    combined_dataset_path = processed_data_dir / "combined_dataset.feather"
    try:
        df = pd.read_feather(combined_dataset_path)
        console.log(f"✅ Loaded {len(df)} passages from local file")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Combined dataset not found. If you haven't already, you should run:\n"
            "  just build-dataset"
        ) from e

    try:
        classifier = Classifier.load(classifier_dir / wikibase_id)
        console.log(f"Loaded {classifier} from {classifier_dir}")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Classifier for {wikibase_id} not found. \n"
            "If you haven't already, you should run:\n"
            f"  just train {wikibase_id}"
        ) from e

    labelled_passages: list[LabelledPassage] = []
    for _, row in track(
        df.iterrows(),
        console=console,
        transient=True,
        total=len(df),
        description=f"Running {classifier} on {len(df)} passages",
    ):
        text = row.get("text", "")
        if text:
            spans = classifier.predict(text)
            # only save the passage if the classifier found something
            labelled_passages.append(
                LabelledPassage(
                    text=text,
                    spans=spans,
                    metadata=row.astype(str).to_dict(),
                )
            )

    n_spans = sum([len(entry.spans) for entry in labelled_passages])
    n_positive_passages = sum([len(entry.spans) > 0 for entry in labelled_passages])
    console.log(
        f"✅ Processed {len(df)} passages. Found {n_positive_passages} which mention "
        f'"{classifier.concept}", with {n_spans} individual spans'
    )

    predictions_path = processed_data_dir / "predictions" / f"{wikibase_id}.jsonl"
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(predictions_path, "w", encoding="utf-8") as f:
        f.writelines([entry.model_dump_json() + "\n" for entry in labelled_passages])

    console.log(f"Saved passages with predictions to {predictions_path}")


if __name__ == "__main__":
    app()
