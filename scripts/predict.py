from typing import Annotated, Type

import pandas as pd
import typer
from rich.console import Console
from rich.progress import track

from scripts.config import classifier_dir, processed_data_dir
from src.classifier import Classifier
from src.classifier.embedding import EmbeddingClassifier
from src.classifier.keyword import KeywordClassifier
from src.classifier.rules_based import RulesBasedClassifier
from src.classifier.stemmed_keyword import StemmedKeywordClassifier
from src.identifiers import WikibaseID
from src.labelled_passage import LabelledPassage
from src.wikibase import WikibaseSession

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

    dataset_path = processed_data_dir / "balanced_dataset_for_sampling.feather"
    try:
        with console.status("🚚 Loading combined dataset"):
            df = pd.read_feather(dataset_path)
        console.log(f"✅ Loaded {len(df)} passages from {dataset_path}")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            "Combined dataset not found. If you haven't already, you should run:\n"
            "  just build-dataset"
        ) from e

    with console.status("🔍 Fetching concept and subconcepts from Wikibase"):
        wikibase = WikibaseSession()
        concept = wikibase.get_concept(wikibase_id)
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
    console.log(f"✅ Fetched {concept} from Wikibase")

    classifier_specs: list[tuple[Type[Classifier], float | None]] = [
        # (classifier_class, threshold)
        (KeywordClassifier, None),
        (RulesBasedClassifier, None),
        (StemmedKeywordClassifier, None),
        (EmbeddingClassifier, 0.3),
        (EmbeddingClassifier, 0.6),
        (EmbeddingClassifier, 0.9),
    ]

    for classifier_class, threshold in classifier_specs:
        classifier = classifier_class(concept)
        classifier.fit()
        console.log(
            f"✅ Trained a {classifier.name} classifier for {concept.wikibase_id}"
        )
        classifier_path = (
            classifier_dir
            / str(concept.wikibase_id)
            / f"{classifier.name}{f'_{threshold}' if threshold else ''}.pickle"
        )
        classifier_path.parent.mkdir(parents=True, exist_ok=True)
        classifier.save(classifier_path)
        console.log(f"Saved {classifier.name} classifier to {classifier_path}")

        labelled_passages: list[LabelledPassage] = []
        for _, row in track(
            df.iterrows(),
            console=console,
            transient=True,
            total=len(df),
            description=f"Running {classifier} on {len(df)} passages",
        ):
            text = row.get("text_block.text", "")
            if text:
                if (
                    isinstance(classifier, EmbeddingClassifier)
                    and threshold is not None
                ):
                    spans = classifier.predict(text, threshold)
                else:
                    spans = classifier.predict(text)
                if spans:
                    # only save the passage if the classifier found something
                    labelled_passages.append(
                        LabelledPassage(
                            text=text,
                            spans=spans,
                            metadata=row.astype(str).to_dict(),
                        )
                    )

        n_spans = sum(len(entry.spans) for entry in labelled_passages)
        n_positive_passages = sum(len(entry.spans) > 0 for entry in labelled_passages)
        console.log(
            f"✅ Processed {len(df)} passages. Found {n_positive_passages} which mention "
            f'"{classifier.concept}", with {n_spans} individual spans'
        )

        predictions_path = (
            processed_data_dir
            / "predictions"
            / wikibase_id
            / f"{classifier.name}.jsonl"
        )
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        with open(predictions_path, "w", encoding="utf-8") as f:
            f.writelines(
                [entry.model_dump_json() + "\n" for entry in labelled_passages]
            )

        console.log(f"Saved passages with predictions to {predictions_path}")


if __name__ == "__main__":
    app()
