from typing import Annotated

import pandas as pd
import typer
from rich.console import Console
from rich.progress import track

from knowledge_graph.classifier import Classifier, ModelPath, get_local_classifier_path
from knowledge_graph.classifier.embedding import EmbeddingClassifier
from knowledge_graph.classifier.keyword import KeywordClassifier
from knowledge_graph.classifier.stemmed_keyword import StemmedKeywordClassifier
from knowledge_graph.config import processed_data_dir
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.wikibase import WikibaseSession

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
    batch_size: int = typer.Option(
        25,
        help="Number of passages to process in each batch",
    ),
):
    """
    Run classifiers on the balanced dataset, and save the results locally.

    This script runs inference for a set of classifiers on the balanced dataset, and
    saves the resulting positive passages for each concept to a file. The results are
    saved to the local filesystem and can be used for visualisation via the vibe_check
    tool.

    The script assumes you have already run the `build-dataset` command to create a
    local copy of the balanced dataset.
    """
    dataset_path = processed_data_dir / "balanced_dataset_for_sampling.feather"

    try:
        with console.status("🚚 Loading combined dataset"):
            df = pd.read_feather(dataset_path)
        console.log(f"✅ Loaded {len(df)} passages from {dataset_path}")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{dataset_path} not found locally. If you haven't already, please run:\n"
            "  just build-dataset"
        ) from e

    with console.status("🔍 Fetching concept and subconcepts from Wikibase"):
        wikibase = WikibaseSession()
        concept = wikibase.get_concept(
            wikibase_id, include_labels_from_subconcepts=True
        )

    console.log(f"✅ Fetched {concept} from Wikibase")

    classifiers: list[Classifier] = [
        KeywordClassifier(concept),
        StemmedKeywordClassifier(concept),
        EmbeddingClassifier(concept, threshold=0.6),
        EmbeddingClassifier(concept, threshold=0.625),
        EmbeddingClassifier(concept, threshold=0.65),
        EmbeddingClassifier(concept, threshold=0.675),
        EmbeddingClassifier(concept, threshold=0.7),
    ]

    for classifier in classifiers:
        classifier.fit()
        console.log(f"✅ Created a {classifier}")

        # Save the classifier
        target_path = ModelPath(wikibase_id=wikibase_id, classifier_id=classifier.id)
        version = str(classifier.version if classifier.version else "v0")
        classifier_path = get_local_classifier_path(
            target_path=target_path, version=version
        )
        classifier_path.parent.mkdir(parents=True, exist_ok=True)
        classifier.save(classifier_path)
        console.log(f"✅ Saved {classifier} to {classifier_path}")

        labelled_passages: list[LabelledPassage] = []

        n_batches = len(df) // batch_size + (1 if len(df) % batch_size else 0)
        for batch_start in track(
            range(0, len(df), batch_size),
            console=console,
            transient=True,
            total=n_batches,
            description=f"Running {classifier} on {len(df)} passages in batches of {batch_size}",
        ):
            batch_end = min(batch_start + batch_size, len(df))
            batch_df = df.iloc[batch_start:batch_end]

            texts = batch_df["text_block.text"].fillna("").tolist()
            spans_batch = classifier.predict_batch(texts)

            for (_, row), text, spans in zip(batch_df.iterrows(), texts, spans_batch):
                if spans:
                    labelled_passages.append(
                        LabelledPassage(
                            text=text,
                            spans=spans,
                            metadata=row.to_dict(),
                        )
                    )

        n_spans = sum(len(entry.spans) for entry in labelled_passages)
        n_positive_passages = sum(len(entry.spans) > 0 for entry in labelled_passages)
        console.log(
            f"✅ Processed {len(df)} passages. Found {n_positive_passages} which mention "
            f'"{classifier.concept}", with {n_spans} individual spans'
        )

        # Save predictions locally
        predictions = "\n".join(
            [entry.model_dump_json() for entry in labelled_passages]
        )
        predictions_path = (
            processed_data_dir
            / "predictions"
            / str(wikibase_id)
            / f"{classifier.id}.jsonl"
        )
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        with open(predictions_path, "w", encoding="utf-8") as f:
            f.write(predictions)
        console.log(f"✅ Saved passages with predictions to {predictions_path}")


if __name__ == "__main__":
    app()
