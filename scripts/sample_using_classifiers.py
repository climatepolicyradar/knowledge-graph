from contextlib import nullcontext
from typing import Annotated

import pandas as pd
import typer
import wandb
from rich.console import Console
from rich.progress import track

from knowledge_graph.classifier import Classifier
from knowledge_graph.classifier.embedding import EmbeddingClassifier
from knowledge_graph.classifier.keyword import KeywordClassifier
from knowledge_graph.classifier.stemmed_keyword import StemmedKeywordClassifier
from knowledge_graph.config import WANDB_ENTITY, processed_data_dir
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.wandb_helpers import log_labelled_passages_artifact_to_wandb_run
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
    max_size_to_sample_from: int = typer.Option(
        default=25_000, help="Number of passages in the source data to sample from."
    ),
    track_and_upload: bool = typer.Option(
        default=True,
        help="Whether to track the run and upload the labelled passages to W&B. Defaults to True",
    ),
):
    """
    Sample passages likely to contain positive examples of a concept.

    Uses a range of classifiers, and combines (but does not deduplicate) results. Saves
    a list of labelled passages locally to the "classifier_sampled_passages" subdir of
    the processed data directory.

    If `track_and_upload` is True, also uploads the labelled passages to the W&B project
    for the concept.

    The script assumes you have already run the `build-dataset` command to create a
    local copy of the balanced dataset.
    """
    dataset_path = processed_data_dir / "combined_dataset.feather"

    try:
        with console.status("🚚 Loading combined dataset"):
            df = pd.read_feather(dataset_path)
        console.log(f"✅ Loaded {len(df)} passages from {dataset_path}")
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"{dataset_path} not found locally. If you haven't already, please run:\n"
            "  just build-dataset"
        ) from e

    max_size_to_sample_from = 500_000
    if len(df) > max_size_to_sample_from:
        console.log(
            f"Limiting input data from {len(df)} rows to {max_size_to_sample_from}"
        )
        df = df.iloc[:max_size_to_sample_from]

    with console.status("🔍 Fetching concept and subconcepts from Wikibase"):
        wikibase = WikibaseSession()
        concept = wikibase.get_concept(
            wikibase_id, include_labels_from_subconcepts=True
        )

    console.log(f"✅ Fetched {concept} from Wikibase")

    classifiers: list[Classifier] = [
        KeywordClassifier(concept),
        StemmedKeywordClassifier(concept),
        EmbeddingClassifier(concept, threshold=0.65),
    ]

    job_type = "sample_using_classifiers"

    wandb_config = {
        "concept_hash": concept.__hash__(),
    }

    with (
        wandb.init(
            entity=WANDB_ENTITY,
            project=concept.wikibase_id,
            job_type=job_type,
            config=wandb_config,
        )
        if track_and_upload
        else nullcontext()
    ) as run:
        labelled_passages: list[LabelledPassage] = []

        for classifier in classifiers:
            classifier.fit()
            console.log(f"✅ Created a {classifier}")

            classifier_labelled_passages: list[LabelledPassage] = []

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

                for (_, row), text, spans in zip(
                    batch_df.iterrows(), texts, spans_batch
                ):
                    if spans:
                        classifier_labelled_passages.append(
                            LabelledPassage(
                                text=text,
                                spans=spans,
                                metadata=row.to_dict(),
                            )
                        )

            n_spans = sum(len(entry.spans) for entry in classifier_labelled_passages)
            n_positive_passages = sum(
                len(entry.spans) > 0 for entry in classifier_labelled_passages
            )
            console.log(
                f"✅ Processed {len(df)} passages using classifier {classifier}. Found {n_positive_passages} which mention "
                f'"{classifier.concept}", with {n_spans} individual spans'
            )

            labelled_passages.extend(classifier_labelled_passages)

        # Save predictions locally
        all_predictions = "\n".join(
            [entry.model_dump_json() for entry in labelled_passages]
        )
        predictions_path = (
            processed_data_dir / "classifier_sampled_passages" / f"{wikibase_id}.jsonl"
        )
        predictions_path.parent.mkdir(parents=True, exist_ok=True)
        with open(predictions_path, "w", encoding="utf-8") as f:
            f.write(all_predictions)
        console.log(f"✅ Saved passages with predictions to {predictions_path}")

        if track_and_upload and run:
            console.log("📄 Creating labelled passages artifact")
            log_labelled_passages_artifact_to_wandb_run(
                labelled_passages=labelled_passages,
                run=run,
                concept=concept,
            )
            console.log("✅ Labelled passages uploaded successfully")


if __name__ == "__main__":
    app()
