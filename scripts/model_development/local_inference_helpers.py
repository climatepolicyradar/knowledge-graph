from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from knowledge_graph.classifier import Classifier
from knowledge_graph.config import classifier_dir, processed_data_dir
from knowledge_graph.labelled_passage import LabelledPassage

console = Console(highlight=False)


def label_passages(
    labelled_passages: list[LabelledPassage],
    classifier: Classifier,
    batch_size: int,
    predict_kwargs: dict[str, Any] = {},
) -> list[LabelledPassage]:
    """
    Label a list of passages with a classifier.

    :param list[LabelledPassage] labelled_passages: The passages to label.
    :param Classifier classifier: The classifier to use.
    :param int batch_size: The batch size to use.
    :param dict[str, Any] predict_kwargs: optional keyword arguments to pass to the
    classifier's predict method.

    :return list[LabelledPassage]: The labelled passages.
    """
    predictions: list[LabelledPassage] = []
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        MofNCompleteColumn(),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as progress_bar:
        task = progress_bar.add_task(
            f"Labeling passages with {classifier}", total=len(labelled_passages)
        )
        for batch_start_index in range(0, len(labelled_passages), batch_size):
            batch = labelled_passages[
                batch_start_index : batch_start_index + batch_size
            ]
            batch_texts = [passage.text for passage in batch]
            try:
                batch_spans = classifier.predict_batch(batch_texts, **predict_kwargs)
                predictions.extend(
                    LabelledPassage(text=text, spans=text_spans)
                    for text, text_spans in zip(batch_texts, batch_spans)
                )
            except Exception as e:
                console.log(
                    f"❌ Error predicting batch {batch_start_index // batch_size}: {e}"
                )
                continue
            n_positives = len([passage for passage in predictions if passage.spans])
            progress_bar.update(
                task,
                advance=batch_size,
                description=f"Found {n_positives} positive passages",
            )
    return predictions


def save_labelled_passages_and_classifier(
    labelled_passages: list[LabelledPassage],
    classifier: Classifier,
):
    """Save the labelled passages and classifier to a set of standardised paths"""

    predictions_path = (
        processed_data_dir
        / "predictions"
        / str(classifier.concept.wikibase_id)
        / f"{classifier.id}.jsonl"
    )
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    with open(predictions_path, "w", encoding="utf-8") as f:
        f.write("\n".join([passage.model_dump_json() for passage in labelled_passages]))
    console.log(f"💾 Saved labelled passages to {predictions_path}")

    classifier_path = (
        classifier_dir
        / str(classifier.concept.wikibase_id or "unknown")
        / f"{classifier.id}.pickle"
    )
    classifier_path.parent.mkdir(parents=True, exist_ok=True)
    classifier.save(classifier_path)
    console.log(f"💾 Saved classifier to {classifier_path}")
