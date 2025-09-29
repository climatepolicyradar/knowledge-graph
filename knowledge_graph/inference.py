from knowledge_graph.classifier import Classifier
from knowledge_graph.labelled_passage import LabelledPassage


def label_passages_with_classifier(
    classifier: Classifier,
    labelled_passages: list[LabelledPassage],
    batch_size: int = 15,
) -> list[LabelledPassage]:
    """
    Label passages using the provided classifier.

    Overwrites any spans that already exist in the labelled passages.
    """

    input_texts = [lp.text for lp in labelled_passages]
    model_predicted_spans = classifier.predict_many(
        input_texts,
        batch_size=batch_size,
        show_progress=True,
    )
    output_labelled_passages = [
        labelled_passage.model_copy(
            update={"spans": model_predicted_spans[idx]},
            deep=True,
        )
        for idx, labelled_passage in enumerate(labelled_passages)
    ]

    return output_labelled_passages
