from argilla.feedback import FeedbackDataset, FeedbackRecord
from argilla.feedback.fields import TextField
from argilla.feedback.questions import SpanQuestion
from src.identifiers import WikibaseID
from src.labelled_passage import LabelledPassage
from src.wikibase import Concept


def concept_to_dataset_name(concept: Concept) -> str:
    return f"{concept.preferred_label}-{concept.wikibase_id}".replace(" ", "-")


def dataset_name_to_wikibase_id(name: str) -> WikibaseID:
    return WikibaseID(name.split("-")[-1])


def labelled_passages_to_feedback_dataset(
    labelled_passages: list[LabelledPassage], concept: Concept
) -> FeedbackDataset:
    """
    Convert a list of LabelledPassages into an Argilla FeedbackDataset.

    Args:
        labelled_passages: List of LabelledPassage objects to convert
        concept: The concept being annotated

    Returns:
        An Argilla FeedbackDataset ready to be pushed
    """
    dataset = FeedbackDataset(
        guidelines="Highlight the entity if it is present in the text",
        fields=[
            TextField(name="text", title="Text", use_markdown=True),
        ],
        questions=[
            SpanQuestion(
                name="entities",
                labels={concept.wikibase_id: concept.preferred_label},
                field="text",
                required=True,
                allow_overlapping=False,
            )
        ],
    )

    records = [
        FeedbackRecord(fields={"text": passage.text}, metadata=passage.metadata)
        for passage in labelled_passages
    ]
    dataset.add_records(records)

    return dataset


def dataset_to_labelled_passages(dataset: FeedbackDataset) -> list[LabelledPassage]:
    return [LabelledPassage.from_argilla_record(record) for record in dataset.records]
