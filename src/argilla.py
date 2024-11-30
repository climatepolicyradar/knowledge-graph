import os

import argilla as rg
from argilla import SpanQuestion, TextField
from argilla.feedback import FeedbackDataset, FeedbackRecord
from src.identifiers import WikibaseID
from src.labelled_passage import LabelledPassage
from src.wikibase import Concept


def init_argilla_client(func):
    def wrapper(*args, **kwargs):
        rg.init(  # type: ignore
            api_key=os.getenv("ARGILLA_API_KEY"), api_url=os.getenv("ARGILLA_API_URL")
        )
        return func(*args, **kwargs)

    return wrapper


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
            TextField(name="text", title="Text", use_markdown=True),  # type: ignore
        ],
        questions=[
            SpanQuestion(  # type: ignore
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


@init_argilla_client
def get_labelled_passages_from_argilla(concept: Concept) -> list[LabelledPassage]:
    # First, see whether the dataset exists with the name we expect
    dataset_name = concept_to_dataset_name(concept)
    dataset = rg.load(name=dataset_name)  # type: ignore

    # If it doesn't exist with the exact name, we can still try to find it by the
    # wikibase_id. This might happen if the concept has been renamed.
    if not dataset:
        datasets = rg.list_datasets()  # type: ignore
        if len(datasets) == 0:
            raise ValueError(
                "No datasets were found in Argilla, "
                "you may need to be granted access to the workspace(s)"
            )
        for dataset in datasets:
            try:
                # If the dataset.name ends with our wikibase_id, then it's one we want
                # to process
                if dataset_name_to_wikibase_id(dataset.name) == concept.wikibase_id:
                    break
            except ValueError:
                continue

    if not dataset:
        raise ValueError(
            f'No dataset matching the concept ID "{concept.wikibase_id}" was found in Argilla'
        )

    return dataset_to_labelled_passages(dataset)
