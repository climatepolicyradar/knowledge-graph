import os
from datetime import datetime
from itertools import cycle
from typing import Generator, Optional

import argilla as rg
from argilla import SpanQuestion, TextField
from argilla.feedback import FeedbackDataset, FeedbackRecord
from src.identifiers import WikibaseID
from src.labelled_passage import LabelledPassage
from src.wikibase import Concept


def init_argilla_client(func):
    def wrapper(*args, **kwargs):
        must_be_set = [
            "ARGILLA_API_KEY",
            "ARGILLA_API_URL",
            "ARGILLA_WORKSPACE",
        ]
        missing = [
            variable_name
            for variable_name in must_be_set
            if not os.getenv(variable_name)
        ]
        if missing:
            raise ValueError(
                "The following environment variables were not found, but must be set: "
                + ", ".join(missing)
            )

        rg.init(  # type: ignore
            api_key=os.getenv("ARGILLA_API_KEY"),
            api_url=os.getenv("ARGILLA_API_URL"),
            workspace=os.getenv("ARGILLA_WORKSPACE"),
        )
        return func(*args, **kwargs)

    return wrapper


def concept_to_dataset_name(concept: Concept) -> str:
    if not concept.wikibase_id:
        raise ValueError("Concept has no Wikibase ID")
    return concept.wikibase_id


def dataset_name_to_wikibase_id(name: str) -> WikibaseID:
    return WikibaseID(name)


def labelled_passages_to_feedback_dataset(
    labelled_passages: list[LabelledPassage], concept: Concept
) -> FeedbackDataset:
    """
    Convert a list of LabelledPassages into an Argilla FeedbackDataset.

    :param list[LabelledPassage] labelled_passages: The labelled passages to convert
    :param Concept concept: The concept being annotated
    :return FeedbackDataset: An Argilla FeedbackDataset, ready to be pushed
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
    """
    Convert an Argilla FeedbackDataset into a list of LabelledPassages.

    :param FeedbackDataset dataset: The Argilla FeedbackDataset to convert
    :return list[LabelledPassage]: A list of LabelledPassage objects
    """
    return [LabelledPassage.from_argilla_record(record) for record in dataset.records]


def is_between_timestamps(
    timestamp: datetime,
    min_timestamp: Optional[datetime],
    max_timestamp: Optional[datetime],
) -> bool:
    """
    Check whether a timestamp falls within a given time range.

    :param datetime timestamp: The timestamp to check
    :param Optional[datetime] min_timestamp: The minimum timestamp (inclusive). If None, no minimum limit.
    :param Optional[datetime] max_timestamp: The maximum timestamp (inclusive). If None, no maximum limit.
    :return bool: True if the timestamp is within the range, False otherwise
    """
    if max_timestamp and timestamp > max_timestamp:
        return False
    if min_timestamp and timestamp < min_timestamp:
        return False
    return True


def filter_labelled_passages_by_timestamp(
    labelled_passages: list[LabelledPassage],
    min_timestamp: Optional[datetime] = None,
    max_timestamp: Optional[datetime] = None,
) -> list[LabelledPassage]:
    filtered_passages = []
    for passage in labelled_passages:
        passage_copy = passage.model_copy(update={"spans": []})
        for span in passage.spans:
            span_copy = span.model_copy(update={"labellers": [], "timestamps": []})
            for labeller, timestamp in zip(span.labellers, span.timestamps):
                if is_between_timestamps(
                    timestamp=timestamp,
                    min_timestamp=min_timestamp,
                    max_timestamp=max_timestamp,
                ):
                    span_copy.labellers.append(labeller)
                    span_copy.timestamps.append(timestamp)

            if len(span_copy.labellers) > 0:
                passage_copy.spans.append(span_copy)

        if len(passage_copy.spans) > 0:
            filtered_passages.append(passage_copy)

    return filtered_passages


@init_argilla_client
def get_labelled_passages_from_argilla(
    concept: Concept,
    workspace: str = "knowledge-graph",
    min_timestamp: Optional[datetime] = None,
    max_timestamp: Optional[datetime] = None,
) -> list[LabelledPassage]:
    """
    Get the labelled passages from Argilla for a given concept.

    :param Concept concept: The concept to get the labelled passages for
    :param Optional[datetime] min_timestamp: Only get annotations made after this timestamp (inclusive), defaults to None
    :param Optional[datetime] max_timestamp: Only get annotations made before this timestamp (inclusive), defaults to None
    :raises ValueError: If no dataset matching the concept ID was found in Argilla
    :raises ValueError: If no datasets were found in Argilla, you may need to be granted access to the workspace(s)
    :return list[LabelledPassage]: A list of LabelledPassage objects
    """
    # First, see whether the dataset exists with the name we expect

    dataset = rg.FeedbackDataset.from_argilla(
        name=concept_to_dataset_name(concept), workspace=workspace
    )

    labelled_passages = dataset_to_labelled_passages(dataset)
    if min_timestamp or max_timestamp:
        labelled_passages = filter_labelled_passages_by_timestamp(
            labelled_passages, min_timestamp, max_timestamp
        )
    return labelled_passages


def distribute_labelling_projects(
    datasets: list, labellers: list[str], min_labellers: int = 2
) -> Generator[tuple[FeedbackDataset, str], None, None]:
    """
    Distribute labelling projects to labellers.

    For efficient labelling, tasks should be distributed such that each dataset is
    labelled by at least `min_labellers` labellers, and each labeller is assigned to a
    minimal number of datasets.

    :param list[] datasets: datasets to distribute among labellers
    :param list[str] labellers: list of labellers
    :param int min_labellers: minimum number of labellers per dataset, defaults to 2
    :return Generator[tuple[FeedbackDataset, str], None, None]: a generator of tuples containing
        the dataset and the labeller assigned to it
    """
    if len(labellers) < min_labellers:
        raise ValueError(
            "number of items in labellers must be greater than or equal to min_labellers"
        )

    labeller_cycle = cycle(labellers)
    for dataset in datasets:
        for _ in range(min_labellers):
            yield dataset, next(labeller_cycle)


def combine_datasets(*datasets: FeedbackDataset) -> FeedbackDataset:  # type: ignore
    """
    Combine an arbitrary number of argilla datasets into one.

    :param FeedbackDataset *datasets: Unspecified number of datasets to combine, at
    least one.
    :return FeedbackDataset: The combined dataset
    """
    if not datasets:
        raise ValueError("At least one dataset must be provided")

    combined_dataset = FeedbackDataset(  # type: ignore
        fields=datasets[0].fields,  # type: ignore
        questions=datasets[0].questions,  # type: ignore
        metadata_properties=datasets[0].metadata_properties,  # type: ignore
        vectors_settings=datasets[0].vectors_settings,  # type: ignore
        guidelines=datasets[0].guidelines,  # type: ignore
        allow_extra_metadata=datasets[0].allow_extra_metadata,  # type: ignore
    )

    records_dict: dict[str, FeedbackRecord] = {}  # type: ignore
    for dataset in datasets:
        for record in dataset.records:  # type: ignore
            # Use the 'text' field as the key (assuming it's unique)
            key = record.fields.get("text", "")

            if key in records_dict:
                # If the record already exists, merge the responses
                existing_record = records_dict[key]
                existing_record.responses.extend(record.responses)  # type: ignore
            else:
                # If it's a new record, add it to the dictionary
                records_dict[key] = record  # type: ignore

    # Convert the dictionary values back to a list of records
    combined_records = list(records_dict.values())

    # Add the combined records to the new dataset
    combined_dataset.add_records(combined_records)  # type: ignore

    return combined_dataset
