from itertools import cycle
from typing import Any, Dict, Generator, Tuple

from argilla import FeedbackDataset, FeedbackRecord


def distribute_labelling_projects(
    datasets: list, labellers: list[str], min_labellers: int = 2
) -> Generator[Tuple[Any, str], None, None]:
    """
    Distribute labelling projects to labellers.

    For efficient labelling, tasks should be distributed such that each dataset is
    labelled by at least `min_labellers` labellers, and each labeller is assigned to a
    minimal number of datasets.

    :param list[] datasets: datasets to distribute among labellers
    :param list[str] labellers: list of labellers
    :param int min_labellers: minimum number of labellers per dataset, defaults to 2
    :return Generator[Tuple[Any, str], None, None]: a generator of tuples containing
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

    records_dict: Dict[str, FeedbackRecord] = {}  # type: ignore
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
