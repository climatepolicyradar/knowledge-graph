import itertools
from collections import defaultdict
from functools import lru_cache
from typing import Callable

import argilla as rg
from argilla import Response, ResponseStatus, User
from pydantic import BaseModel

from scripts.evaluate import create_gold_standard_labelled_passages
from src.argilla_v2 import client, dataset_to_labelled_passages
from src.labelled_passage import LabelledPassage
from src.metrics import count_span_level_metrics
from src.span import Span

DATASET_CACHE: dict[str, list[LabelledPassage]] = {}


def dataset_to_labelled_passages_with_cache(
    dataset: rg.Dataset,
) -> list[LabelledPassage]:
    """Turns the dataset into LabelledPassages using a cache"""
    dataset_name = dataset.name  # type: ignore
    if dataset_name in DATASET_CACHE:
        return DATASET_CACHE[dataset_name]

    labelled_passages = dataset_to_labelled_passages(dataset)
    merged_passages = create_gold_standard_labelled_passages(labelled_passages)

    DATASET_CACHE[dataset_name] = merged_passages
    return merged_passages


class LabellingIssue(BaseModel):
    """Base class for all labelling issues"""

    dataset_name: str
    message: str
    type: str


class PassageLevelIssue(LabellingIssue):
    """Issue at the passage level"""

    passage_text: str


class DatasetLevelIssue(LabellingIssue):
    """Issue at the dataset level"""

    pass


def all_dataset_level_checks(dataset: rg.Dataset) -> list[DatasetLevelIssue]:
    """Orchestrate all dataset checks"""
    issues = check_whether_dataset_is_empty(dataset)

    # if the dataset is empty, not worth running other checks
    if issues:
        return issues

    issues.extend(check_whether_dataset_has_a_high_discard_ratio(dataset))
    issues.extend(check_if_dataset_contains_few_positives(dataset))
    issues.extend(
        check_whether_dataset_has_a_low_level_of_interannotator_agreement(dataset)
    )
    return issues


def check_if_dataset_contains_few_positives(
    dataset: rg.Dataset,
) -> list[DatasetLevelIssue]:
    """Checks whether the dataset has too few positive responses"""
    labelled_passages = dataset_to_labelled_passages_with_cache(dataset)
    n_positives = len([p for p in labelled_passages if p.spans])
    positive_ratio = n_positives / len(labelled_passages)

    if positive_ratio < 0.2:
        positive_percentage = positive_ratio * 100

        return [
            DatasetLevelIssue(
                dataset_name=dataset.name,  # type: ignore
                message=f"<strong>{dataset.name}</strong> contains too few positive responses: "  # type: ignore
                f'<span class="text-red-500">{n_positives} positives, only {positive_percentage:.1f}% of the total</span>',
                type="few_positives",
            )
        ]
    return []


def dataset_contains_submitted_records(dataset: rg.Dataset) -> bool:
    for record in dataset.records:
        for response in record.responses:
            if response.status == ResponseStatus.submitted:
                return True
    return False


def check_whether_dataset_is_empty(
    dataset: rg.Dataset,
) -> list[DatasetLevelIssue]:
    if dataset_contains_submitted_records(dataset):
        return []
    else:
        return [
            DatasetLevelIssue(
                dataset_name=dataset.name,  # type: ignore
                message=f"<strong>{dataset.name}</strong> contains no submitted responses!",  # type: ignore
                type="empty_dataset",
            )
        ]


def check_whether_dataset_has_a_high_discard_ratio(
    dataset: rg.Dataset, threshold: float = 0.05
) -> list[DatasetLevelIssue]:
    """Returns the proportion of discarded items in the dataset"""
    dataset_name = dataset.name  # type: ignore

    total = 0
    discarded = 0
    for record in dataset.records:
        for response in record.responses:
            if response.status == ResponseStatus.discarded:
                discarded += 1
            total += 1

    ratio = discarded / total

    if ratio > threshold:
        return [
            DatasetLevelIssue(
                dataset_name=dataset_name,
                message=(
                    f"<strong>{dataset_name}</strong> discard ratio is too high:"
                    f' <span class="text-red-500">{discarded} discarded, {100 * ratio:.1f}% of the total</span>'
                ),
                type="high_discard_ratio",
            )
        ]
    return []


@lru_cache(maxsize=64)
def _get_username_from_id(user_id: str) -> str:
    user = client.users(id=user_id)
    assert isinstance(user, User)
    return user.username


def check_whether_dataset_has_a_low_level_of_interannotator_agreement(
    dataset: rg.Dataset, threshold: float = 0.5
) -> list[DatasetLevelIssue]:
    """Returns an issue if span-level interannotator agreement is low"""

    # Get all unique labeller names
    labeller_names = set()
    for record in dataset.records:
        responses: list[Response] = record.responses["entities"]
        for response in responses:
            user_name = _get_username_from_id(response.user_id)
            labeller_names.add(user_name)

    # If there's only one labeller, we can't calculate IAA
    if len(labeller_names) < 2:
        return []

    # Organize records by labeller
    passages_by_labeller = defaultdict(list)
    for record in dataset.records:
        text = record.fields.get("text", "")
        responses: list[Response] = record.responses["entities"]

        # Create a passage for each labeller's annotations
        for labeller in labeller_names:
            spans = []
            for response in responses:
                user_name = _get_username_from_id(response.user_id)
                if user_name == labeller:
                    try:
                        for value in response.value:
                            spans.append(
                                Span(
                                    text=text,
                                    start_index=value["start"],
                                    end_index=value["end"],
                                    concept_id=value["label"],
                                    labellers=[user_name],
                                    timestamps=[
                                        record.updated_at
                                    ],  # so it's the record that bears this attribute now rather than the response...
                                )
                            )
                    except KeyError:
                        continue

            labelled_passage = LabelledPassage(text=text, spans=spans)
            passages_by_labeller[labeller].append(labelled_passage)

    # Calculate pairwise IAA scores
    labeller_pairs = list(itertools.combinations(labeller_names, 2))
    iaa_scores = []

    for labeller_1, labeller_2 in labeller_pairs:
        confusion_matrix = count_span_level_metrics(
            passages_by_labeller[labeller_1],
            passages_by_labeller[labeller_2],
            threshold=0,  # Check for any overlap between annotators at the span level
        )
        iaa_scores.append(confusion_matrix.cohens_kappa())

    if any(iaa < threshold for iaa in iaa_scores):
        return [
            DatasetLevelIssue(
                dataset_name=dataset.name,
                message=(
                    f"Annotators seem to disagree on the labels in <strong>{dataset.name}</strong>.<br />"
                    + "".join(
                        [
                            f"The IAA between {labeller_1} and {labeller_2} is {iaa:.3f}<br />"
                            for (labeller_1, labeller_2), iaa in zip(
                                labeller_pairs, iaa_scores
                            )
                        ]
                    )
                ),
                type="low_interannotator_agreement",
            )
        ]
    return []


def _check_span_wrapper(
    dataset: rg.Dataset, span_issue: Callable[[Span], bool], issue_type: str
) -> list[PassageLevelIssue]:
    """Wrapper function for any checks that are run on the passage level, and create issues for each span that fails the criteria"""
    issues: list[PassageLevelIssue] = []
    dataset_name = dataset.name  # type: ignore

    merged_passages = dataset_to_labelled_passages_with_cache(dataset)

    for passage in merged_passages:
        for span in passage.spans:
            if span_issue(span):
                individual_span_labelled_passage = LabelledPassage(
                    text=passage.text, spans=[span]
                )
                highlighted_text = (
                    individual_span_labelled_passage.get_highlighted_text(
                        start_pattern='<span class="bg-red-500">', end_pattern="</span>"
                    )
                )
                issues.append(
                    PassageLevelIssue(
                        dataset_name=dataset_name,
                        passage_text=passage.text,
                        message=f"<strong>{issue_type}</strong>: {highlighted_text}",
                        type=issue_type,
                    )
                )

    return issues


def check_whether_span_border_is_in_word(
    dataset: rg.Dataset,
) -> list[PassageLevelIssue]:
    """Checks whether the span's start or end is in the middle of a word"""
    return _check_span_wrapper(dataset, _span_border_in_word, "span_border_in_word")


def _span_border_in_word(span: Span) -> bool:
    previous_character = (
        span.text[span.start_index - 1] if span.start_index > 0 else " "
    )
    next_character = (
        span.text[span.end_index] if span.end_index < len(span.text) else " "
    )

    if previous_character.isalnum() or next_character.isalnum():
        return True
    return False


def check_whether_spans_have_high_non_alphabetical_ratio(
    dataset: rg.Dataset,
) -> list[PassageLevelIssue]:
    """Finds and flags spans that have a high non-alphabetical ratio"""
    return _check_span_wrapper(
        dataset, _span_has_high_non_alphabetical_ratio, "high_non_alphabetical_ratio"
    )


def _span_has_high_non_alphabetical_ratio(span: Span) -> bool:
    labelled_text: str = span.labelled_text  # type: ignore
    non_alphabetical_ratio = sum(not c.isalpha() for c in labelled_text) / len(
        labelled_text
    )

    if non_alphabetical_ratio > 0.5:
        return True
    return False


def check_whether_spans_are_long(
    dataset: rg.Dataset,
) -> list[PassageLevelIssue]:
    """Finds and flags spans that are too long"""
    return _check_span_wrapper(dataset, _span_is_long, "long_span")


def _span_is_long(span: Span) -> bool:
    return span.end_index - span.start_index > 50
