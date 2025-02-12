from typing import Callable

from pydantic import BaseModel

import argilla as rg
from argilla.client.feedback.schemas.responses import ResponseStatus
from scripts.evaluate import create_gold_standard_labelled_passages
from src.argilla import dataset_to_labelled_passages
from src.labelled_passage import LabelledPassage
from src.span import Span

DATASET_CACHE: dict[str, list[LabelledPassage]] = {}


def dataset_to_labelled_passages_with_cache(
    dataset: rg.FeedbackDataset,
) -> list[LabelledPassage]:
    """Turns the dataset into LabelledPassages using a cache"""
    dataset_name = dataset.name  # type: ignore
    if dataset_name in DATASET_CACHE:
        return DATASET_CACHE[dataset_name]

    labelled_passages = dataset_to_labelled_passages(dataset, unescape_html=False)
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


def all_dataset_level_checks(dataset: rg.FeedbackDataset) -> list[DatasetLevelIssue]:
    """Orchestrate all dataset checks"""
    issues = check_whether_dataset_is_empty(dataset)

    # if the dataset is empty, not worth running other checks
    if issues:
        return issues

    issues.extend(check_whether_dataset_has_a_high_discard_ratio(dataset))
    issues.extend(check_if_dataset_contains_few_positives(dataset))
    return issues


def check_if_dataset_contains_few_positives(
    dataset: rg.FeedbackDataset,
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


def dataset_contains_submitted_records(dataset: rg.FeedbackDataset) -> bool:
    for record in dataset.records:
        for response in record.responses:
            if response.status == ResponseStatus.submitted:
                return True
    return False


def check_whether_dataset_is_empty(
    dataset: rg.FeedbackDataset,
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
    dataset: rg.FeedbackDataset, threshold: float = 0.05
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


def _check_span_wrapper(
    dataset: rg.FeedbackDataset, span_issue: Callable[[Span], bool], issue_type: str
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
    dataset: rg.FeedbackDataset,
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
    dataset: rg.FeedbackDataset,
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
    dataset: rg.FeedbackDataset,
) -> list[PassageLevelIssue]:
    """Finds and flags spans that are too long"""
    return _check_span_wrapper(dataset, _span_is_long, "long_span")


def _span_is_long(span: Span) -> bool:
    return span.end_index - span.start_index > 50
