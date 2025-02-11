from typing import Optional
from pydantic import BaseModel
from argilla.client.feedback.schemas.responses import ResponseStatus

import argilla as rg

from scripts.evaluate import create_gold_standard_labelled_passages
from src.argilla import dataset_to_labelled_passages
from src.labelled_passage import LabelledPassage


class LabellingIssue(BaseModel):
    dataset_name: str
    message: str
    type: str
    passage_text: Optional[str] = None


def dataset_discard_ratio(dataset: rg.FeedbackDataset) -> list[LabellingIssue]:
    """Returns the proportion of discarded items in the dataset"""
    dataset_name = dataset.name  # type: ignore

    discarded = 0
    total = 0
    for record in dataset.records:
        for response in record.responses:
            if response.status == ResponseStatus.discarded:
                discarded += 1
            total += 1

    if total == 0:
        return [
            LabellingIssue(
                dataset_name=dataset_name,
                message=f"<strong>{dataset_name}</strong> has no records",
                type="empty_dataset",
            )
        ]

    ratio = discarded / total

    if ratio > 0.05:
        return [
            LabellingIssue(
                dataset_name=dataset_name,
                message=(
                    f"<strong>{dataset_name}</strong> discard ratio is too high:"
                    f' <span class="bg-red-500">{discarded}/{total} ({ratio:.2f})</span>'
                ),
                type="high_discard_ratio",
            )
        ]
    return []


def find_long_spans(dataset: rg.FeedbackDataset) -> list[LabellingIssue]:
    """Finds and flags long spans in the dataset"""
    dataset_name = dataset.name  # type: ignore

    issues = []

    labelled_passages = dataset_to_labelled_passages(dataset, unescape_html=False)
    merged_passages = create_gold_standard_labelled_passages(labelled_passages)

    for passage in merged_passages:
        for span in passage.spans:
            if span.end_index - span.start_index > 50:
                individual_span_labelled_passage = LabelledPassage(
                    text=passage.text, spans=[span]
                )
                highlighted_text = (
                    individual_span_labelled_passage.get_highlighted_text(
                        start_pattern='<span class="bg-red-500">', end_pattern="</span>"
                    )
                )
                issues.append(
                    LabellingIssue(
                        dataset_name=dataset_name,
                        passage_text=passage.text,
                        message=(
                            f"<strong>{dataset_name}</strong>:\n"
                            f"{highlighted_text}\n"
                        ),
                        type="long_span",
                    )
                )

    return issues
