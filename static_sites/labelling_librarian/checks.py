from pydantic import BaseModel

import argilla as rg

from scripts.evaluate import create_gold_standard_labelled_passages
from src.argilla import dataset_to_labelled_passages


class LabellingIssue(BaseModel):
    passage_text: str


def find_long_spans(dataset: rg.FeedbackDataset) -> list[LabellingIssue]:
    """Finds and flags long spans in the dataset"""
    issues = []

    labelled_passages = dataset_to_labelled_passages(dataset, unescape_html=False)
    merged_passages = create_gold_standard_labelled_passages(labelled_passages)

    for passage in merged_passages:
        if any(
            span for span in passage.spans if span.end_index - span.start_index > 50
        ):
            issues.append(LabellingIssue(passage_text=passage.text))

    return issues
