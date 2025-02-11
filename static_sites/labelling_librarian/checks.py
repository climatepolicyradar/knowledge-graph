from pydantic import BaseModel

import argilla as rg

from scripts.evaluate import create_gold_standard_labelled_passages
from src.argilla import dataset_to_labelled_passages
from src.labelled_passage import LabelledPassage


class LabellingIssue(BaseModel):
    passage_text: str
    message: str


def find_long_spans(dataset: rg.FeedbackDataset) -> list[LabellingIssue]:
    """Finds and flags long spans in the dataset"""
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
                        start_pattern='<span class="bg-red">', end_pattern="</span>"
                    )
                )
                issues.append(
                    LabellingIssue(
                        passage_text=passage.text,
                        message=(
                            f"Found a really long span in {dataset.name}:\n"
                            f"{highlighted_text}\n"
                        ),
                    )
                )

    return issues
