import html
import itertools

from pydantic import BaseModel, Field, model_validator

from argilla import FeedbackRecord, User
from src.identifiers import generate_identifier
from src.span import Span, merge_overlapping_spans


class LabelledPassage(BaseModel):
    """Represents a passage of text which has been labelled by an annotator"""

    id: str = Field(..., title="ID", description="The unique identifier of the passage")
    text: str = Field(..., title="Text", description="The text of the passage")
    spans: list[Span] = Field(
        default_factory=list,
        title="Spans",
        description="The spans in the passage which have been labelled by the annotator",
    )
    metadata: dict = Field(
        default_factory=dict,
        title="Metadata",
        description="Additional data, eg translation status or dataset",
    )

    def __init__(self, text: str, spans: list[Span], **kwargs):
        id = kwargs.pop("id", generate_identifier(text))
        super().__init__(text=text, spans=spans, id=id, **kwargs)

    @model_validator(mode="after")
    def check_whether_spans_are_within_text(self):
        """Check whether the spans are within the text"""
        for span in self.spans:
            if span.end_index > len(self.text):
                raise ValueError("end_index must be less than the length of the text")
        return self

    @classmethod
    def from_argilla_record(
        cls, record: FeedbackRecord, filter_for_submitted: bool = True
    ) -> "LabelledPassage":
        """
        Create a LabelledPassage object from an Argilla record

        :param FeedbackRecord record: The Argilla record to create the LabelledPassage
        object from
        :param bool filter_for_submitted: Whether to filter for only submitted records,
        defaults to True
        :return LabelledPassage: The created LabelledPassage object
        """
        text = html.unescape(record.fields.get("text", ""))

        metadata = record.metadata or {}
        spans = []

        # we've observed that users can submit multiple annotations for the same text!
        # we should only consider the most recent annotation from each.
        most_recent_annotation_from_each_user = [
            max(group, key=lambda record: record.updated_at)
            for _, group in itertools.groupby(
                sorted(record.responses, key=lambda response: response.user_id),
                key=lambda response: response.user_id,
            )
        ]
        for response in most_recent_annotation_from_each_user:
            user_name = User.from_id(response.user_id).username
            try:
                for entity in response.values["entities"].value:
                    spans.extend(
                        [
                            Span(
                                text=text,
                                start_index=entity.start,
                                end_index=entity.end,
                                concept_id=entity.label,
                                labellers=[user_name],
                            )
                        ]
                    )
            except KeyError:
                pass

        return cls(text=text, spans=spans, metadata=metadata)

    def get_highlighted_text(self, format="cyan") -> str:
        """
        Returns the text with highlighted spans, usable in a rich console output

        :param str format: Rich formatting style to use for highlights, default "cyan"
        :return str: The text with highlighted spans
        """
        # Decode HTML entities
        decoded_text = html.unescape(self.text)

        # merge any overlapping spans
        merged_spans = merge_overlapping_spans(self.spans)

        # create the output text
        output = ""
        last_end = 0
        sorted_spans = sorted(merged_spans, key=lambda span: span.start_index)
        for span in sorted_spans:
            # Add text before the span
            output += decoded_text[last_end : span.start_index]
            # Add highlighted span
            output += f"[{format}]{decoded_text[span.start_index : span.end_index]}[/{format}]"
            last_end = span.end_index

        # Add any remaining text after the last span
        output += decoded_text[last_end:]

        return output

    @property
    def labellers(self) -> list[str]:
        """
        Returns the set of labellers who labelled the passage

        :return list[str]: The set of labellers who labelled the passage
        """
        return list(
            set([labeller for span in self.spans for labeller in span.labellers])
        )
