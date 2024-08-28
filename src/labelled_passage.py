from argilla import FeedbackRecord
from pydantic import BaseModel, Field, field_validator, model_validator

from src.classifier import Span
from src.identifiers import generate_identifier


class LabelledPassage(BaseModel):
    """Represents a passage of text which has been labelled by an annotator"""

    id: str = Field(..., title="ID", description="The unique identifier of the passage")
    text: str = Field(..., title="Text", description="The text of the passage")
    spans: list[Span] = Field(
        default_factory=list,
        title="Spans",
        description="The spans in the passage which have been labelled by the annotator",
    )

    def __init__(self, text: str, spans: list[Span], **kwargs):
        id = kwargs.pop("id", generate_identifier(text))
        super().__init__(text=text, spans=spans, id=id, **kwargs)

    @field_validator("spans", mode="before")
    @classmethod
    def check_whether_spans_are_valid(cls, value):
        """Check whether the spans are valid"""
        for span in value:
            if span.start_index < 0 or span.end_index < 0:
                raise ValueError("start_index and end_index must be greater than 0")
            if span.start_index >= span.end_index:
                raise ValueError("start_index must be less than end_index")
        return value

    @model_validator(mode="after")
    def check_whether_spans_are_within_text(self):
        """Check whether the spans are within the text"""
        for span in self.spans:
            if span.end_index > len(self.text):
                raise ValueError("end_index must be less than the length of the text")
        return self

    @classmethod
    def from_argilla_record(cls, record: FeedbackRecord) -> "LabelledPassage":
        """
        Create a LabelledPassage object from an Argilla record

        :param record: The Argilla record to create the LabelledPassage object from
        :return: The created LabelledPassage object
        """
        text = record.fields.get("text", "")
        spans = []

        for response in record.responses or []:
            user_id = str(response.user_id)
            try:
                for entity in response.values["entities"].value:
                    spans.extend(
                        [
                            Span(
                                start_index=entity.start,
                                end_index=entity.end,
                                identifier=entity.label,
                                labeller=user_id,
                            )
                        ]
                    )
            except KeyError:
                pass

        return cls(text=text, spans=spans)

    @property
    def highlighted_text(self, format="cyan") -> str:
        """
        Returns the text with highlighted spans, usable in a rich console output

        :param str format: Rich formatting style to use for highlights, default "cyan"
        :return str: The text with highlighted spans
        """
        output = ""
        text = self.text
        for span in self.spans:
            output += text[: span.start_index]
            output += (
                f"[{format}]" + text[span.start_index : span.end_index] + f"[/{format}]"
            )
            text = text[span.end_index :]

        output += text
        return output

    @property
    def annotators(self) -> list[str]:
        """
        Returns the set of annotators who labelled the passage

        :return list[str]: The set of annotators who labelled the passage
        """
        return list(set(span.labeller for span in self.spans if span.labeller))
