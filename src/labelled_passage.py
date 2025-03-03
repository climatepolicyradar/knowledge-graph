import html
import itertools
import re

from argilla import Argilla, Record, Response
from argilla.v1 import FeedbackRecord
from argilla.v1 import User as LegacyUser
from pydantic import BaseModel, Field, model_validator

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
                raise ValueError(
                    "end_index must be less than or equal to the length of the text"
                )
        return self

    @classmethod
    def from_argilla_record_legacy(cls, record: FeedbackRecord) -> "LabelledPassage":  # type: ignore
        """
        Create a LabelledPassage object from an Argilla record

        :param FeedbackRecord record: The Argilla record to create the LabelledPassage
        object from
        :return LabelledPassage: The created LabelledPassage object
        """
        text: str = record.fields.get("text", "")  # type: ignore

        metadata = record.metadata or {}  # type: ignore
        spans = []

        # we've observed that users can submit multiple annotations for the same text!
        # we should only consider the most recent annotation from each.
        most_recent_annotation_from_each_user = [
            max(group, key=lambda record: record.updated_at)
            for _, group in itertools.groupby(  # type: ignore
                sorted(record.responses, key=lambda response: response.user_id),  # type: ignore
                key=lambda response: response.user_id,
            )
        ]
        for response in most_recent_annotation_from_each_user:
            user_name = LegacyUser.from_id(response.user_id).username
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
                                timestamps=[response.updated_at],
                            )
                        ]
                    )
            except KeyError:
                pass

        return cls(text=text, spans=spans, metadata=metadata)

    @classmethod
    def from_argilla_record(cls, record: Record, client: Argilla) -> "LabelledPassage":
        """
        Create a LabelledPassage object from an Argilla record

        :param Record record: The Argilla record to create the LabelledPassage
        object from
        :return LabelledPassage: The created LabelledPassage object
        """
        text: str = record.fields.get("text", "")  # type: ignore

        metadata = record.metadata or {}  # type: ignore
        spans = []

        # we've observed that users can submit multiple annotations for the same text!
        # we should only consider the most recent annotation from each.
        # NOTE we don't seem to have this field anymore
        # most_recent_annotation_from_each_user = [
        #     max(group, key=lambda record: record.updated_at)
        #     for _, group in itertools.groupby(  # type: ignore
        #         sorted(record.responses, key=lambda response: response.user_id),  # type: ignore
        #         key=lambda response: response.user_id,
        #     )
        # ]

        # https://docs.argilla.io/latest/reference/argilla/records/responses/?h=#usage-examples
        # this suggests we call the attribute with the name of the question here
        # I believe all of our questions are named "entities", so this is safe, but not aligned to
        # what they imagined. Also, their typing is utterly unhelpful, so filling that in myself below.
        responses: list[Response] = record.responses["entities"]
        for response in responses:
            user = client.users(id=response.user_id)
            assert user is not None, f"User with id {response.user_id} not found"
            user_name = user.username
            try:
                # a "value" is a dict with the keys "start", "end", and "label"
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
                pass

        return cls(text=text, spans=spans, metadata=metadata)

    def get_highlighted_text(
        self, start_pattern: str = "[cyan]", end_pattern: str = "[/cyan]"
    ) -> str:
        """
        Returns the text with highlighted spans, usable in a rich console output

        :param str start_pattern: The pattern to add to the text at the start of a span
        :param str end_pattern: The pattern to add to the text at the end of a span
        :return str: The text with highlighted spans
        """
        # Decode HTML entities
        decoded_text = html.unescape(self.text)
        # remove all html tags
        decoded_text = re.sub(r"<[^>]*>", "", decoded_text)

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
            output += f"{start_pattern}{decoded_text[span.start_index : span.end_index]}{end_pattern}"
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

    def __eq__(self, other: object) -> bool:
        """Check whether two labelled passages are equal"""
        if not isinstance(other, LabelledPassage):
            return False
        if self.text != other.text:
            return False
        if set(self.spans) != set(other.spans):
            return False
        return True
