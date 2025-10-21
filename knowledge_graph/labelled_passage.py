import html
import re
from collections import defaultdict

from pydantic import BaseModel, Field, model_validator

from knowledge_graph.identifiers import Identifier
from knowledge_graph.span import Span, merge_overlapping_spans


class LabelledPassage(BaseModel):
    """Represents a passage of text which has been labelled by an annotator"""

    id: str = Field(..., title="ID", description="The unique identifier of the passage")
    text: str = Field(..., title="Text", description="The text of the passage")
    spans: list[Span] = Field(
        default_factory=list,
        title="Spans",
        description="The spans in the passage which have been labelled by the annotator",
        repr=False,
    )
    metadata: dict = Field(
        default_factory=dict,
        title="Metadata",
        description="Additional data, eg translation status or dataset",
        repr=False,
    )

    def __init__(self, text: str, spans: list[Span], **kwargs):
        id = kwargs.pop("id", Identifier.generate(text))
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

    @property
    def sanitised_text(self) -> str:
        """A normalised version of the text which can be used for comparison"""
        return self.sanitise(self.text)

    @staticmethod
    def sanitise(text: str) -> str:
        """Sanitise text by replacing bad XML characters and normalizing text."""
        # First handle XML special characters
        bad_xml_strings = ["&", "<", ">", '"', "'"]
        xml_translation = str.maketrans(
            {string: "_" * len(string) for string in bad_xml_strings}
        )
        text = text.translate(xml_translation)

        # Then normalize common Unicode discrepancies and whitespace variations
        normalize_translation = str.maketrans(
            {
                " ": " ",
                "\n": " ",
                "\t": " ",
                "…": "...",
                "'": "'",
                "—": "-",
                "’": "'",
                "“": '"',
                "”": '"',
            }
        )
        return text.translate(normalize_translation)

    def __hash__(self) -> int:
        """Hash based on id for use in sets/dicts"""
        return hash(self.id)


def consolidate_passages_by_text(
    labelled_passages: list[LabelledPassage],
) -> list[LabelledPassage]:
    """
    Merge multiple LabelledPassages for the same text into single LabelledPassages.

    This function combines multiple passages with the same text into single
    passages, merging spans from all labellers. This is useful when pulling
    passages from Argilla where multiple labellers may have annotated the same
    text, resulting in separate passages per response.

    Metadata from the first passage in each group is preserved (since passages with
    identical text should have the same metadata from the same source document).

    Args:
        labelled_passages: List of passages, potentially containing duplicates
            with the same text but different spans from different labellers.

    Returns:
        List of merged_passages passages with unique texts and merged spans.

    Example:
        >>> # Three passages: same text, different labellers
        >>> passage1 = LabelledPassage(text="Hello world", spans=[span_a])
        >>> passage2 = LabelledPassage(text="Hello world", spans=[span_b])
        >>> passage3 = LabelledPassage(text="Different", spans=[span_c])
        >>> merged = consolidate_passages_by_text([passage1, passage2, passage3])
        >>> len(merged)  # Two unique texts
        2
        >>> len(merged[0].spans)  # Spans from both labellers combined
        2
    """
    passage_groups: dict[str, list[LabelledPassage]] = defaultdict(list)
    for passage in labelled_passages:
        passage_groups[passage.id].append(passage)

    merged_passages: list[LabelledPassage] = []
    for group in passage_groups.values():
        merged_passages.append(
            LabelledPassage(
                text=group[0].text,
                spans=[span for passage in group for span in passage.spans],
                metadata=group[0].metadata,
            )
        )
    return merged_passages
