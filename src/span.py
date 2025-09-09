import logging
import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import Optional

from pydantic import BaseModel, Field, computed_field, model_validator
from typing_extensions import Self

from src.identifiers import Identifier, WikibaseID

logger = logging.getLogger(__name__)


class SpanXMLConceptAnnotationError(Exception):
    """Raised when a span XML has incorrectly annotated concepts"""

    def __init__(self, xml: str):
        super().__init__(f"Span XML has incorrectly annotated concepts.\nXML:\t{xml}\n")


class UnitInterval(float):
    """A validated float in the unit interval [0.0, 1.0]."""

    def __new__(cls, value: int | float) -> Self:
        """Create a new value in the unit interval"""
        if not 0 <= value <= 1:
            raise ValueError(f"Values must be between 0 and 1. Got {value}")
        return super().__new__(cls, value)


class Span(BaseModel):
    """Represents a span within a text."""

    text: str = Field(..., description="The text of the span")
    start_index: int = Field(
        ..., ge=0, description="The start index of the span within the text"
    )
    end_index: int = Field(
        ..., gt=0, description="The end index of the span within the text"
    )
    concept_id: Optional[WikibaseID] = Field(
        None,
        description="The wikibase identifier associated with the span",
        examples=["Q42"],
    )
    labellers: list[str] = Field(
        default_factory=list,
        description=(
            "A list of identifiers for the labeller of the span. "
            "Could be a username, a user ID, a model name, etc."
        ),
        examples=[
            "alice",
            "bob",
            "68edec6f-fe74-413d-9cf1-39b1c3dad2c0",
            'KeywordClassifier("extreme weather")',
        ],
    )
    timestamps: list[datetime] = Field(
        default_factory=list,
        description=(
            "The timestamps at which the span was labelled. "
            "The list of timestamps should be aligned with the list of labellers, "
            "ie they should be the same length, and their order should match."
        ),
    )

    class Config:
        """Pydantic configuration for the Span model"""

        json_encoders = {datetime: lambda dt: dt.isoformat()}

    @computed_field
    @property
    def id(self) -> Identifier:
        """Return the unique identifier for the span."""
        return Identifier.generate(
            self.text,
            self.start_index,
            self.end_index,
            self.concept_id,
        )

    @model_validator(mode="after")
    def check_whether_span_is_valid(self):
        """Check whether the span is valid."""
        if self.start_index >= self.end_index:
            raise ValueError(
                f"The end index must be greater than the start index. Got {self}"
            )
        if self.end_index > len(self.text):
            raise ValueError(
                f"The end index must be less than or equal to the length of the text. Got {self}"
            )
        return self

    @model_validator(mode="after")
    def check_whether_timestamps_are_aligned(self):
        """Check whether the list of timestamps is aligned with the list of labellers."""
        if self.timestamps:  # timestamps can be an empty list
            if len(self.labellers) != len(self.timestamps):
                # but if they're supplied, they should be aligned with the labellers
                raise ValueError(
                    f"The lists of labellers and timestamps must be the same length. "
                    f"Got {self}"
                )
        return self

    @computed_field
    @property
    def labelled_text(self) -> str:
        """The span's labelled substring"""
        return self.text[self.start_index : self.end_index]

    def __len__(self):
        """Return the length of the span."""
        return self.end_index - self.start_index

    def __hash__(self) -> int:
        """Return a hash for the span."""
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Check whether two spans are equal."""
        if not isinstance(other, Span):
            return False
        return self.id == other.id

    @staticmethod
    def _validate_merge_candidates(spans: list["Span"]):
        """Check whether the spans can be merged."""
        if not all(span.text == spans[0].text for span in spans):
            raise ValueError("All spans must have the same text")
        if not all(span.concept_id == spans[0].concept_id for span in spans):
            raise ValueError("All spans must have the same concept_id")
        if len(spans) == 0:
            raise ValueError("Cannot merge an empty list of spans")

    @classmethod
    def union(cls, spans: list["Span"]) -> "Span":
        """
        Return the union of a set of overlapping spans

        The union of a set of spans is the smallest span that contains all of the spans.

        :param Span spans: The spans to union
        :return Span: A new span that is the union of the input spans
        """
        cls._validate_merge_candidates(spans)
        if len(spans) == 1:
            return spans[0]
        else:
            return Span(
                text=spans[0].text,
                start_index=min(span.start_index for span in spans),
                end_index=max(span.end_index for span in spans),
                concept_id=spans[0].concept_id,
                labellers=list(
                    set(labeller for span in spans for labeller in span.labellers)
                ),
            )

    @classmethod
    def intersection(cls, spans: list["Span"]) -> "Span":
        """
        Return the intersection of a set of overlapping spans

        The intersection of a set of spans is the largest span that is contained within
        all of the spans.

        :param Span spans: The spans to intersect
        :return Span: A new span that is the intersection of the input spans
        """
        cls._validate_merge_candidates(spans)
        if len(spans) == 1:
            return spans[0]
        else:
            return Span(
                text=spans[0].text,
                start_index=max(span.start_index for span in spans),
                end_index=min(span.end_index for span in spans),
                concept_id=spans[0].concept_id,
                labellers=list(
                    set(labeller for span in spans for labeller in span.labellers)
                ),
            )

    def overlaps(self, other: "Span") -> bool:
        """
        Check whether this span overlaps with another span in the same text.

        :param Span other: The other span
        :return bool: True if the spans overlap, False otherwise
        """
        return jaccard_similarity(self, other) > 0

    @classmethod
    def _validate_xml(
        cls,
        xml: str,
    ) -> None:
        """
        Validates that <concept>, </concept> tags have been applied correctly.

        This means that following every <concept> tag should be a </concept> tag.
        Nested annotations should not be attempted, and there should be an equal number
        of start and end tags.
        """

        tags = re.findall(r"</?concept>", xml)

        if not (
            tags[0] == "<concept>"
            and len(set(tags)) == 2
            and all(a != b for a, b in zip(tags, tags[1:]))
        ):
            raise SpanXMLConceptAnnotationError(xml)

    @classmethod
    def from_xml(
        cls,
        xml: str,
        concept_id: Optional[WikibaseID],
        labellers: list[str],
        input_text: Optional[str] = None,
    ) -> list["Span"]:
        """
        Convert an XML string to a list of Spans.

        :param str xml: an XML string with <concept> and </concept> tags
        :param WikibaseID concept_id: the Wikibase ID of the concept
        :param list[str] labellers: the labellers of the spans
        :param str input_text_to_align_with: input text to align the spans with. Useful
        if the original text has been modified by e.g. a generative model
        :return list[Span]: a list of Spans
        """

        cls._validate_xml(xml)

        text_without_tags = xml.replace("<concept>", "").replace("</concept>", "")
        span_timestamps = [datetime.now()] * len(labellers)

        if input_text is not None and input_text != text_without_tags:
            return Span._from_xml_with_alignment(
                xml=xml,
                concept_id=concept_id,
                labellers=labellers,
                input_text=input_text,
            )

        spans = []
        offset = 0
        for match in re.finditer(r"<concept>(.*?)</concept>", xml):
            start_index = match.start() - (offset * len("<concept></concept>"))
            end_index = start_index + len(match.group(1))
            offset += 1
            spans.append(
                Span(
                    text=text_without_tags,
                    start_index=start_index,
                    end_index=end_index,
                    concept_id=concept_id,
                    labellers=labellers,
                    timestamps=span_timestamps,
                )
            )

        return spans

    @classmethod
    def _from_xml_with_alignment(
        cls,
        xml: str,
        concept_id: Optional[WikibaseID],
        labellers: list[str],
        input_text: Optional[str],
    ) -> list["Span"]:
        """
        Convert an XML string to a list of spans which are aligned with input text.

        This is to address the fact that LLM classifiers don't reliably follow
        instructions, so will often subtly modify the input text whilst adding concept
        predictions. Instead of marking all of the predictions as invalid, we align
        the predictions with the text we gave the LLM (`input_text`) instead.

        See the tests for this method for real-world examples.
        """

        span_timestamps = [datetime.now()] * len(labellers)

        if input_text is None:
            raise ValueError(
                "Input text must be set to use `Span._from_xml_with_alignment`. You might need Span.from_xml instead."
            )

        spans = []
        for offset, match in enumerate(re.finditer(r"<concept>(.*?)</concept>", xml)):
            span_text = match.group(1)
            start_index_in_original = match.start() - (
                offset * len("<concept></concept>")
            )

            found_indices = find_span_text_in_input_text(
                input_text=input_text,
                span_text=span_text,
                span_start_index=start_index_in_original,
            )

            if found_indices is None:
                logger.warning(
                    f"No spans found matching {span_text} near to character offset {start_index_in_original} in original.\n{xml}"
                )
            else:
                start_index, end_index = found_indices
                spans.append(
                    Span(
                        text=input_text,
                        start_index=start_index,
                        end_index=end_index,
                        concept_id=concept_id,
                        labellers=labellers,
                        timestamps=span_timestamps,
                    )
                )

        return spans


def jaccard_similarity(span_a: Span, span_b: Span) -> UnitInterval:
    """
    Calculate the Jaccard similarity of two spans.

    The Jaccard similarity of two spans is defined as the size of the intersection
    divided by the size of their union. Also known as the Jaccard index, or intersection
    over union (IoU). See https://en.wikipedia.org/wiki/Jaccard_index.

    :param Span span_a: The first span
    :param Span span_b: The second span
    :return float: The Jaccard similarity of the two spans (guaranteed to be in [0.0, 1.0])
    """
    if span_a.text != span_b.text:
        raise ValueError("The spans must have the same text")
    intersection = max(
        0,
        min(span_a.end_index, span_b.end_index)
        - max(span_a.start_index, span_b.start_index),
    )
    union = max(span_a.end_index, span_b.end_index) - min(
        span_a.start_index, span_b.start_index
    )

    return UnitInterval(intersection / union)


def jaccard_similarity_for_span_lists(
    spans_a: list[Span], spans_b: list[Span]
) -> UnitInterval:
    """
    Calculate the Jaccard similarity between two lists of spans.

    This is calculated by creating a set of all character indices covered by spans in
    each list, and then calculating the Jaccard similarity (intersection over union)
    of these two sets. This provides a holistic measure of overlap that naturally
    handles multiple disjoint spans.

    Returns:
        float: The Jaccard similarity (guaranteed to be in [0.0, 1.0])
    """
    indices_a = {i for span in spans_a for i in range(span.start_index, span.end_index)}
    indices_b = {i for span in spans_b for i in range(span.start_index, span.end_index)}

    # If both lists are empty, return 1.0, ie perfect agreement
    if not indices_a and not indices_b:
        return UnitInterval(1.0)

    # If one list is empty but the other is not, return 0.0, ie maximum disagreement
    if not indices_a or not indices_b:
        return UnitInterval(0.0)

    # Otherwise, calculate the ratio of the intersection and union of the two sets
    intersection = len(indices_a.intersection(indices_b))
    union = len(indices_a.union(indices_b))

    return UnitInterval(intersection / union)


def group_overlapping_spans(
    spans: list[Span], jaccard_threshold: float = 0
) -> list[list[Span]]:
    """
    Create a list of groups of spans according to their overlap.

    :param list[Span] spans: The spans to group
    :param float jaccard_threshold: The minimum Jaccard similarity for two spans to be
    considered overlapping
    :return list[list[Span]]: A list of groups of overlapping spans
    """
    groups: list[list[Span]] = []
    for span in spans:
        found = False
        for group in groups:
            if any(
                jaccard_similarity(span, other) > jaccard_threshold for other in group
            ):
                group.append(span)
                found = True
                break
        if not found:
            groups.append([span])

    return groups


def merge_overlapping_spans(
    spans: list[Span], jaccard_threshold: float = 0
) -> list[Span]:
    """
    Merge a list of overlapping spans into a list of non-overlapping spans.

    :param list[Span] spans: The spans to merge
    :param float jaccard_threshold: The minimum Jaccard similarity for two spans to be
    considered overlapping
    :return list[Span]: A list of non-overlapping spans
    """
    return [
        Span.union(spans=group)
        for group in group_overlapping_spans(spans, jaccard_threshold)
    ]


def find_span_text_in_input_text(
    input_text: str,
    span_text: str,
    span_start_index: int,
    fuzzy_match_threshold: float = 0.9,
    n_spans_length_to_search: int = 4,
    span_length_error_margin: int = 1,
) -> Optional[tuple[int, int]]:
    """
    Find a span's text in an input text string.

    Used where the text might've been modified from the original, by e.g. a
    generative model. It first looks for an exact match at the expected location,
    and then a fuzzier match in a window around the expected location.

    :param str input_text: the text to search within
    :param str span_text: the text of the span to find
    :param int span_start_index: the expected start index of the span within the input text
    :param float fuzzy_match_threshold: the minimum similarity ratio for a fuzzy
    match to be considered a match
    :param int n_spans_length_to_search_either_side: the window (in units length of
    the input span) to search. The search window is centered on `span_start_idx`
    :param int span_length_error_margin: during fuzzy matching, also search for spans
    with length ± this parameter.
    :return Optional[tuple[int, int]]: the start and end indices of the span in the
    input text if found, otherwise None.
    """

    span_text = span_text.strip()
    span_text = re.sub(r"\s+", " ", span_text)

    # If an exact match is found at the expected location, return it
    if input_text[span_start_index : span_start_index + len(span_text)] == span_text:
        return span_start_index, span_start_index + len(span_text)

    # If not, then look for a fuzzy match in a window around the expected location,
    # and with span length within the error margin.
    window_length = len(span_text) * n_spans_length_to_search
    window_start = max(0, span_start_index - window_length // 2)
    window_end = min(len(input_text), window_start + window_length)
    span_length_range = range(
        len(span_text) - span_length_error_margin,
        len(span_text) + span_length_error_margin + 1,
    )

    best_match, best_start_index, best_end_index = None, None, None
    best_ratio = 0.0

    for candidate_span_length in span_length_range:
        for i in range(window_start, window_end - candidate_span_length + 1):
            candidate = input_text[i : i + candidate_span_length]
            ratio = SequenceMatcher(None, span_text, candidate).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = candidate
                best_start_index = i
                best_end_index = i + candidate_span_length

    if (
        all(_ is not None for _ in [best_match, best_start_index, best_end_index])
        and best_ratio > fuzzy_match_threshold
    ):
        return best_start_index, best_end_index  # type: ignore

    return None
