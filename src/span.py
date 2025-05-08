import re
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, computed_field, model_validator

from src.identifiers import WikibaseID, deterministic_hash, generate_identifier


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
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="The confidence score of the classifier",
    )

    class Config:
        """Pydantic configuration for the Span model"""

        json_encoders = {datetime: lambda dt: dt.isoformat()}

    @computed_field
    def id(self) -> str:
        """Return the unique identifier for the span."""
        return generate_identifier(
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
    def labelled_text(self) -> str:
        """The span's labelled substring"""
        return self.text[self.start_index : self.end_index]

    def __len__(self):
        """Return the length of the span."""
        return self.end_index - self.start_index

    def __hash__(self) -> int:
        """Return a unique hash for the span."""
        return deterministic_hash(
            [
                self.text,
                self.start_index,
                self.end_index,
                self.concept_id,
            ]
        )

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
    def from_xml(
        cls,
        xml: str,
        concept_id: Optional[WikibaseID],
        labellers: list[str],
    ) -> list["Span"]:
        """Convert an XML string to a list of Spans."""
        text_without_tags = xml.replace("<concept>", "").replace("</concept>", "")
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
                    timestamps=[datetime.now()],
                )
            )

        return spans


def jaccard_similarity(span_a: Span, span_b: Span) -> float:
    """
    Calculate the Jaccard similarity of two spans.

    The Jaccard similarity of two spans is defined as the size of the intersection
    divided by the size of their union. Also known as the Jaccard index, or intersection
    over union (IoU). See https://en.wikipedia.org/wiki/Jaccard_index.

    :param Span span_a: The first span
    :param Span span_b: The second span
    :return float: The Jaccard similarity of the two spans
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
    return intersection / union


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
