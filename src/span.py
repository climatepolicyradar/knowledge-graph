from typing import Optional

from pydantic import BaseModel, Field, computed_field, model_validator

from src.identifiers import WikibaseID, generate_identifier


class Span(BaseModel):
    """Represents a span within a text."""

    id: str = Field(..., description="A unique identifier for the span")
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

    def __init__(self, text: str, start_index: int, end_index: int, **kwargs):
        concept_id = kwargs.pop("concept_id", None)
        id = kwargs.pop(
            "id",
            generate_identifier(
                text,
                start_index,
                end_index,
                concept_id,
                # shouldn't matter who the labeller is
            ),
        )
        super().__init__(
            text=text, start_index=start_index, end_index=end_index, id=id, **kwargs
        )

    @model_validator(mode="after")
    def check_whether_span_is_valid(self):
        """Check whether the span is valid."""
        if self.start_index > self.end_index:
            raise ValueError(
                f"The end index must be greater than the start index. Got {self}"
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
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        """Check whether two spans are equal."""
        if not isinstance(other, Span):
            return False
        return self.id == other.id

    @classmethod
    def union(cls, spans: list["Span"]) -> "Span":
        """
        Return the union of a set of overlapping spans

        The union of a set of spans is the smallest span that contains all of the spans.

        :param Span spans: The spans to union
        :return Span: A new span that is the union of the input spans
        """
        if not all(span.text == spans[0].text for span in spans):
            raise ValueError("All spans must have the same text")
        if not all(span.concept_id == spans[0].concept_id for span in spans):
            raise ValueError("All spans must have the same concept_id")
        if len(spans) == 0:
            raise ValueError("Cannot union an empty list of spans")
        if len(spans) == 1:
            return spans[0]
        else:
            labellers = list(
                set(labeller for span in spans for labeller in span.labellers)
            )
            return Span(
                text=spans[0].text,
                start_index=min(span.start_index for span in spans),
                end_index=max(span.end_index for span in spans),
                concept_id=spans[0].concept_id,
                labellers=labellers,
            )


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
    spans: list[Span], jaccard_threshold: float = 0.5
) -> list[list[Span]]:
    """
    Create a list of groups of spans according to their overlap.

    :param list[Span] spans: The spans to group
    :param float jaccard_threshold: The minimum Jaccard similarity for two spans to be
    considered overlapping, default 0.5
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
    spans: list[Span], jaccard_threshold: float = 0.5
) -> list[Span]:
    """
    Merge a list of overlapping spans into a list of non-overlapping spans.

    :param list[Span] spans: The spans to merge
    :param float jaccard_threshold: The minimum Jaccard similarity for two spans to be
    considered overlapping, default 0.5
    :return list[Span]: A list of non-overlapping spans
    """
    return [
        Span.union(spans=group)
        for group in group_overlapping_spans(spans, jaccard_threshold)
    ]
