from typing import Optional

from pydantic import BaseModel, Field, model_validator

from src.identifiers import WikibaseID


class Span(BaseModel):
    """Represents a span within a text."""

    start_index: int = Field(
        ..., ge=0, description="The start index of the span within the text"
    )
    end_index: int = Field(
        ..., gt=0, description="The end index of the span within the text"
    )
    identifier: Optional[WikibaseID] = Field(
        None,
        description="The wikibase identifier associated with the span",
        examples=["Q42"],
    )
    labeller: Optional[str] = Field(
        None,
        description="An identifier for the labeller of the span. Could be a username, a user ID, a model name, etc.",
        examples=[
            "alice",
            "bob",
            "68edec6f-fe74-413d-9cf1-39b1c3dad2c0",
            'KeywordClassifier("extreme weather")',
        ],
    )

    def __len__(self):
        """Return the length of the span."""
        return self.end_index - self.start_index

    def __hash__(self) -> int:
        """Return a unique hash for the span."""
        return hash((self.start_index, self.end_index, self.identifier, self.labeller))

    @model_validator(mode="after")
    def check_whether_span_is_valid(self):
        """Check whether the span is valid."""
        if self.start_index >= self.end_index:
            raise ValueError(
                f"The end index must be greater than the start index. Got {self}"
            )
        return self


def spans_overlap(*spans: Span) -> bool:
    """
    Check whether the span overlaps with another given span.

    :param Span spans: The spans to check for overlap
    :return bool: True if the spans overlap, False otherwise
    """
    return max([span.start_index for span in spans]) < min(
        span.end_index for span in spans
    )


def jaccard_similarity(span_a: Span, span_b: Span) -> float:
    """
    Calculate the Jaccard similarity of two spans.

    The Jaccard similarity is defined as the size of the intersection divided by
    the size of the union. Also known as the Jaccard index, or intersection over
    union (IoU). See https://en.wikipedia.org/wiki/Jaccard_index.

    :param Span span_a: The first span
    :param Span span_b: The second span
    :return float: The Jaccard similarity of the two spans
    """
    intersection = max(
        0,
        min(span_a.end_index, span_b.end_index)
        - max(span_a.start_index, span_b.start_index),
    )
    union = len(span_a) + len(span_b) - intersection
    return intersection / union
