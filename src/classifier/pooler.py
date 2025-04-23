from enum import Enum
from itertools import combinations
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field

from src.classifier.classifier import Classifier
from src.classifier.llm import LLMOutputMismatchError
from src.concept import Concept
from src.span import Span, jaccard_similarity


class ConfidenceType(str, Enum):
    """Confidence type for the classifier"""

    AGGREGATE = "aggregate"
    SELF_REPORTED = "self-reported"


class PoolResult(BaseModel):
    """Results from a classifier pooler"""

    span: Span = Field(
        description="The span of text that was classified",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="The confidence score of the classifier",
    )
    confidence_type: ConfidenceType = Field(
        default=ConfidenceType.AGGREGATE,
        description="The type of confidence score returned by the classifier",
    )
    confidence_reasonings: Optional[dict[str, str]] = Field(
        default=None,
        description="The reasoning (if provided) behind the classification. In the form of {model_name: reasoning}",
    )


class ClassifierPooler:
    """
    Manages a pool of classifiers, and is responsible for predicting, and aggregating their results.

    NOTE: Currently only implemented for LLMClassifier.
    """

    def __init__(self, classifiers: list[Classifier], concept: Concept):
        if any(classifier.concept != concept for classifier in classifiers):
            raise ValueError("All classifiers must have the same concept.")
        self.classifiers = classifiers
        self.concept = concept

    def predict(self, text: str) -> list[PoolResult]:
        """Run prediction with all classifiers and aggregate their results"""

        predictions = self._get_predictions(text)
        return self._aggregate(predictions)

    def _get_predictions(self, text: str) -> list[Span]:
        predictions: list[Span] = []

        for classifier in self.classifiers:
            try:
                predictions.extend(classifier.predict(text))
            except LLMOutputMismatchError:
                # If the classifier fails, we just ignore it for now. TODO
                continue

        return predictions

    def _aggregate(self, spans: list[Span]) -> list[PoolResult]:
        """
        Aggregates the results of the classifiers

        The aggregation confidence is based on the overlap of the results.
        If all the classifiers perfectly align for a particular span, the confidence is 1.0

        If a span is predicted by only a single classifier, the confidence is 1/n
        where n is the number of total classifiers.

        Hence the confidence is in (0, 1] in limit.
        """
        handled_spans = set()
        results = []
        for span in spans:
            if span in handled_spans:
                continue
            overlapping_spans = [s for s in spans if s.overlaps(span) and s != span]
            handled_spans.update(overlapping_spans)

            if len(overlapping_spans) == 0:
                results.append(
                    PoolResult(
                        span=span,
                        confidence=1.0 / len(self.classifiers),
                        confidence_type=ConfidenceType.AGGREGATE,
                    )
                )
            else:
                mergable_spans = overlapping_spans + [span]
                jaccard_similarities = []
                for s1, s2 in combinations(mergable_spans, 2):
                    jaccard_similarities.append(jaccard_similarity(s1, s2))

                confidence = (
                    np.mean(jaccard_similarities)
                    * len(mergable_spans)
                    / len(self.classifiers)
                )
                results.append(
                    PoolResult(
                        span=Span.union(mergable_spans),
                        confidence=float(confidence),
                        confidence_type=ConfidenceType.AGGREGATE,
                    )
                )

        return results
