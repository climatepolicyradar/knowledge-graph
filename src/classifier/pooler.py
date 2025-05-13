from itertools import combinations

import numpy as np

from src.classifier.classifier import Classifier
from src.classifier.llm import LLMOutputMismatchError
from src.concept import Concept
from src.span import Span, jaccard_similarity


class ClassifierPooler:
    """Manages a pool of classifiers, and is responsible for predicting, and aggregating their results."""

    def __init__(self, classifiers: list[Classifier], concept: Concept):
        if any(classifier.concept != concept for classifier in classifiers):
            raise ValueError("All classifiers must have the same concept.")
        self.classifiers = classifiers
        self.concept = concept

    def predict(self, text: str) -> list[Span]:
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

    def _same_spans(self, span1: Span, span2: Span) -> bool:
        """
        Does a comparison of the spans

        This is required because the spans may have different labellers, and the labeller is not
        part of the id on purpose. For this, see the test `test_whether_equivalent_spans_with_different_labellers_are_equal`
        for more details.
        """
        return span1 == span2 and span1.labellers == span2.labellers

    def _aggregate_two_overlapping_span_confidence(
        self, spans: list[Span], n_classifiers: int
    ) -> float:
        """
        Aggregates the confidence of two overlapping spans

        NOTE: I've first implemented this with the pairwise Jaccard index, but it penalised
        low overlaps too strongly, resulting in c=0.09 for example two below. I didn't
        feel this is representative of how confident we would be in that case.

        Instead, I use the average confidence for each of the characters in the merged span.
        For each character, the sum of the confidences of the classifiers that predicted it
        is divided by the number of classifiers in total.

        ```
        Example 1:
        Span 1: -----000000------- length: 6, confidence: 0.7
        Span 2: ---000000--------- length: 6, confidence: 0.3
        Span 3: ------------------ (this is to illustrate classifier 3 not predicting)
        Merged: ---00000000------- length: 8, confidence: 0.125

        > c_merged = (0.3 * 2 + 1 * 4 + 0.7 * 2) / 3 / 8 = 3 / 8 = 0.125

        Example 2:
        Span 1: -----000000------- length: 6, confidence: 0.9
        Span 2: ----------00000--- length: 5, confidence: 0.9
        Merged: -----0000000000--- length: 10, confidence: 0.495

        > c_merged = (0.9 * 5 + 1.8 * 1 + 0.9 * 4) / 2 / 10 = 0.495

        Example 3:
        Span 1: -----000000------- length: 6, confidence: 0.9
        Span 2: -------00000000--- length: 5, confidence: 0.8
        Span 3: ----------00000--- length: 5, confidence: 0.8
        Merged: -----0000000000--- length: 10, confidence: 0.5266...

        > c_merged = (0.9 * 2 + 1.7 * 3 + 2.5 * 1 + 1.6 * 4) / 3 / 10 = 0.5266...
        ```

        TODO: I'm still slightly uncomfortable with this aggregation. I'm not quite sure
        what the last 2 examples would mean in terms of confidence. I'm starting to think
        that we could either:
            > consider confidence on the token level
            > separate type-confidence and border-confidence
        """
        merged_span = Span.union(spans)

        char_confidences: list[float] = []
        for char_index in range(merged_span.start_index, merged_span.end_index):
            char_confidences.append(
                sum(
                    [
                        s.confidence if s.confidence is not None else 1.0
                        for s in spans
                        if s.start_index <= char_index < s.end_index
                    ]
                )
                / n_classifiers
            )

        return np.mean(char_confidences).astype(float)

    def _aggregate(self, spans: list[Span]) -> list[Span]:
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
            overlapping_spans = [
                s for s in spans if s.overlaps(span) and not self._same_spans(s, span)
            ]

            handled_spans.update(overlapping_spans)

            if len(overlapping_spans) == 0:
                results.append(
                    Span(
                        **{
                            **span.model_dump(),
                            "confidence": 1.0 / len(self.classifiers),
                        }
                    )
                )
            else:
                mergable_spans = overlapping_spans + [span]

                confidence = self._aggregate_two_overlapping_span_confidence(
                    mergable_spans, len(self.classifiers)
                )

                results.append(
                    Span(
                        **{
                            **Span.union(mergable_spans).model_dump(),
                            "confidence": float(confidence),
                        }
                    )
                )

        return results
