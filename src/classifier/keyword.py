import re
from typing import List

from src.classifier.classifier import Classifier, Span
from src.concept import Concept


class KeywordClassifier(Classifier):
    """
    Classifier uses keyword matching to find instances of a concept in text.

    Keywords are based on the preferred and alternative labels of the concept. The
    classifier uses regular expressions to match the keywords in the text. Regexes are
    case-sensitive based on the casing of the keyword itself, and are applied at word
    boundaries in order of decreasing length to ensure that longer keywords are matched
    first.
    """

    def __init__(self, concept: Concept):
        """
        Create a new KeywordClassifier instance.

        :param Concept concept: The concept which the classifier will identify in text
        """
        super().__init__(concept)
        self.case_sensitive_labels = []
        self.case_insensitive_labels = []

        for label in sorted(self.concept.all_labels, key=len, reverse=True):
            if label.strip():  # Ensure the label is not just whitespace
                if any(char.isupper() for char in label):
                    self.case_sensitive_labels.append(re.escape(label))
                else:
                    self.case_insensitive_labels.append(re.escape(label))

        # Allow matches at the end by using word boundary \b
        self.case_sensitive_pattern = re.compile(
            r"\b(?:" + "|".join(self.case_sensitive_labels) + r")\b"
        )
        self.case_insensitive_pattern = re.compile(
            r"\b(?:" + "|".join(self.case_insensitive_labels) + r")\b", re.IGNORECASE
        )

    def predict(self, text: str) -> List[Span]:
        """
        Predict whether the supplied text contains an instance of the concept.

        :param str text: The text to predict on
        :return List[Span]: A list of spans in the text
        """
        spans = []
        matched_positions = set()

        # Case-sensitive matching
        for match in self.case_sensitive_pattern.finditer(text):
            start, end = match.span()
            if start != end and not any(start <= p < end for p in matched_positions):
                spans.append(
                    Span(
                        identifier=self.concept.wikibase_id,
                        start_index=start,
                        end_index=end,
                    )
                )
                matched_positions.update(range(start, end))

        # Case-insensitive matching
        for match in self.case_insensitive_pattern.finditer(text):
            start, end = match.span()
            if start != end and not any(start <= p < end for p in matched_positions):
                spans.append(
                    Span(
                        identifier=self.concept.wikibase_id,
                        start_index=start,
                        end_index=end,
                    )
                )
                matched_positions.update(range(start, end))

        return spans
