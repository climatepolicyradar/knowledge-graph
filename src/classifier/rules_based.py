import re

from src.classifier import KeywordClassifier
from src.concept import Concept


class RulesBasedClassifier(KeywordClassifier):
    """
    Classifier uses keyword matching to find instances of a concept in text.

    This modified version of the KeywordClassifier uses regular expressions to match
    positive labels, while excluding matches that appear within negative labels.
    """

    def __init__(self, concept: Concept):
        """
        Create a new RulesBasedClassifier instance.

        :param Concept concept: The concept which the classifier will identify in text
        """
        super().__init__(concept)

        def create_pattern(positive_labels, negative_labels):
            # Sort labels by length in descending order to ensure longer matches take precedence
            positive_labels = sorted(positive_labels, key=len, reverse=True)
            negative_labels = sorted(negative_labels, key=len, reverse=True)

            # Escape special regex characters in labels
            positive_labels = [re.escape(label) for label in positive_labels]
            negative_labels = [re.escape(label) for label in negative_labels]

            # Create the positive pattern
            positive_pattern = r"\b(?:" + "|".join(positive_labels) + r")\b"

            # If there are negative labels, create a pattern that excludes them
            if negative_labels:
                negative_pattern = r"\b(?:" + "|".join(negative_labels) + r")\b"
                # The final pattern matches a positive label only if it's not part of a negative label
                return f"(?!{negative_pattern})({positive_pattern})"
            else:
                return f"({positive_pattern})"

        # Create case-sensitive and case-insensitive patterns
        self.case_sensitive_pattern = re.compile(
            create_pattern(
                [
                    label
                    for label in self.concept.all_labels
                    if any(c.isupper() for c in label)
                ],
                [
                    label
                    for label in self.concept.negative_labels
                    if any(c.isupper() for c in label)
                ],
            )
        )
        self.case_insensitive_pattern = re.compile(
            create_pattern(
                [
                    label.lower()
                    for label in self.concept.all_labels
                    if not any(c.isupper() for c in label)
                ],
                [
                    label.lower()
                    for label in self.concept.negative_labels
                    if not any(c.isupper() for c in label)
                ],
            ),
            re.IGNORECASE,
        )
