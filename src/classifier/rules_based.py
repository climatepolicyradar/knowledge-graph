import re

from src.classifier import KeywordClassifier
from src.concept import Concept


class RulesBasedClassifier(KeywordClassifier):
    """
    Classifier uses keyword matching to find instances of a concept in text.

    This modified version of the KeywordClassifier uses regular expressions to match the
    keywords in the text. Regexes are case-sensitive based on the casing of the keyword
    itself, and are applied at word boundaries in order of decreasing length to ensure
    that longer keywords are matched first.

    The classifier also supports negative keywords, which are used to exclude matches
    based on the context in which the keyword appears.
    """

    def __init__(self, concept: Concept):
        """
        Create a new RulesBasedClassifier instance.

        :param Concept concept: The concept which the classifier will identify in text
        """
        super().__init__(concept)
        negative_labels = sorted(self.concept.negative_labels, key=len, reverse=True)

        case_sensitive_labels = []
        case_insensitive_labels = []
        for label in sorted(self.concept.all_labels, key=len, reverse=True):
            if label.strip():  # Ensure the label is not just whitespace
                if any(char.isupper() for char in label):
                    case_sensitive_labels.append(re.escape(label))
                else:
                    case_insensitive_labels.append(re.escape(label))

        negative_pattern = "|".join(
            re.escape(label) for label in negative_labels if label.strip()
        )

        def create_pattern(labels):
            if labels:
                return rf'\b(?!(?:{negative_pattern})\b)({"|".join(labels)})\b'
            return r"(?!)"

        self.case_sensitive_pattern = re.compile(create_pattern(case_sensitive_labels))
        self.case_insensitive_pattern = re.compile(
            create_pattern(case_insensitive_labels), re.IGNORECASE
        )
