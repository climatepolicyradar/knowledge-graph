import re
from datetime import datetime

from src.classifier.classifier import Classifier, Span
from src.concept import Concept


class KeywordClassifier(Classifier):
    """
    Classifier uses keyword matching to find instances of a concept in text.

    Keywords are based on the preferred and alternative labels of the concept. The
    classifier uses regular expressions to match the keywords in the text. Regexes are
    applied differently based on the casing of the original label:

    1. Case-sensitive matching: Applied to incoming labels containing any uppercase
    characters.
    2. Case-insensitive matching: Applied to labels containing only lowercase
    characters (this should apply to most labels).

    Regexes are applied at word boundaries in order of decreasing length to ensure that
    longer keywords are matched first.

    This approach allows for nuanced matching where:
    - Uppercase-containing labels (e.g., "WHO") will only match exactly ("WHO", not
    "who")
    - Lowercase-only labels (e.g., "virus") will match regardless of case ("virus",
    "Virus", "VIRUS")

    This distinction is particularly useful for differentiating between common words
    and specific entities (e.g., "who" vs "WHO" for the World Health Organization).
    """

    def __init__(self, concept: Concept):
        """
        Create a new KeywordClassifier instance.

        :param Concept concept: The concept which the classifier will identify in text
        """
        super().__init__(concept)

        def create_pattern(labels):
            return r"\b(?:" + "|".join(labels) + r")\b"

        self.case_sensitive_labels = []
        self.case_insensitive_labels = []

        # Sort labels by length in descending order so that longer labels are matched first
        sorted_labels = sorted(self.concept.all_labels, key=len, reverse=True)

        for label in sorted_labels:
            if label.strip():  # Ensure the label is not just whitespace
                if any(char.isupper() for char in label) or any(
                    ord(char) > 127 for char in label
                ):
                    # Labels including uppercase or non-ASCII characters are added to the case-sensitive list
                    self.case_sensitive_labels.append(re.escape(label))
                else:
                    # Only pure ASCII lowercase labels are added to the case-insensitive list
                    self.case_insensitive_labels.append(re.escape(label))

        # Case-sensitive pattern: matches exactly as provided
        self.case_sensitive_pattern = re.compile(
            create_pattern(self.case_sensitive_labels)
        )

        # Case-insensitive pattern: matches regardless of case
        self.case_insensitive_pattern = re.compile(
            create_pattern(self.case_insensitive_labels), re.IGNORECASE
        )

    def predict(self, text: str) -> list[Span]:
        """
        Predict whether the supplied text contains an instance of the concept.

        This method applies both case-sensitive and case-insensitive patterns to find
        matches. It ensures that:

        1. Longer matches take precedence over shorter ones
        2. No overlapping matches are returned
        3. Case-sensitive matches are found exactly as provided
        4. Case-insensitive matches can be found regardless of casing

        :param str text: The text to predict on
        :return list[Span]: A list of spans in the text
        """
        spans = []
        matched_positions = set()

        # Case-sensitive matching
        for match in self.case_sensitive_pattern.finditer(text):
            start, end = match.span()
            if start != end and not any(start <= p < end for p in matched_positions):
                spans.append(
                    Span(
                        text=text,
                        concept_id=self.concept.wikibase_id,
                        start_index=start,
                        end_index=end,
                        labellers=[str(self)],
                        timestamps=[datetime.now()],
                    )
                )
                matched_positions.update(range(start, end))

        # Case-insensitive matching
        for match in self.case_insensitive_pattern.finditer(text):
            start, end = match.span()
            if start != end and not any(start <= p < end for p in matched_positions):
                spans.append(
                    Span(
                        text=text,
                        concept_id=self.concept.wikibase_id,
                        start_index=start,
                        end_index=end,
                        labellers=[str(self)],
                        timestamps=[datetime.now()],
                    )
                )
                matched_positions.update(range(start, end))

        return spans
