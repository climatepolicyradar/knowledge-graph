import re
from datetime import datetime

from src.classifier.classifier import Classifier, ZeroShotClassifier
from src.concept import Concept
from src.identifiers import ClassifierID
from src.span import Span, merge_overlapping_spans


class KeywordClassifier(Classifier, ZeroShotClassifier):
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

    If the concept has negative labels, the classifier will also match these negative
    terms and filter out any positive matches that overlap with negative matches.

    For example, given a concept like:
        Concept(preferred_label="gas", negative_labels=["greenhouse gas"])
    the classifier will match
        "I need to fill up my gas tank"
    but not
        "The greenhouse gas emissions are a major contributor to climate change."
    """

    def __init__(self, concept: Concept):
        """
        Create a new KeywordClassifier instance.

        :param Concept concept: The concept which the classifier will identify in text
        """
        super().__init__(concept)

        def create_pattern(
            labels: list[str], case_sensitive: bool = False
        ) -> re.Pattern:
            """Create a regex pattern from a list of labels."""
            pattern = r"\b(?:" + "|".join(labels) + r")\b" if labels else ""
            flags = re.IGNORECASE if not case_sensitive else 0
            return re.compile(pattern, flags)

        def split_by_case_handling(labels: list[str]) -> tuple[list[str], list[str]]:
            """Partition labels into case-sensitive and case-insensitive lists."""
            case_sensitive_labels = []
            case_insensitive_labels = []

            # Sort labels by length in descending order so that longer labels are matched first
            sorted_labels = sorted(labels, key=len, reverse=True)

            for label in sorted_labels:
                if label.strip():
                    if any(char.isupper() for char in label) or any(
                        ord(char) > 127 for char in label
                    ):
                        # Labels including uppercase or non-ASCII characters are added to the case-sensitive list
                        case_sensitive_labels.append(re.escape(label))
                    else:
                        # Only pure ASCII lowercase labels are added to the case-insensitive list
                        case_insensitive_labels.append(re.escape(label))

            return case_sensitive_labels, case_insensitive_labels

        # Process positive labels
        self.case_sensitive_positive_labels, self.case_insensitive_positive_labels = (
            split_by_case_handling(self.concept.all_labels)
        )

        # Process negative labels
        self.case_sensitive_negative_labels, self.case_insensitive_negative_labels = (
            split_by_case_handling(self.concept.negative_labels)
        )

        # Create positive patterns
        self.case_sensitive_positive_pattern = create_pattern(
            self.case_sensitive_positive_labels, case_sensitive=True
        )

        self.case_insensitive_positive_pattern = create_pattern(
            self.case_insensitive_positive_labels, case_sensitive=False
        )

        # Create negative patterns
        self.case_sensitive_negative_pattern = create_pattern(
            self.case_sensitive_negative_labels, case_sensitive=True
        )

        self.case_insensitive_negative_pattern = create_pattern(
            self.case_insensitive_negative_labels, case_sensitive=False
        )

    @property
    def id(self) -> ClassifierID:
        """Return a deterministic, human-readable identifier for the classifier."""
        return ClassifierID.generate(self.name, self.concept.id)

    def _match_spans(self, text: str, pattern: re.Pattern | None) -> list[Span]:
        """
        Find spans in text using the provided pattern.

        :param str text: The text to search in
        :param re.Pattern | None pattern: The compiled regex pattern (can be None)
        :return list[Span]: List of spans found by the pattern
        """
        if not pattern:
            return []

        spans = []
        for match in pattern.finditer(text):
            start, end = match.span()
            if start != end:
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
        return spans

    def predict(self, text: str) -> list[Span]:
        """
        Predict whether the supplied text contains an instance of the concept.

        This method applies both case-sensitive and case-insensitive patterns to find
        matches. It ensures that:

        1. Longer matches take precedence over shorter ones
        2. No overlapping matches are returned
        3. Case-sensitive matches are found exactly as provided
        4. Case-insensitive matches can be found regardless of casing
        5. Positive matches that overlap with negative matches are filtered out

        :param str text: The text to predict on
        :return list[Span]: A list of spans in the text
        """
        # Find all positive matches (allowing overlaps for now)
        positive_spans = []
        positive_spans.extend(
            self._match_spans(text, self.case_sensitive_positive_pattern)
        )
        positive_spans.extend(
            self._match_spans(text, self.case_insensitive_positive_pattern)
        )

        # Merge overlapping positive spans
        positive_spans = merge_overlapping_spans(positive_spans)

        # Find all negative matches (allowing overlaps for now)
        negative_spans = []
        negative_spans.extend(
            self._match_spans(text, self.case_sensitive_negative_pattern)
        )
        negative_spans.extend(
            self._match_spans(text, self.case_insensitive_negative_pattern)
        )

        # Merge overlapping negative spans
        negative_spans = merge_overlapping_spans(negative_spans)

        # Filter out positive matches that overlap with negative matches
        filtered_spans = [
            span
            for span in positive_spans
            if not any(span.overlaps(negative_span) for negative_span in negative_spans)
        ]

        return filtered_spans
