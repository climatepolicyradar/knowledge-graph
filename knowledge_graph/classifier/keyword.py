import re
from datetime import datetime

from knowledge_graph.classifier.classifier import Classifier, ZeroShotClassifier
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import ClassifierID
from knowledge_graph.span import Span, merge_overlapping_spans


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

    valid_separator_characters = [
        r"\-",  # hyphen (escaped because we're going to use it in a regex character class)
        "–",  # en dash
        "—",  # em dash
    ]
    separator_pattern = r"[\s" + "".join(valid_separator_characters) + r"]+"

    def __init__(self, concept: Concept):
        r"""
        Create a new KeywordClassifier instance.

        During initialization, concept labels undergo transformation:
        1. Plain string labels (e.g., "greenhouse gas") are split by case sensitivity
        2. Each label is transformed into a regex string with flexible separators
           (e.g., "greenhouse gas" becomes "greenhouse[\s\-–—]+gas")
        3. Transformed strings are then compiled into case-sensitive/insensitive regex
           patterns

        The stored label attributes (case_sensitive_positive_labels, etc.) contain
        regex strings, NOT the original plain labels. If you need the original labels,
        you can access them via the classifier.concept.all_labels and
        classifier.concept.negative_labels attributes.

        :param Concept concept: The concept which the classifier will identify in text
        """
        super().__init__(concept)

        def make_separator_flexible(label: str) -> str:
            r"""
            Convert a label to a regex pattern that matches different word separators.

            This allows labels like "greenhouse gas" to match:
            - "greenhouse gas" (space)
            - "greenhouse-gas" (hyphen)
            - "greenhouse\ngas" (newline)
            - "greenhouse -gas" (multiple consecutive separators)

            :param str label: The label to convert
            :return str: A regex pattern string that matches the label with flexible separators
            """
            # Split by any common separator characters (space, hyphen, newline, etc.)
            parts = re.split(self.separator_pattern, label.strip())

            # Filter out empty parts and escape each word part
            word_parts = [re.escape(part) for part in parts if part]

            # If the label has no separators, return the escaped label as-is
            if len(word_parts) == 1:
                return word_parts[0]

            # Join parts of the label using the separator pattern
            return self.separator_pattern.join(word_parts)

        def create_pattern(
            labels: list[str], case_sensitive: bool = False
        ) -> re.Pattern | None:
            r"""
            Create a regex pattern from a list of labels.

            Args:
                labels: List of regex pattern strings (e.g., "greenhouse[\s\-–—]+gas").
                        Note: These are not plain labels - they've been transformed by
                        make_separator_flexible() to include flexible separator matching.
                case_sensitive: Whether to use case-sensitive matching.

            Returns:
                Compiled regex pattern with word boundaries, or None if labels is empty.
            """
            if not labels:
                return None

            pattern = r"(?<!\w)(?:" + "|".join(labels) + r")(?!\w)"
            flags = re.IGNORECASE if not case_sensitive else 0
            return re.compile(pattern, flags)

        def split_by_case_handling(labels: list[str]) -> tuple[list[str], list[str]]:
            """
            Partition labels into case-sensitive and case-insensitive lists.

            Returns the original labels, sorted by length (longest first).
            """
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
                        case_sensitive_labels.append(label)
                    else:
                        # Only pure ASCII lowercase labels are added to the case-insensitive list
                        case_insensitive_labels.append(label)

            return case_sensitive_labels, case_insensitive_labels

        # Split labels by case sensitivity
        case_sensitive_positive, case_insensitive_positive = split_by_case_handling(
            self.concept.all_labels
        )
        case_sensitive_negative, case_insensitive_negative = split_by_case_handling(
            self.concept.negative_labels
        )

        # Apply separator flexibility to create regex patterns
        self.case_sensitive_positive_labels = [
            make_separator_flexible(label) for label in case_sensitive_positive
        ]
        self.case_insensitive_positive_labels = [
            make_separator_flexible(label) for label in case_insensitive_positive
        ]
        self.case_sensitive_negative_labels = [
            make_separator_flexible(label) for label in case_sensitive_negative
        ]
        self.case_insensitive_negative_labels = [
            make_separator_flexible(label) for label in case_insensitive_negative
        ]

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

    def _predict(self, text: str) -> list[Span]:
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
