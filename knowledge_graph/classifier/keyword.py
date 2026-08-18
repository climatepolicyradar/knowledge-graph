import re
from datetime import datetime

from knowledge_graph.classifier.classifier import Classifier, ZeroShotClassifier
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import ClassifierID
from knowledge_graph.span import Span, merge_overlapping_spans
from knowledge_graph.utils import get_logger

logger = get_logger(__name__)

# Strictly 1:1 character mappings, so folding preserves string length – so the classifier
# can operate over the modified text and report spans against the original
_NUMBER_FOLD_PAIRS: dict[str, str] = {
    # subscript digits ₀-₉ (U+2080-U+2089)
    **{chr(0x2080 + digit): str(digit) for digit in range(10)},
    # superscript digits
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    **{chr(0x2074 + digit - 4): str(digit) for digit in range(4, 10)},
    # sub/superscript signs
    "₊": "+",
    "⁺": "+",
    "₋": "-",
    "⁻": "-",
}

_NUMBER_FOLD_TABLE = str.maketrans(_NUMBER_FOLD_PAIRS)


def fold_subscript_characters(text: str) -> str:
    """
    Replace subscript and superscript digits and signs with their ASCII equivalents.

    The mapping is 1:1, so the returned string has exactly the same length as the input
    and indices into it are valid indices into the input.

    e.g. "CO₂" -> "CO2", "m³" -> "m3"

    :param str text: The text to fold
    :return str: The folded text, of identical length
    """
    return text.translate(_NUMBER_FOLD_TABLE)


_VOWELS = "aeiou"
_ES_SUFFIX_ENDINGS = ("s", "x", "z", "sh", "ch")

_S = "[sS]"
_ES = "[eE][sS]"
_IES = "[iI][eE][sS]"


def make_suffix_flexible_regex(word: str, minimum_word_length_chars=3) -> str:
    """
    Convert a word into a regex fragment which also matches its plural form.

    Only English plural inflection is handled, and only via suffix rules:

    - consonant + "y" -> "polic(?:y|ies)"
    - "s"/"x"/"z"/"sh"/"ch" -> "gas(?:es)?"
    - anything else -> "target[sS]?"

    Note: this widens singular labels to also match their plural, not the
    other way around, so an already-plural label gains nothing. Verb suffixes (-ing,
    -ed) and irregular plurals are deliberately not handled. Irregular forms belong in
    the concept's alternative labels.

    :param str word: The word to convert (unescaped)
    :return str: An escaped regex fragment
    """
    if len(word) < minimum_word_length_chars or word[-1].isdigit():
        return re.escape(word)

    lowered = word.lower()

    if lowered.endswith("y") and lowered[-2] not in _VOWELS:
        return f"{re.escape(word[:-1])}(?:{re.escape(word[-1])}|{_IES})"

    if lowered.endswith(_ES_SUFFIX_ENDINGS):
        return f"{re.escape(word)}(?:{_ES})?"

    return f"{re.escape(word)}{_S}?"


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

    Two optional relaxations of the matching are available, both off by default, and
    they can be combined:

    - fold_subscripts: matches subscript and superscript digits against their ASCII
      equivalents, in both directions, so a "CO2" label matches "CO₂" in the text and
      a "CO₂" label matches "CO2".
    - match_word_forms: also matches the English plural of a label's final word, so
      "greenhouse gas" matches "greenhouse gases" and "policy" matches "policies".
      Uses regex rules which are imprecise but quick.

    Both are applied to negative labels as well as positive ones, so that a loosened
    positive match cannot slip past a still-strict veto. Enabling either changes the
    classifier's id, so a relaxed variant never collides with the default
    configuration. Spans are always reported as offsets into the original text.

    KeywordClassifier does not output prediction probabilities, so spans identified by
    this classifier will not have prediction_probability values set.
    """

    fold_subscripts: bool = False
    match_word_forms: bool = False

    valid_separator_characters = [
        r"\-",  # hyphen (escaped because we're going to use it in a regex character class)
        "–",  # en dash
        "—",  # em dash
    ]
    separator_pattern = r"[\s" + "".join(valid_separator_characters) + r"]+"

    def __init__(
        self,
        concept: Concept,
        fold_subscripts: bool = False,
        match_word_forms: bool = False,
    ):
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
        :param bool fold_subscripts: Match subscript and superscript digits against
            their ASCII equivalents, in both directions
        :param bool match_word_forms: Also match the English plural of each label's
            final word
        """
        super().__init__(concept)

        self.fold_subscripts = fold_subscripts
        self.match_word_forms = match_word_forms

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
            parts = [
                part for part in re.split(self.separator_pattern, label.strip()) if part
            ]

            if self.match_word_forms and parts:
                word_parts = [re.escape(part) for part in parts[:-1]]
                word_parts.append(make_suffix_flexible_regex(parts[-1]))
            else:
                word_parts = [re.escape(part) for part in parts]

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

        positive_labels = self.concept.all_labels
        negative_labels = self.concept.negative_labels

        # Fold the labels before splitting them by case, so that a label like "co₂" is
        # recognised as pure-ASCII lowercase and matched case-insensitively. Doing this
        # the other way around would strand every folded label in the case-sensitive
        # bucket, because of its non-ASCII characters.
        if self.fold_subscripts:
            positive_labels = [
                fold_subscript_characters(label) for label in positive_labels
            ]
            negative_labels = [
                fold_subscript_characters(label) for label in negative_labels
            ]

        # Split labels by case sensitivity
        case_sensitive_positive, case_insensitive_positive = split_by_case_handling(
            positive_labels
        )
        case_sensitive_negative, case_insensitive_negative = split_by_case_handling(
            negative_labels
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
        """
        Return a deterministic, human-readable identifier for the classifier.

        The match options are only hashed in when they are enabled, so that default
        classifiers keep the ids they had before the introduction of these.
        """
        if not (self.fold_subscripts or self.match_word_forms):
            return ClassifierID.generate(self.name, self.concept.id)

        return ClassifierID.generate(
            self.name, self.concept.id, self.fold_subscripts, self.match_word_forms
        )

    def _match_spans(
        self, text: str, pattern: re.Pattern | None, search_text: str | None = None
    ) -> list[Span]:
        """
        Find spans in text using the provided pattern.

        :param str text: The text the returned spans will refer to
        :param re.Pattern | None pattern: The compiled regex pattern (can be None)
        :param str | None search_text: The text to actually search, if it differs from
            `text`. Must be the same length as `text`, so that match offsets remain
            valid offsets into `text`.
        :return list[Span]: List of spans found by the pattern
        """
        if not pattern:
            return []

        spans = []
        for match in pattern.finditer(search_text if search_text is not None else text):
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

    def _predict(self, text: str, threshold: float | None = None) -> list[Span]:
        """
        Predict whether the supplied text contains an instance of the concept.

        This method applies both case-sensitive and case-insensitive patterns to find
        matches. It ensures that:

        1. Longer matches take precedence over shorter ones
        2. No overlapping matches are returned
        3. Case-sensitive matches are found exactly as provided
        4. Case-insensitive matches can be found regardless of casing
        5. Positive matches that overlap with negative matches are filtered out
        6. Spans are always offsets into the supplied text, even when the text is
           folded before searching

        :param str text: The text to predict on
        :param float | None threshold: Optional prediction threshold. Logs a warning if
            used here; kept for API consistency.
        :return list[Span]: A list of spans in the text
        """
        if threshold is not None:
            logger.warning(
                "`threshold` parameter ignored - KeywordClassifier does not output "
                "prediction probabilities"
            )

        # Search the folded text, but report spans as offsets into the original text.
        # Folding is 1:1, so the two strings have identical lengths and the offsets are
        # interchangeable.
        search_text = fold_subscript_characters(text) if self.fold_subscripts else text

        # Find all positive matches (allowing overlaps for now)
        positive_spans = []
        positive_spans.extend(
            self._match_spans(text, self.case_sensitive_positive_pattern, search_text)
        )
        positive_spans.extend(
            self._match_spans(text, self.case_insensitive_positive_pattern, search_text)
        )

        # Merge overlapping positive spans
        positive_spans = merge_overlapping_spans(positive_spans)

        # Find all negative matches (allowing overlaps for now)
        negative_spans = []
        negative_spans.extend(
            self._match_spans(text, self.case_sensitive_negative_pattern, search_text)
        )
        negative_spans.extend(
            self._match_spans(text, self.case_insensitive_negative_pattern, search_text)
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
