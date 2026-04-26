"""
Currency classifier using keywords appearing closely together.

Matches currency labels in close proximity to numbers
(digits or number words), with tight/loose proximity and single-char alphanumeric
handling. Same output shape as KeywordClassifier.
"""

import logging
import re

from knowledge_graph.classifier.keyword import KeywordClassifier
from knowledge_graph.concept import Concept

logger = logging.getLogger(__name__)

# Number words for currency-amount matching (singular and plural for "hundreds of USD" etc.)
NUMBER_WORDS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "hundreds",
    "thousand",
    "thousands",
    "million",
    "millions",
    "billion",
    "billions",
    "trillion",
    "trillions",
]
# Longest first for regex alternation so "hundreds" matches before "hundred"
NUMBER_WORDS_SORTED = sorted(NUMBER_WORDS, key=len, reverse=True)
NUMBER_WORDS_ALTERNATION = "|".join(re.escape(w) for w in NUMBER_WORDS_SORTED)

# Digits with optional decimal (comma or period): e.g. 100, 1.5, 1,5
DIGITS_PATTERN = r"\d+(?:[.,]\d+)?"
# Number words with word boundaries
NUMBER_WORDS_PATTERN = r"(?<!\w)(?:" + NUMBER_WORDS_ALTERNATION + r")(?!\w)"
# Combined number pattern (digits or number words)
NUMBER_PATTERN = r"(?:" + DIGITS_PATTERN + r"|" + NUMBER_WORDS_PATTERN + r")"

# Tight: currency and number with 0-1 chars between
TIGHT_CURRENCY_THEN_NUMBER = r"(?:{currency}).{{0,1}}(?:{number})"
TIGHT_NUMBER_THEN_CURRENCY = r"(?:{number}).{{0,1}}(?:{currency})"
# Loose: currency and number with 0-4 words between
LOOSE_CURRENCY_THEN_NUMBER = r"(?:{currency})\s+(?:\S+\s+){{0,4}}(?:{number})"
LOOSE_NUMBER_THEN_CURRENCY = r"(?:{number})\s+(?:\S+\s+){{0,4}}(?:{currency})"


def _make_separator_flexible(label: str) -> str:
    """Convert a label to a regex pattern with flexible separators (same as KeywordClassifier)."""
    parts = re.split(KeywordClassifier.separator_pattern, label.strip())
    word_parts = [re.escape(part) for part in parts if part]
    if len(word_parts) == 1:
        return word_parts[0]
    return KeywordClassifier.separator_pattern.join(word_parts)


def _split_by_case_handling(labels: list[str]) -> tuple[list[str], list[str]]:
    """Partition labels into case-sensitive and case-insensitive (same as KeywordClassifier)."""
    case_sensitive_labels = []
    case_insensitive_labels = []
    sorted_labels = sorted(labels, key=len, reverse=True)
    for label in sorted_labels:
        if not label.strip():
            continue
        if any(c.isupper() for c in label) or any(ord(c) > 127 for c in label):
            case_sensitive_labels.append(label)
        else:
            case_insensitive_labels.append(label)
    return case_sensitive_labels, case_insensitive_labels


def _is_single_char_alphanumeric(label: str) -> bool:
    """True if label is exactly one character and alphanumeric (e.g. 'R' for Rand)."""
    return len(label) == 1 and label.isalnum()


def _build_currency_number_pattern(
    currency_labels: list[str],
    case_sensitive: bool,
) -> re.Pattern | None:
    """
    Build and compile a regex pattern.

    Build a single compiled pattern that matches currency + number or number + currency,
    with tight and/or loose proximity. Single-char alphanumeric labels get tight only.
    """

    if not currency_labels:
        return None

    # Prepare currency regex parts (escaped, optional separator flexibility)
    tight_only_alternation_parts: list[str] = []
    tight_and_loose_alternation_parts: list[str] = []

    for label in currency_labels:
        escaped = _make_separator_flexible(label)
        if _is_single_char_alphanumeric(label):
            tight_only_alternation_parts.append(escaped)
        else:
            tight_and_loose_alternation_parts.append(escaped)

    alternatives: list[str] = []

    if tight_only_alternation_parts:
        tight_only = "|".join(tight_only_alternation_parts)
        alternatives.append(
            TIGHT_CURRENCY_THEN_NUMBER.format(
                currency=tight_only, number=NUMBER_PATTERN
            )
        )
        alternatives.append(
            TIGHT_NUMBER_THEN_CURRENCY.format(
                number=NUMBER_PATTERN, currency=tight_only
            )
        )

    if tight_and_loose_alternation_parts:
        tight_and_loose = "|".join(tight_and_loose_alternation_parts)
        # Tight
        alternatives.append(
            TIGHT_CURRENCY_THEN_NUMBER.format(
                currency=tight_and_loose, number=NUMBER_PATTERN
            )
        )
        alternatives.append(
            TIGHT_NUMBER_THEN_CURRENCY.format(
                number=NUMBER_PATTERN, currency=tight_and_loose
            )
        )
        # Loose
        alternatives.append(
            LOOSE_CURRENCY_THEN_NUMBER.format(
                currency=tight_and_loose, number=NUMBER_PATTERN
            )
        )
        alternatives.append(
            LOOSE_NUMBER_THEN_CURRENCY.format(
                number=NUMBER_PATTERN, currency=tight_and_loose
            )
        )

    if not alternatives:
        return None

    pattern_str = "(?:" + "|".join(alternatives) + ")"
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(pattern_str, flags)


class CurrencyKeywordClassifier(KeywordClassifier):
    """
    Classifier that matches currency synonyms in close proximity to numbers

    Numbers can be digits or number words like "hundred", "million").
    Same output shape as KeywordClassifier (list[Span]).
    Uses tight and loose proximity patterns to prevent false positives.
    single-character alphanumeric labels (e.g. "R" for Rand) use tight-only,
    meaning it only matches when the currency label is immediately followed by a number.
    Capitalisation behaviour matches KeywordClassifier.
    """

    def __init__(self, concept: Concept):
        # Build negative patterns and (temporary) positive patterns via parent
        super().__init__(concept)

        # Replace positive patterns with currency-number proximity patterns
        case_sensitive_positive, case_insensitive_positive = _split_by_case_handling(
            self.concept.all_labels
        )

        self.case_sensitive_positive_pattern = _build_currency_number_pattern(
            case_sensitive_positive, case_sensitive=True
        )
        self.case_insensitive_positive_pattern = _build_currency_number_pattern(
            case_insensitive_positive, case_sensitive=False
        )
