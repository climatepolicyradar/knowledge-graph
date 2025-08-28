from datetime import datetime

import nltk  # type: ignore
from nltk.stem import PorterStemmer  # type: ignore
from nltk.tokenize import word_tokenize  # type: ignore

from src.classifier.keyword import KeywordClassifier
from src.concept import Concept
from src.identifiers import ClassifierID
from src.span import Span


class StemmedKeywordClassifier(KeywordClassifier):
    """
    Uses positive and negative keywords to find concepts in lightly-analysed text.

    This classifier is a modified version of the KeywordClassifier that uses stemmed
    keywords to match concepts in stemmed text, allowing for variations in word forms.
    For example, looking for the term "horse" will match "horse","horses", "horsing",
    "horsed", etc.
    """

    def __init__(self, concept: Concept):
        """
        Create a new StemmedKeywordClassifier instance.

        :param Concept concept: The concept which the classifier will identify in text
        """
        # Ensure NLTK data is available
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

        self.stemmer = PorterStemmer()

        positive_labels = list(
            set([self._stem_label(label) for label in concept.all_labels])
        )
        # Stem negative labels, but keep them in their original form if they match a
        # stemmed positive label, eg we would include an unstemmed "horsing" in the
        # negative labels if there was a positive label for "horse", as both have the
        # stemmed form "hors"
        negative_labels = list(
            set(
                [
                    (
                        self._stem_label(label)
                        if self._stem_label(label) not in positive_labels
                        else label
                    )
                    for label in concept.negative_labels
                ]
            )
        )
        self.stemmed_concept = Concept(
            wikibase_id=concept.wikibase_id,
            preferred_label=concept.preferred_label,
            alternative_labels=positive_labels,
            negative_labels=negative_labels,
        )

        # initialise the parent KeywordClassifier with the stemmed concept so that it
        # uses the stemmed forms of the labels in its regex patterns
        super().__init__(self.stemmed_concept)

        # override the classifier's internal .concept so that its id, name, hash etc are
        # based on the original concept
        self.concept = concept

    @property
    def id(self) -> ClassifierID:
        """Return a deterministic, human-readable identifier for the classifier."""
        return ClassifierID.generate(self.name, self.concept.id)

    def _stem_label(self, label: str) -> str:
        """
        Convert a label into its stemmed form.

        :param str label: Original label
        :return str: Stemmed version of the label
        """
        return " ".join(token["stemmed_token"] for token in self.stem_text(label))

    def stem_text(self, text: str) -> list[dict]:
        """
        Stem the input text using NLTK's Porter Stemmer.

        :param str text: The text to stem
        :return list[dict]: A list of dictionaries containing the original and stemmed
            tokens and their positions in the text
        """
        tokens = word_tokenize(text)

        # For numbers or non-ASCII characters, keep the original token instead of stemming
        stemmed_tokens = [
            (
                token
                if token.isdigit() or not all(ord(char) < 128 for char in token)
                else self.stemmer.stem(token)
            )
            for token in tokens
        ]

        token_info = []
        position = 0
        stemmed_position = 0

        for token, stemmed_token in zip(tokens, stemmed_tokens):
            while position < len(text) and text[position].isspace():
                position += 1

            token_info.append(
                {
                    "original_token": token,
                    "stemmed_token": stemmed_token,
                    "original_start": position,
                    "original_end": position + len(token),
                    "stemmed_start": stemmed_position,
                    "stemmed_end": stemmed_position + len(stemmed_token),
                }
            )

            position += len(token)
            stemmed_position += len(stemmed_token) + 1

        return token_info

    def predict(self, text: str) -> list[Span]:
        """
        Predict whether the concept is present in the text.

        :param str text: The text to analyse
        :return list[Span]: A list of spans in the text
        """
        token_info = self.stem_text(text)
        stemmed_text = " ".join(token["stemmed_token"] for token in token_info)
        stemmed_spans = super().predict(stemmed_text)

        seen_positions = set()
        result_spans = []
        stemmed_tokens = stemmed_text.split()

        for stemmed_span in stemmed_spans:
            stemmed_substring = stemmed_text[
                stemmed_span.start_index : stemmed_span.end_index
            ]
            stemmed_match_tokens = stemmed_substring.split()
            match_len = len(stemmed_match_tokens)

            for i in range(len(stemmed_tokens) - match_len + 1):
                if stemmed_tokens[i : i + match_len] == stemmed_match_tokens:
                    position_key = (
                        token_info[i]["original_start"],
                        token_info[i + match_len - 1]["original_end"],
                    )

                    if position_key not in seen_positions:
                        seen_positions.add(position_key)
                        result_spans.append(
                            Span(
                                text=text,
                                concept_id=stemmed_span.concept_id,
                                start_index=token_info[i]["original_start"],
                                end_index=token_info[i + match_len - 1]["original_end"],
                                labellers=[str(self)],
                                timestamps=[datetime.now()],
                            )
                        )

        # Finally, filter spans for any negative labels. Because we match on the stemmed
        # version of the input text, we need to filter out any negative labels that match
        # the stemmed version of the positive labels. For example, if we're given a
        # concept like:
        #   Concept(preferred_label="horse", negative_labels=["horsing"])
        # we want to filter out any spans containing the word "Horsing" which would be
        # treated as a positive match by the regex for stemmed "horse".

        # Split negative labels into case sensitive and insensitive lists
        case_sensitive_negative_labels = {
            label
            for label in self.stemmed_concept.negative_labels
            if any(c.isupper() for c in label)
        }
        case_insensitive_negative_labels = {
            label.lower()
            for label in self.stemmed_concept.negative_labels
            if not any(c.isupper() for c in label)
        }

        # Filter spans for any negative labels
        filtered_spans = [
            span
            for span in result_spans
            if not any(
                token == label
                for label in case_sensitive_negative_labels
                for token in span.labelled_text.split()
            )
            and not any(
                token == label.lower()
                for label in case_insensitive_negative_labels
                for token in span.labelled_text.lower().split()
            )
        ]

        return filtered_spans
