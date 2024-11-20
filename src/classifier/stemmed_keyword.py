import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from src.classifier.rules_based import RulesBasedClassifier
from src.concept import Concept
from src.span import Span


class StemmedKeywordClassifier(RulesBasedClassifier):
    """
    Uses positive and negative keywords to find concepts in lightly-analysed text.

    This classifier is a modified version of the RulesBasedClassifier that uses stemmed
    keywords to match concepts in stemmed text. This allows for variations in word
    forms. For example, looking for the term "horse" should match "horse","horses",
    "horsing", "horsed", etc.
    """

    def __init__(self, concept: Concept):
        """
        Create a new RulesBasedClassifier instance.

        :param Concept concept: The concept which the classifier will identify in text
        """
        nltk.download("punkt", quiet=True)
        self.stemmer = PorterStemmer()
        self._original_concept = concept

        stemmed_concept = Concept(
            wikibase_id=concept.wikibase_id,
            # keep the preferred label intact for naming and hashing purposes
            preferred_label=concept.preferred_label,
            alternative_labels=[
                self._stem_label(label)
                for label in concept.alternative_labels + [concept.preferred_label]
            ],
            negative_labels=[
                self._stem_label(label) for label in concept.negative_labels
            ],
        )
        super().__init__(stemmed_concept)

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

        # For non-ASCII characters, keep the original token instead of stemming
        stemmed_tokens = []
        for token in tokens:
            if all(ord(char) < 128 for char in token):
                stemmed_tokens.append(self.stemmer.stem(token))
            else:
                stemmed_tokens.append(token)

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
                    pos_key = (
                        token_info[i]["original_start"],
                        token_info[i + match_len - 1]["original_end"],
                    )

                    if pos_key not in seen_positions:
                        seen_positions.add(pos_key)
                        result_spans.append(
                            Span(
                                text=text,
                                concept_id=stemmed_span.concept_id,
                                start_index=token_info[i]["original_start"],
                                end_index=token_info[i + match_len - 1]["original_end"],
                                labellers=[str(self)],
                            )
                        )

        return result_spans
