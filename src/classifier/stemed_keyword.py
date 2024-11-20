from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from src.classifier.rules_based import RulesBasedClassifier
from src.concept import Concept


class StemmedKeywordClassifier(RulesBasedClassifier):
    """
    Uses positive and negative keywords to find concepts in lightly-analysed text.

    This classifier is a modified version of the RulesBasedClassifier that uses stemmed
    keywords to match concepts in stemmed text. This allows for variations in word
    forms. For example, the term "virus" should match "virus", "viruses", "viral", etc.
    """

    def __init__(self, concept: Concept):
        """
        Create a new RulesBasedClassifier instance.

        :param Concept concept: The concept which the classifier will identify in text
        """
        self.stemmer = PorterStemmer()

        # before we create the regexes, we stem the input keywords
        concept.preferred_label = self.stem_text(concept.preferred_label)
        concept.alternative_labels = list(
            set([self.stem_text(label) for label in concept.alternative_labels])
        )
        concept.negative_labels = list(
            set([self.stem_text(label) for label in concept.negative_labels])
        )

        super().__init__(concept)

    def stem_text(self, text: str) -> str:
        """
        Stem the input text using the Porter Stemmer.

        :param str text: The text to stem
        :return str: The stemmed text
        """
        return " ".join(self.stemmer.stem(word) for word in word_tokenize(text))

    def predict(self, text):
        """
        Predict whether the concept is present in the text.

        :param str text: The text to analyse
        :return bool: Whether the concept is present in the text
        """
        text = self.stem_text(text)
        return super().predict(text)
