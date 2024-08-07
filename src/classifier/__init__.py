from src.classifier.classifier import Classifier, Span
from src.classifier.keyword import KeywordClassifier
from src.concept import Concept

__all__ = [
    "Classifier",
    "KeywordClassifier",
]


class ClassifierFactory:
    @staticmethod
    def create(concept: Concept) -> Classifier:
        """
        Create a classifier for a concept

        :param Concept concept: The concept to classify, with variable amounts of data
        :return BaseClassifier: The classifier for the concept, trained where applicable
        """
        return KeywordClassifier(concept)
