from src.classifier.classifier import Classifier
from src.classifier.embedding import EmbeddingClassifier
from src.classifier.keyword import KeywordClassifier
from src.classifier.rules_based import RulesBasedClassifier
from src.concept import Concept

__all__ = [
    "Classifier",
    "KeywordClassifier",
    "RulesBasedClassifier",
    "EmbeddingClassifier",
]


class ClassifierFactory:
    @staticmethod
    def create(concept: Concept) -> Classifier:
        """
        Create a classifier for a concept

        :param Concept concept: The concept to classify, with variable amounts of data
        :return BaseClassifier: The classifier for the concept, trained where applicable
        """
        if concept.negative_labels:
            return RulesBasedClassifier(concept)
        return KeywordClassifier(concept)
