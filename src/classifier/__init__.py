import importlib

from src.classifier.classifier import Classifier
from src.classifier.embedding import EmbeddingClassifier
from src.classifier.keyword import KeywordClassifier
from src.classifier.rules_based import RulesBasedClassifier
from src.classifier.stemmed_keyword import StemmedKeywordClassifier
from src.classifier.targets import (
    EmissionsReductionTargetClassifier,
    NetZeroTargetClassifier,
    TargetClassifier,
)
from src.concept import Concept

__all__ = [
    "Classifier",
    "KeywordClassifier",
    "RulesBasedClassifier",
    "EmbeddingClassifier",
    "StemmedKeywordClassifier",
    "EmissionsReductionTargetClassifier",
    "NetZeroTargetClassifier",
    "TargetClassifier",
]


class ClassifierFactory:
    @staticmethod
    def create(concept: Concept) -> Classifier:
        """
        Create a classifier for a concept

        :param Concept concept: The concept to classify, with variable amounts of data
        :return BaseClassifier: The classifier for the concept, trained where applicable
        """
        if concept.wikibase_id == "Q1651":
            return TargetClassifier(concept)
        elif concept.wikibase_id == "Q1652":
            return EmissionsReductionTargetClassifier(concept)
        elif concept.wikibase_id == "Q1653":
            return NetZeroTargetClassifier(concept)
        elif concept.negative_labels:
            return RulesBasedClassifier(concept)
        return KeywordClassifier(concept)
