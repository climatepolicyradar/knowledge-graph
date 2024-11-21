import importlib

from src.classifier.classifier import Classifier
from src.classifier.keyword import KeywordClassifier
from src.classifier.rules_based import RulesBasedClassifier
from src.concept import Concept


def __getattr__(name):
    """Only import particular classifiers when they are actually requested"""
    if name == "EmbeddingClassifier":
        # This adds a special case for embeddings because they rely on very large external
        # libraries. Importing these libraries is very slow and having them installed
        # requires much more disc space, so we gave them a distinct group in the
        # pyproject.toml file (see: f53a404).
        module = importlib.import_module(".embedding", __package__)
        return getattr(module, name)
    if name == "StemmedKeywordClassifier":
        # For similar reasons, only import the stemmed keyword classifier and download
        # the nltk data when we actually request it.
        module = importlib.import_module(".stemmed_keyword", __package__)
        return getattr(module, name)
    else:
        return globals()[name]


__all__ = [
    "Classifier",
    "KeywordClassifier",
    "RulesBasedClassifier",
    "EmbeddingClassifier",
    "StemmedKeywordClassifier",
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
