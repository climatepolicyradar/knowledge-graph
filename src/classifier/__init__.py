import importlib

from src.classifier.classifier import Classifier
from src.classifier.keyword import KeywordClassifier
from src.classifier.rules_based import RulesBasedClassifier
from src.concept import Concept


def __getattr__(name):
    """
    Only import embeddings when we want it

    This adds a special case for embeddings because they rely on very large external
    libraries. Importing these libraries is very slow and having them installed
    requires much more disc space, so we gave them a distinct group in the
    pyproject.toml file (see: f53a404).

    This makes it so we only import them when we explicitly want to. So the following
    import statments will both work, but only the second requires the embeddings group
    to have been installed with poetry:

    ```
    from src.classifier import Classifier
    from src.classifier import EmbeddingClassifier
    ```
    """
    if name == "EmbeddingClassifier":
        module = importlib.import_module(".embedding", __package__)
        return getattr(module, name)
    else:
        return globals()[name]


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
