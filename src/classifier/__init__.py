import importlib

from src.classifier.classifier import Classifier
from src.classifier.keyword import KeywordClassifier
from src.classifier.rules_based import RulesBasedClassifier
from src.concept import Concept
from src.identifiers import WikibaseID


def __getattr__(name):
    """Only import particular classifiers when they are actually requested"""
    if name == "EmbeddingClassifier":
        # This adds a special case for embeddings because they rely on very large external
        # libraries. Importing these libraries is very slow and having them installed
        # requires much more disc space, so we gave them a distinct group in the
        # pyproject.toml file (see: f53a404).
        module = importlib.import_module(".embedding", __package__)
        return getattr(module, name)
    elif name == "StemmedKeywordClassifier":
        # For similar reasons, only import the stemmed keyword classifier and download
        # the nltk data when we actually request it.
        module = importlib.import_module(".stemmed_keyword", __package__)
        return getattr(module, name)
    elif name in (
        "EmissionsReductionTargetClassifier",
        "NetZeroTargetClassifier",
        "TargetClassifier",
    ):
        # As above, only import bespoke targets classifiers when specifically requested
        module = importlib.import_module(".targets", __package__)
        return getattr(module, name)
    else:
        return globals()[name]


__all__ = [
    "Classifier",
    "KeywordClassifier",
    "RulesBasedClassifier",
    "EmbeddingClassifier",  # type: ignore
    "StemmedKeywordClassifier",  # type: ignore
    "EmissionsReductionTargetClassifier",  # type: ignore
    "NetZeroTargetClassifier",  # type: ignore
    "TargetClassifier",  # type: ignore
]


class ClassifierFactory:
    # Map of Wikibase IDs to their bespoke classifier classes
    bespoke_classifier_map: dict[WikibaseID, tuple[str, str]] = {
        WikibaseID("Q1651"): ("TargetClassifier", ".targets"),
        WikibaseID("Q1652"): ("EmissionsReductionTargetClassifier", ".targets"),
        WikibaseID("Q1653"): ("NetZeroTargetClassifier", ".targets"),
    }

    @staticmethod
    def create(concept: Concept) -> Classifier:
        """Create a classifier for a concept, depending on its attributes"""
        # First, check whether we have a bespoke classifier for the concept
        if concept.wikibase_id in ClassifierFactory.bespoke_classifier_map:
            name, module_path = ClassifierFactory.bespoke_classifier_map[
                concept.wikibase_id
            ]
            module = importlib.import_module(module_path, __package__)
            classifier_class = getattr(module, name)
            return classifier_class(concept)

        # Then handle more generic cases
        if concept.negative_labels:
            return RulesBasedClassifier(concept)

        return KeywordClassifier(concept)
