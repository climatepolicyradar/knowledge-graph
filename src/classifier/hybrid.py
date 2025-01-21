from src.classifier import Classifier
from src.classifier.embedding import EmbeddingClassifier
from src.classifier.rules_based import RulesBasedClassifier
from src.concept import Concept
from src.span import Span


class MagicSpecialCustomHybridClassifier(Classifier):
    """
    Hybrid classifier for the impacted workers concept

    Uses a keyword classifier to find matches, but also requires that the passage
    surpasses a threshold similarity to a justice-y concept to emphasise the "impacted"
    aspect of the concept
    """

    def __init__(
        self,
        concept: Concept,
        vibey_concepts: list[Concept],
        threshold: float = 0.675,
    ):
        super().__init__(concept)

        self.keyword_classifier = RulesBasedClassifier(concept=concept)

        self.embedding_classifiers = [
            EmbeddingClassifier(concept=concept, threshold=threshold)
            for concept in vibey_concepts
        ]

    def predict(self, text: str, threshold: float = 0.675) -> list[Span]:
        """Predict whether the supplied text contains an instance of the concept."""
        keyword_predictions = self.keyword_classifier.predict(text)
        if keyword_predictions:
            embedding_predictions = [
                classifier.predict(text, threshold)
                for classifier in self.embedding_classifiers
            ]
            if any(embedding_predictions):
                return keyword_predictions

        return []
