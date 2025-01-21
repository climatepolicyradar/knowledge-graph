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

    def __init__(self, concept: Concept, threshold: float = 0.65):
        super().__init__(concept)

        self.keyword_classifier = RulesBasedClassifier(concept=concept)

        justice = Concept(
            preferred_label="Justice",
            description="a concept which has justice-y vibes",
            definition="also injustice, and, like, when things aren't nice for workers",
        )
        self.embedding_classifier = EmbeddingClassifier(
            concept=justice,
            threshold=threshold,
        )

    def predict(self, text: str) -> list[Span]:
        """Predict whether the supplied text contains an instance of the concept."""
        keyword_predictions = self.keyword_classifier.predict(text)
        if keyword_predictions:
            embedding_prediction = self.embedding_classifier.predict(text)
            if embedding_prediction:
                return keyword_predictions

        return []
