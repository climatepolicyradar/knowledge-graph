from src.classifier.targets.base import BaseTargetClassifier
from src.concept import Concept


class TargetClassifier(BaseTargetClassifier):
    """Target (Q1651) classifier"""

    def __init__(self, concept: Concept, threshold: float = 0.5):
        assert concept.wikibase_id == "Q1651", 'Concept must be "target (Q1651)"'
        super().__init__(concept, threshold)

    def _check_prediction_conditions(self, prediction: dict, threshold: float) -> bool:
        """Check if the prediction meets the conditions for a generic target."""
        return prediction["score"] >= threshold
