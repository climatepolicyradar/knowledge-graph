from src.classifier.targets.base import BaseTargetClassifier
from src.concept import Concept


class NetZeroTargetClassifier(BaseTargetClassifier):
    """Net-zero target (Q1653) classifier"""

    def __init__(self, concept: Concept, threshold: float = 0.5):
        assert (
            concept.wikibase_id == "Q1653"
        ), 'Concept must be "net-zero target (Q1653)"'

        super().__init__(concept, threshold)

    def _check_prediction_conditions(self, prediction: dict, threshold: float) -> bool:
        """Check if the prediction meets the conditions for a net-zero target."""
        return prediction["label"] == "NZT" and prediction["score"] >= threshold
