from src.classifier.targets.base import BaseTargetClassifier
from src.concept import Concept


class EmissionsReductionTargetClassifier(BaseTargetClassifier):
    """Emissions reduction target (Q1652) classifier"""

    def __init__(self, concept: Concept, threshold: float = 0.5):
        assert (
            concept.wikibase_id == "Q1652"
        ), 'Concept must be "emissions reduction target (Q1652)"'
        super().__init__(concept, threshold)

    def _check_prediction_conditions(self, prediction: dict, threshold: float) -> bool:
        """Check if the prediction meets the conditions for an emissions reduction target."""
        return (
            prediction["label"] in ["Reduction", "NZT"]
            and prediction["score"] >= threshold
        )
