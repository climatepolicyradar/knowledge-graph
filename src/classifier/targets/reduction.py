from src.classifier.targets.base import BaseTargetClassifier
from src.concept import Concept
from src.identifiers import WikibaseID


class EmissionsReductionTargetClassifier(BaseTargetClassifier):
    """Emissions reduction target (Q1652) classifier"""

    def __init__(self, concept: Concept, threshold: float = 0.5):
        self._check_whether_supplied_concept_is_correct_for_this_classifier(
            expected_wikibase_id=WikibaseID("Q1652"), supplied_concept=concept
        )
        super().__init__(concept, threshold)

    def _check_prediction_conditions(self, prediction: dict, threshold: float) -> bool:
        """Check if the prediction meets the conditions for an emissions reduction target."""
        return (
            prediction["label"] in ["Reduction", "NZT"]
            and prediction["score"] >= threshold
        )
