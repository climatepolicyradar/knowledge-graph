from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, Optional

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from src.classifier import Classifier
from src.concept import Concept
from src.identifiers import WikibaseID
from src.span import Span


class BaseTargetClassifier(Classifier, ABC):
    """Base class for target classifiers."""

    allowed_concept_ids = [
        WikibaseID("Q1651"),
        WikibaseID("Q1652"),
        WikibaseID("Q1653"),
    ]

    def __init__(self, concept: Concept, threshold: float = 0.5):
        super().__init__(
            concept, threshold=threshold, allowed_concept_ids=self.allowed_concept_ids
        )

        self.model_name = "ClimatePolicyRadar/national-climate-targets"

        self.pipeline: Callable = pipeline(
            "text-classification",
            model=AutoModelForSequenceClassification.from_pretrained(self.model_name),
            tokenizer=AutoTokenizer.from_pretrained(self.model_name),
            function_to_apply="sigmoid",
        )
        self.threshold = threshold

    @abstractmethod
    def _check_prediction_conditions(self, prediction: dict, threshold: float) -> bool:
        """Check whether the prediction meets the conditions for this target type."""

    def predict(self, text: str, threshold: Optional[float] = None) -> list[Span]:
        """Predict whether the supplied text contains a target."""
        return self.predict_batch([text], threshold)[0]

    def predict_batch(
        self, texts: list[str], threshold: Optional[float] = None
    ) -> list[list[Span]]:
        """Predict whether the supplied texts contain targets."""
        threshold = threshold or self.threshold
        predictions = self.pipeline(texts, padding=True, truncation=True)

        results = []
        for text, prediction in zip(texts, predictions):
            if not prediction or not self._check_prediction_conditions(
                prediction, threshold
            ):
                results.append([])
            else:
                results.append(
                    [
                        Span(
                            text=text,
                            start_index=0,
                            end_index=len(text),
                            concept_id=self.concept.wikibase_id,
                            labellers=[str(self)],
                            timestamps=[datetime.now()],
                        )
                    ]
                )

        return results

    def fit(self) -> "BaseTargetClassifier":
        """Targets classifiers cannot be trained directly."""
        raise NotImplementedError(
            "Targets classifiers in the knowledge graph are based on the pre-trained "
            f"{self.model_name} model. As such, they cannot be trained directly."
        )


class TargetClassifier(BaseTargetClassifier):
    """Target (Q1651) classifier"""

    allowed_concept_ids = [WikibaseID("Q1651")]

    def _check_prediction_conditions(self, prediction: dict, threshold: float) -> bool:
        """Check whether the prediction meets the conditions for a generic target."""
        return prediction["score"] >= threshold


class EmissionsReductionTargetClassifier(BaseTargetClassifier):
    """Emissions reduction target (Q1652) classifier"""

    allowed_concept_ids = [WikibaseID("Q1652")]

    def _check_prediction_conditions(self, prediction: dict, threshold: float) -> bool:
        """Check whether the prediction meets the conditions for a reduction target."""
        return (
            prediction["label"] in ["Reduction", "NZT"]
            and prediction["score"] >= threshold
        )


class NetZeroTargetClassifier(BaseTargetClassifier):
    """Net-zero target (Q1653) classifier"""

    allowed_concept_ids = [WikibaseID("Q1653")]

    def _check_prediction_conditions(self, prediction: dict, threshold: float) -> bool:
        """Check whether the prediction meets the conditions for a net-zero target."""
        return prediction["label"] == "NZT" and prediction["score"] >= threshold
