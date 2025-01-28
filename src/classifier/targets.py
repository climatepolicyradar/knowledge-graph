from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable

from src.classifier import Classifier
from src.concept import Concept
from src.identifiers import WikibaseID
from src.span import Span

# optimal threshold for the "ClimatePolicyRadar/national-climate-targets" model as defined in
# https://github.com/climatepolicyradar/targets-sprint-cop28/blob/5c778d73cf4ca4c563fd9488d2cd29f824bc7dd7/src/config.py#L4
DEFAULT_THRESHOLD = 0.524


class BaseTargetClassifier(Classifier, ABC):
    """Base class for target classifiers."""

    allowed_concept_ids = [
        WikibaseID("Q1651"),
        WikibaseID("Q1652"),
        WikibaseID("Q1653"),
    ]
    model_name = "ClimatePolicyRadar/national-climate-targets"

    def __init__(
        self,
        concept: Concept,
    ):
        super().__init__(
            concept,
            allowed_concept_ids=self.allowed_concept_ids,
        )

        try:
            from transformers import (
                AutoModelForSequenceClassification,
                AutoTokenizer,
                pipeline,
            )
        except ImportError:
            raise ImportError(
                f"The `transformers` library is required to run {self.name}s. "
                "Install it with 'poetry install --with transformers'"
            )

        self.pipeline: Callable = pipeline(
            "text-classification",
            model=AutoModelForSequenceClassification.from_pretrained(self.model_name),
            tokenizer=AutoTokenizer.from_pretrained(self.model_name),
            function_to_apply="sigmoid",
        )

    @abstractmethod
    def _check_prediction_conditions(
        self, prediction: dict, threshold: float = DEFAULT_THRESHOLD
    ) -> bool:
        """Check whether the prediction meets the conditions for this target type."""

    def predict(self, text: str, threshold: float = DEFAULT_THRESHOLD) -> list[Span]:
        """Predict whether the supplied text contains a target."""
        return self.predict_batch([text], threshold=threshold)[0]

    def predict_batch(
        self, texts: list[str], threshold: float = DEFAULT_THRESHOLD
    ) -> list[list[Span]]:
        """Predict whether the supplied texts contain targets."""

        predictions: list[list[dict]] = self.pipeline(
            texts, padding=True, truncation=True, return_all_scores=True
        )

        results = []
        for text, prediction in zip(texts, predictions):
            text_results = []
            for labels in prediction:
                if self._check_prediction_conditions(labels, threshold):
                    span = Span(
                        text=text,
                        start_index=0,
                        end_index=len(text),
                        concept_id=self.concept.wikibase_id,
                        labellers=[str(self)],
                        timestamps=[datetime.now()],
                    )

                    if span not in text_results:
                        # this is needed so that for "target" we're not duplicating spans,
                        # but so that we're capturing everything for Reduction and NZT
                        text_results.append(span)

            results.append(text_results)
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

    def _check_prediction_conditions(
        self, prediction: dict, threshold: float = DEFAULT_THRESHOLD
    ) -> bool:
        """Check whether the prediction meets the conditions for a generic target."""
        return prediction["score"] >= threshold


class EmissionsReductionTargetClassifier(BaseTargetClassifier):
    """Emissions reduction target (Q1652) classifier"""

    allowed_concept_ids = [WikibaseID("Q1652")]

    def _check_prediction_conditions(
        self, prediction: dict, threshold: float = DEFAULT_THRESHOLD
    ) -> bool:
        """Check whether the prediction meets the conditions for a reduction target."""
        return (
            prediction["label"] in ["Reduction", "NZT"]
            and prediction["score"] >= threshold
        )


class NetZeroTargetClassifier(BaseTargetClassifier):
    """Net-zero target (Q1653) classifier"""

    allowed_concept_ids = [WikibaseID("Q1653")]

    def _check_prediction_conditions(
        self, prediction: dict, threshold: float = DEFAULT_THRESHOLD
    ) -> bool:
        """Check whether the prediction meets the conditions for a net-zero target."""
        return prediction["label"] == "NZT" and prediction["score"] >= threshold
