import warnings
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
    commit_hash = "c920e288551f415e0085f89475d9acbb9969cfb8"

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
            model=AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                revision=self.commit_hash,
            ),
            tokenizer=AutoTokenizer.from_pretrained(
                self.model_name,
                revision=self.commit_hash,
            ),
            function_to_apply="sigmoid",
            device="cpu",
        )

    @abstractmethod
    def _check_prediction_conditions(
        self, prediction: list[dict], threshold: float = DEFAULT_THRESHOLD
    ) -> bool:
        """
        Check whether the prediction meets the conditions for this target type.

        This is necessary, because we're not only using a multilabel model for prediction here,
        but the 3 output labels are also in a hierarchical relationship with in the concept
        store. This means, that each of the "target", "net-zero" and "emission reduction" classifiers
        have custom logic implemented here to transform the model predictions into a binary on the concept.

        Args:
            prediction: The prediction returned by the model -- a list of dictionaries for each of
                the 3 labels, containing the keys "label" and "score"
            threshold: The threshold above which a prediction is considered positive.

        Returns:
            bool: Whether the prediction meets the conditions for this target type.
        """

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
            if self._check_prediction_conditions(prediction, threshold):
                span = Span(
                    text=text,
                    start_index=0,
                    end_index=len(text),
                    concept_id=self.concept.wikibase_id,
                    labellers=[str(self)],
                    timestamps=[datetime.now()],
                )
                text_results.append(span)

            results.append(text_results)
        return results

    def fit(self) -> "BaseTargetClassifier":
        """Targets classifiers cannot be trained directly."""
        warnings.warn(
            "Targets classifiers in the knowledge graph are based on the pre-trained "
            f"{self.model_name} model. As such, they cannot be trained directly."
        )
        return self


class TargetClassifier(BaseTargetClassifier):
    """Target (Q1651) classifier"""

    allowed_concept_ids = [WikibaseID("Q1651")]

    def _check_prediction_conditions(
        self, prediction: list[dict], threshold: float = DEFAULT_THRESHOLD
    ) -> bool:
        """Check whether the prediction meets the conditions for a generic target."""
        return any(label["score"] >= threshold for label in prediction)


class EmissionsReductionTargetClassifier(BaseTargetClassifier):
    """Emissions reduction target (Q1652) classifier"""

    allowed_concept_ids = [WikibaseID("Q1652")]

    def _check_prediction_conditions(
        self, prediction: list[dict], threshold: float = DEFAULT_THRESHOLD
    ) -> bool:
        """Check whether the prediction meets the conditions for a reduction target."""
        return any(
            label["score"] >= threshold and label["label"] in {"NZT", "Reduction"}
            for label in prediction
        )


class NetZeroTargetClassifier(BaseTargetClassifier):
    """Net-zero target (Q1653) classifier"""

    allowed_concept_ids = [WikibaseID("Q1653")]

    def _check_prediction_conditions(
        self, prediction: list[dict], threshold: float = DEFAULT_THRESHOLD
    ) -> bool:
        """Check whether the prediction meets the conditions for a net-zero target."""
        return any(
            label["score"] >= threshold and label["label"] in {"NZT"}
            for label in prediction
        )
