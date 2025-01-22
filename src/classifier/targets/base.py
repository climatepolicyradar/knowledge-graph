from abc import ABC, abstractmethod
from datetime import datetime
from typing import Callable, Optional

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from src.classifier import Classifier
from src.concept import Concept
from src.span import Span


class BaseTargetClassifier(Classifier, ABC):
    """Base class for target classifiers."""

    def __init__(self, concept: Concept, threshold: float = 0.5):
        super().__init__(concept)

        self.model_name = "ClimatePolicyRadar/national-climate-targets"
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.classifier: Callable = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            function_to_apply="sigmoid",
        )

        self.threshold = threshold

    @abstractmethod
    def _check_prediction_conditions(self, prediction: dict, threshold: float) -> bool:
        """Check if the prediction meets the specific conditions for this target type."""

    def predict(self, text: str, threshold: Optional[float] = None) -> list[Span]:
        """Predict whether the supplied text contains a target."""
        return self.predict_batch([text], threshold)[0]

    def predict_batch(
        self, texts: list[str], threshold: Optional[float] = None
    ) -> list[list[Span]]:
        """Predict whether the supplied texts contain targets."""
        threshold = threshold or self.threshold
        predictions = self.classifier(texts, padding=True, truncation=True)

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
