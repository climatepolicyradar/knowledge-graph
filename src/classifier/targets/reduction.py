from datetime import datetime
from typing import Optional

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from src.classifier import Classifier
from src.concept import Concept
from src.span import Span


class EmissionsReductionTargetClassifier(Classifier):
    """Emissions reduction target (Q1652) classifier"""

    def __init__(self, concept: Concept, threshold: float = 0.5):
        assert (
            concept.wikibase_id == "Q1652"
        ), 'Concept must be "emissions reduction target (Q1652)"'

        super().__init__(concept)

        self.model_name = "ClimatePolicyRadar/national-climate-targets"
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            function_to_apply="sigmoid",
        )

        self.threshold = threshold

    def predict(self, text: str, threshold: Optional[float] = None) -> list[Span]:
        """Predict whether the supplied text contains an emissions reduction target."""
        threshold = threshold or self.threshold
        prediction = self.classifier(text, padding=True, truncation=True)

        if (
            not prediction or
            prediction[0]["label"] not in ["Reduction", "NZT"]
            or prediction[0]["score"] < threshold
        ):
            return []

        return [
            Span(
                text=text,
                start_index=0,
                end_index=len(text),
                concept_id=self.concept.wikibase_id,
                labellers=[str(self)],
                timestamps=[datetime.now()],
            )
        ]
