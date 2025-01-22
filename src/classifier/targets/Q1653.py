from datetime import datetime
from typing import Optional

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

from src.classifier import Classifier
from src.concept import Concept
from src.span import Span


class Q1653(Classifier):
    """Net-zero target (Q1653) classifier"""

    def __init__(self, concept: Concept, threshold: float = 0.5):
        assert (
            concept.wikibase_id == "Q1653"
        ), 'Concept must be "net-zero target (Q1653)"'

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
        """Predict whether the supplied text contains a net-zero target."""
        threshold = threshold or self.threshold
        prediction = self.classifier(text, padding=True, truncation=True)

        if prediction[0]["label"] != "NZT" or prediction[0]["score"] < threshold:
            return []

        return [
            Span(
                text=text,
                start=0,
                end=len(text),
                concept_id=self.concept.wikibase_id,
                labellers=[str(self)],
                timestamps=[datetime.now()],
            )
        ]
