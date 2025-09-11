import warnings
from abc import abstractmethod
from datetime import datetime
from typing import Callable

from src.classifier.classifier import (
    Classifier,
    GPUBoundClassifier,
    ProbabilityCapableClassifier,
)
from src.concept import Concept
from src.identifiers import ClassifierID, WikibaseID
from src.span import Span

# optimal threshold for the "ClimatePolicyRadar/national-climate-targets" model as defined in
# https://github.com/climatepolicyradar/targets-sprint-cop28/blob/5c778d73cf4ca4c563fd9488d2cd29f824bc7dd7/src/config.py#L4
DEFAULT_THRESHOLD = 0.524


class BaseTargetClassifier(
    Classifier, GPUBoundClassifier, ProbabilityCapableClassifier
):
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
            from transformers import (  # type: ignore[import-untyped]
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
            from transformers.pipelines import pipeline  # type: ignore[import-untyped]
        except ImportError:
            raise ImportError(
                f"The `transformers` library is required to run {self.name}s. "
                "Install it with 'uv install --extra transformers'"
            )

        self.pipeline: Callable = pipeline(
            "text-classification",
            model=AutoModelForSequenceClassification.from_pretrained(
                self.model_name, revision=self.commit_hash
            ),
            tokenizer=AutoTokenizer.from_pretrained(
                self.model_name,
                revision=self.commit_hash,
            ),
            function_to_apply="sigmoid",
            device="cpu",
        )

    @property
    def id(self) -> ClassifierID:
        """Return a deterministic, human-readable identifier for the classifier."""
        return ClassifierID.generate(
            self.name, self.concept.id, self.model_name, self.commit_hash
        )

    @abstractmethod
    def _get_score(self, prediction: list[dict]) -> float:
        """
        Get a score for the target type.

        :param list[dict] prediction: the prediction returned by the model -- a list of
            dictionaries for each of the 3 labels, containing the keys "label" and "score"
        :return PredictionProbability: a score for the target type
        """

    def predict(self, text: str, threshold: float = DEFAULT_THRESHOLD) -> list[Span]:
        """Predict whether the supplied text contains a target."""
        return self.predict_batch([text], threshold=threshold)[0]

    def predict_batch(
        self, texts: list[str], threshold: float = DEFAULT_THRESHOLD
    ) -> list[list[Span]]:
        """Predict whether the supplied texts contain targets."""

        predictions: list[list[dict]] = self.pipeline(
            texts, padding=True, truncation=True, top_k=None
        )

        results = []
        for text, prediction in zip(texts, predictions):
            prediction_probability = self._get_score(prediction)
            if prediction_probability >= threshold:
                spans = [
                    Span(
                        text=text,
                        start_index=0,
                        end_index=len(text),
                        prediction_probability=prediction_probability,
                        concept_id=self.concept.wikibase_id,
                        labellers=[str(self)],
                        timestamps=[datetime.now()],
                    )
                ]
            else:
                spans = []

            results.append(spans)
        return results

    def fit(self, **kwargs) -> "BaseTargetClassifier":
        """Targets classifiers cannot be trained directly."""
        warnings.warn(
            "Targets classifiers in the knowledge graph are based on the pre-trained "
            f"{self.model_name} model. As such, they cannot be trained directly."
        )
        return self


class TargetClassifier(BaseTargetClassifier):
    """Target (Q1651) classifier"""

    allowed_concept_ids = [WikibaseID("Q1651")]

    def _get_score(self, prediction: list[dict]) -> float:
        """
        Get the score for an emissions reduction target.

        This is the maximum score any target type.
        """

        return max(label["score"] for label in prediction)


class EmissionsReductionTargetClassifier(BaseTargetClassifier):
    """Emissions reduction target (Q1652) classifier"""

    allowed_concept_ids = [WikibaseID("Q1652")]

    def _get_score(self, prediction: list[dict]) -> float:
        """
        Get the score for an emissions reduction target.

        This is the maximum score of types net-zero and reduction.
        """

        return max(
            label["score"]
            for label in prediction
            if label["label"] in {"NZT", "Reduction"}
        )


class NetZeroTargetClassifier(BaseTargetClassifier):
    """Net-zero target (Q1653) classifier"""

    allowed_concept_ids = [WikibaseID("Q1653")]

    def _get_score(self, prediction: list[dict]) -> float:
        """
        Get the score for a net zero target.

        This is the score for the class "NZT".
        """

        return [label["score"] for label in prediction if label["label"] in {"NZT"}][0]
