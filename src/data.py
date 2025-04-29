import time

from abc import abstractmethod
from typing import Iterable, Optional, TypeVar
import pandas as pd
from pydantic_ai import Agent
from pydantic import BaseModel, Field
from rich.console import Console

from src.classifier.bert_based import BertBasedClassifier
from src.classifier.classifier import Classifier
from src.classifier.targets import TargetClassifier
from src.concept import Concept
from src.labelled_passage import LabelledPassage


console = Console(highlight=False)


class SyntheticPassageWithConfidence(BaseModel):
    """Response from the LLM"""

    text: str = Field(
        description="The text of the response",
    )
    expected_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="The expected confidence score attached to the response",
    )


class SyntheticPassageWithClassifierConfidence(BaseModel):
    """Response from the LLM"""

    text: str = Field(
        description="The text of the response",
    )
    expected_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="The expected confidence score attached to the response",
    )
    actual_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="The actual confidence score attached to the response",
    )


class SyntheticData:
    """A class for generating and handling of synthetic data for training"""

    def __init__(
        self, concept: Concept, human_labelled_passages: list[LabelledPassage]
    ):
        self.concept = concept
        self.human_labelled_passages = human_labelled_passages

    @abstractmethod
    def generate(
        self, num_samples: int, **kwargs
    ) -> list[SyntheticPassageWithClassifierConfidence]:
        pass


SYSTEM_PROMPT = """
You are a specialist climate policy analyst, tasked with writing sentences similar
to the examples you are provided. We have trained a classifier to identify sentences
matching the concept desctiption (in <concept_description> tags), and the examples are 
annotated with a confidence score of how probable the classifier thought they contain the 
concept (1.0 = certain to contain the conept, 0.0 = certain to not contain it).

We need examples that are similar to the ones you see, and are on the decision boundary
(around the 0.5 mark) in confidence. These are likely to be ambiguous, in similar ways 
to the ones that are provided in the examples below.

First, carefully review the following description of the concept:

<concept_description>
{concept_description}
</concept_description>

Examples:
{examples}

YOU CANNOT REPEAT THESE EXAMPLES! Yet, you should aim to generate similar examples.

All examples you are provided come from NATIONAL CLIMATE LAWS or POLICIES.
Your examples should also plausibly come from these sources, with similar language and context. 
"""

ITERATION_PROMPT = """
You have predicted the following examples. I attach the confidence you predicted, and the classifier's predicted confidences.
Your aim is to generate examples where the actual confidence is as close to 0.5 as possible.
You'll get a number of chances after each failure. You'll have 10 rounds to try in total to get it right. After each round,
I'll add the new examples + predicted confidences + actual confidences to the list below.

Your previous examples:
{examples}

If any of the  actual confidence scores are above 0.8 or below 0.2, you're doing it wrong. If there's a large difference (> 0.2)
between your predicted confidence and the actual confidence, you're doing it wrong.
Your 3 examples which the classifier will predict ~0.5 confidence for, that are better than your previous attempts:
"""


NNClassifier = TypeVar("NNClassifier", BertBasedClassifier, TargetClassifier)


class ActiveLearningData:
    """Class for identifying data on the decision boundary of a classifier"""

    def __init__(
        self,
        classifier: NNClassifier,
        upper_bound: float = 0.7,
        lower_bound: float = 0.3,
    ):
        self.classifier = classifier
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

    def filter_text(self, passages: Iterable[str]) -> list[str]:
        """Filters the text to only include those with confidence scores between the upper and lower bounds."""
        filtered_passages = []
        predictions = self.classifier.predict_batch(list(passages), threshold=0.0)
        for passage, spans in zip(passages, predictions):
            if spans and spans[0].confidence is not None:
                c = spans[0].confidence
                if self.lower_bound <= c <= self.upper_bound:
                    filtered_passages.append(passage)

        console.log(
            f"Identified {len(filtered_passages)} passages near the decision boundary (in [{self.lower_bound}, {self.upper_bound}]) "
            f"out of {len(list(passages))} total passages."
        )
        return filtered_passages

    def filter_labelled_passages(
        self, labelled_passages: list[LabelledPassage]
    ) -> list[LabelledPassage]:
        """Filters the labelled passages to only include those with confidence scores between the upper and lower bounds."""
        text_to_passage = {passage.text: passage for passage in labelled_passages}
        filtered_text = self.filter_text(text_to_passage.keys())
        return [text_to_passage[text] for text in filtered_text]


class ActiveLearningSyntheticData(SyntheticData):
    """A class for generating and handling of synthetic data on the decision boundary"""

    UPPER_BOUND = 0.7
    LOWER_BOUND = 0.3

    def __init__(
        self,
        concept: Concept,
        human_labelled_passages: list[LabelledPassage],
        classifier: NNClassifier,
        model_name: str,
    ):
        super().__init__(concept, human_labelled_passages)
        self.classifier = classifier
        self.human_labelled_passages = human_labelled_passages
        self.agent = Agent(
            model_name,
            system_prompt=SYSTEM_PROMPT.format(
                concept_description=concept.description,
                examples=self._sample_ambiguous_passages(),
            ),
            result_type=list[SyntheticPassageWithConfidence],
        )

    def _sample_ambiguous_passages(self) -> list[dict]:
        """Samples passages, that are near the decision boundary of the classifier."""
        predictions = self._get_all_predictions()
        sampled_passages_with_confidence = self._filter_ambiguous_passages(
            predictions,
        )
        return sampled_passages_with_confidence

    def _get_all_predictions(self) -> list[LabelledPassage]:
        """Generates classifier predictions for all the human labelled passages."""
        texts = [
            labelled_passage.text for labelled_passage in self.human_labelled_passages
        ]

        predictions = [
            labelled_passage.model_copy(update={"spans": spans}, deep=True)
            for labelled_passage, spans in zip(
                self.human_labelled_passages,
                self.classifier.predict_batch(texts, threshold=0.0),
                # generating with threshold=0.0, to get all the confidence scores in [0.0, 1.0]
            )
        ]

        return predictions

    def _filter_ambiguous_passages(
        self, predictions: list[LabelledPassage]
    ) -> list[dict]:
        """Filters the passages to only include those with confidence scores between 0.3 and 0.7."""
        filtered_passages = []
        for passage in predictions:
            if passage.spans and passage.spans[0].confidence is not None:
                if (
                    passage.spans[0].confidence < self.UPPER_BOUND
                    and passage.spans[0].confidence > self.LOWER_BOUND
                ):
                    filtered_passages.append(
                        {
                            "passage": str(passage.text),
                            "confidence": f"{passage.spans[0].confidence:.2f}",
                        }
                    )

        console.log(
            f"Filtered {len(filtered_passages)} passages near the decision boundary"
        )
        return filtered_passages

    def generate(  # type: ignore
        self, num_samples: int, max_iterations: int = 20
    ) -> list[SyntheticPassageWithClassifierConfidence]:
        with console.status(f"Making predictions with {self.agent}"):
            output = self.agent.run_sync(
                "Your examples with text and expected confidence:"
            )

        synthetic_batch = []

        previous_examples = []
        for i in range(max_iterations):
            passage_spans = self.classifier.predict_batch(
                [r.text for r in output.data], threshold=0.0
            )

            if passage_spans:
                for span, example in zip(passage_spans, output.data):
                    actual_confidence = span[0].confidence
                    if actual_confidence is None:
                        continue
                    previous_examples.append(
                        {
                            "text": example.text,
                            "predicted_confidence": example.expected_confidence,
                            "actual_confidence": actual_confidence,
                        }
                    )
                    if self.LOWER_BOUND <= actual_confidence <= self.UPPER_BOUND:
                        synthetic_batch.append(
                            SyntheticPassageWithClassifierConfidence(
                                text=example.text,
                                expected_confidence=example.expected_confidence,
                                actual_confidence=actual_confidence,
                            )
                        )

                    if len(synthetic_batch) >= num_samples:
                        return synthetic_batch

                console.log(
                    f"Iteration {i + 1}, confidence scores: {[span[0].confidence for span in passage_spans]}"
                )
            output = self.agent.run_sync(
                ITERATION_PROMPT.format(
                    examples=previous_examples,
                )
            )

        console.log(
            f"Haven't converged in {max_iterations} iterations, "
            f"returning {len(synthetic_batch)} examples",
        )

        return synthetic_batch
