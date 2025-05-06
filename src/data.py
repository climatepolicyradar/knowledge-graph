from abc import abstractmethod
from typing import Iterable, TypeVar

from pydantic_ai import Agent
from rich.console import Console

from src._prompts import ITERATION_PROMPT, SYSTEM_PROMPT
from src.classifier.bert_based import BertBasedClassifier
from src.classifier.targets import TargetClassifier
from src.concept import Concept
from src.labelled_passage import LabelledPassage
from src.passage import (
    SyntheticPassageWithClassifierConfidence,
    SyntheticPassageWithConfidence,
)

console = Console(highlight=False)


NNClassifier = TypeVar("NNClassifier", BertBasedClassifier, TargetClassifier)


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
        """Generates synthetic data for training"""
        pass


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


class ActiveLearningSyntheticData(SyntheticData, ActiveLearningData):
    """
    A class for generating and handling of synthetic data on the decision boundary

    Implements both the SyntheticData and ActiveLearningData classes. It uses the Agent of
    the SyntheticData class to generate synthetic data, and the ActiveLearningData class to
    filter the generated data to only include those with confidence scores between the upper
    and lower bounds.
    """

    def __init__(
        self,
        concept: Concept,
        human_labelled_passages: list[LabelledPassage],
        classifier: NNClassifier,
        model_name: str,
    ):
        SyntheticData.__init__(self, concept, human_labelled_passages)
        ActiveLearningData.__init__(self, classifier)

        self.agent = Agent(
            model_name,
            system_prompt=SYSTEM_PROMPT.format(
                concept_description=concept.description,
                examples=self.filter_labelled_passages(human_labelled_passages),
            ),
            result_type=list[SyntheticPassageWithConfidence],
        )

    def generate(  # type: ignore
        self, num_samples: int, max_iterations: int = 20
    ) -> list[SyntheticPassageWithClassifierConfidence]:
        """Generates synthetic data for training near the decision boundary"""
        with console.status(f"Making predictions with {self.agent}"):
            output = self.agent.run_sync(
                "Your examples with text and expected confidence:"
            )

        correct_generated_passages: list[SyntheticPassageWithClassifierConfidence] = []
        generated_passages: list[SyntheticPassageWithClassifierConfidence] = []
        for i in range(max_iterations):
            passage_spans = self.classifier.predict_batch(
                [r.text for r in output.data], threshold=0.0
            )

            for span, example in zip(passage_spans, output.data):
                actual_confidence = span[0].confidence
                if actual_confidence is None:
                    continue

                synth_passage = SyntheticPassageWithClassifierConfidence(
                    **example.model_dump(),
                    actual_confidence=actual_confidence,
                )

                generated_passages.append(synth_passage)
                if self.lower_bound <= actual_confidence <= self.upper_bound:
                    correct_generated_passages.append(synth_passage)

                if len(correct_generated_passages) >= num_samples:
                    return correct_generated_passages

            console.log(
                f"Iteration {i + 1}, confidence scores: {[span[0].confidence for span in passage_spans]}"
            )
            output = self.agent.run_sync(
                ITERATION_PROMPT.format(
                    examples=[sp.model_dump() for sp in generated_passages],
                )
            )

        console.log(
            f"Haven't converged in {max_iterations} iterations, "
            f"returning {len(correct_generated_passages)} examples",
        )

        return correct_generated_passages
