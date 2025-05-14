from typing import Iterable, TypeVar

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings

from src._prompts import ITERATION_PROMPT, SYSTEM_PROMPT
from src.classifier.bert_based import BertBasedClassifier
from src.classifier.targets import TargetClassifier
from src.concept import Concept
from src.models.labelled_passage import LabelledPassage
from src.models.data import SyntheticData, console
from src.models.passage import (
    Passage,
    PassageWithClassifierConfidence,
    Source,
    SyntheticPassageWithClassifierConfidence,
    SyntheticPassageWithConfidence,
)
from src.span import Span

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

    @staticmethod
    def _aggregate_spans_to_confidence(spans: list[Span]) -> float:
        """The rationale here is that we're running text classification, hence taking the max confidence"""
        confidences = [s.confidence for s in spans if s.confidence is not None]
        if not confidences:
            return 0.0
        return max(confidences)

    def predict(self, passages: Iterable[str]) -> list[PassageWithClassifierConfidence]:
        """Runs predictions to obtain the confidence for each passage"""
        predictions = self.classifier.predict_batch(list(passages), threshold=0.0)
        return [
            PassageWithClassifierConfidence(
                text=passage, confidence=self._aggregate_spans_to_confidence(spans)
            )
            for passage, spans in zip(passages, predictions)
        ]

    def filter_labelled_passages(
        self, labelled_passages: list[LabelledPassage]
    ) -> list[LabelledPassage]:
        """Filters the labelled passages to only include those with confidence scores between the upper and lower bounds."""
        predicted_passages = self.predict(
            [labelled_passage.text for labelled_passage in labelled_passages]
        )
        filtered_passages = []
        for pred, passage in zip(predicted_passages, labelled_passages):
            if self.lower_bound <= pred.confidence <= self.upper_bound:
                filtered_passages.append(passage)

        return filtered_passages

    def filter_passages(
        self, passages: list[Passage]
    ) -> list[PassageWithClassifierConfidence]:
        """Filters the raw passages to those between the upper and lower bounds"""
        predicted_passages = self.predict([p.text for p in passages])

        for pred, passage in zip(predicted_passages, passages):
            pred.source = passage.source

        return [
            p
            for p in predicted_passages
            if self.lower_bound <= p.confidence <= self.upper_bound
        ]


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
        temperature: float = 0.1,
    ):
        SyntheticData.__init__(self, concept, human_labelled_passages)
        ActiveLearningData.__init__(self, classifier)

        self.system_prompt = SYSTEM_PROMPT.format(
            concept_description=concept.description,
            examples="\n\t- ".join(
                [
                    lp.get_highlighted_text()
                    for lp in self.filter_labelled_passages(human_labelled_passages)[:5]
                ]
            ),
        )
        self.model_name = model_name
        self.agent = Agent(
            self.model_name,
            system_prompt=self.system_prompt,
            result_type=list[SyntheticPassageWithConfidence],
            model_settings=ModelSettings(temperature=temperature),
        )

    def generate(  # type: ignore
        self, num_samples: int, max_iterations: int = 20
    ) -> list[SyntheticPassageWithClassifierConfidence]:
        """Generates synthetic data for training near the decision boundary"""
        with console.status(f"Making predictions with {self.agent}"):
            output = self.agent.run_sync(
                "Your examples with text and expected confidence:"
            )

        task_prompt = ""
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
                    text=example.text,
                    expected_confidence=example.expected_confidence,
                    actual_confidence=actual_confidence,
                    source=Source(
                        type="Synthetic",
                        model=self.model_name,
                        prompt="system_prompt_v1",  # TODO: need to decide whether we want to retain the whole prompt
                    ),
                )

                generated_passages.append(synth_passage)
                if self.lower_bound <= actual_confidence <= self.upper_bound:
                    correct_generated_passages.append(synth_passage)

                if len(correct_generated_passages) >= num_samples:
                    return correct_generated_passages

            console.log(
                f"Iteration {i + 1}, confidence scores: {[span[0].confidence for span in passage_spans]}"
            )
            task_prompt = ITERATION_PROMPT.format(
                examples=[
                    sp.model_dump(exclude={"source"}) for sp in generated_passages
                ],
            )
            output = self.agent.run_sync(task_prompt)

        console.log(
            f"Haven't converged in {max_iterations} iterations, "
            f"returning {len(correct_generated_passages)} examples",
        )

        return correct_generated_passages


class ActiveLearningCorpusData(ActiveLearningData):
    """Active Learning filtering for the existing CPR corpus"""

    def __init__(
        self,
        classifier: NNClassifier,
        upper_bound: float = 0.7,
        lower_bound: float = 0.3,
    ):
        ActiveLearningData.__init__(self, classifier, upper_bound, lower_bound)

    def find_examples(
        self, num_samples: int, dataset: Iterable, skip: int = 100
    ) -> list[PassageWithClassifierConfidence]:
        """
        Finds examples on the decision boundary of the classifier

        Takes a dataset (e.g. a stream from `HuggingfaceSession`) and finds example in it, using the
        `filter_text` method. The `skip` argument is used to skip every nth passage in the dataset for
        speed and variety in sampling. If you want to scan all passages, set it to 1.

        NOTE: currently only supporting the CPR HF dataset format, with the `text_block.text` and
        `document_id` fields.
        """

        passages = []

        counter = 0
        for i in dataset:
            counter += 1
            if counter % skip == 0:
                text = i["text_block.text"]
                if text is not None:
                    passages += self.filter_passages(
                        [
                            Passage(
                                text=text,
                                source=Source(
                                    type="Natural", document_id=i["document_id"]
                                ),
                            )
                        ]
                    )

                    if len(passages) >= num_samples:
                        console.print(
                            f"Scanned a total of {counter / skip} passages and found {len(passages)}"
                        )
                        return passages

        console.print(
            f"Scanned a total of {counter / skip} passages and found {len(passages)}"
        )
        return passages
