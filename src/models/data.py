from abc import abstractmethod
from typing import TypeVar

from rich.console import Console

from src.classifier.bert_based import BertBasedClassifier
from src.classifier.targets import TargetClassifier
from src.concept import Concept
from src.labelled_passage import LabelledPassage
from src.models.passage import (
    SyntheticPassageWithClassifierConfidence,
)

console = Console(highlight=False)


NNClassifier = TypeVar("NNClassifier", BertBasedClassifier, TargetClassifier)


class SyntheticData:
    """An abstract class for generating and handling of synthetic data for training"""

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
