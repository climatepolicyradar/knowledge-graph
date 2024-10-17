import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from src.concept import Concept
from src.identifiers import generate_identifier
from src.span import Span


class Classifier(ABC):
    """Abstract class for all classifier types."""

    def __init__(self, concept: Concept):
        self.concept = concept

    def fit(self) -> "Classifier":
        """
        Train the classifier on the data in the concept.

        :return Classifier: The trained classifier
        """
        return self

    @abstractmethod
    def predict(self, text: str) -> list[Span]:
        """
        Predict whether the supplied text contains an instance of the concept.

        :param str text: The text to predict on
        :return list[Span]: A list of spans in the text
        """
        raise NotImplementedError

    def __repr__(self):
        """Return a string representation of the classifier."""
        return f'{self.name}("{self.concept.preferred_label}")'

    @property
    def name(self):
        """Return the name of the classifier."""
        return self.__class__.__name__

    def __hash__(self):
        """
        Return a hash of the classifier.

        NB. This should be re-implemented for classifiers whose behaviour could be
        affected by different training runs.
        """
        return hash(str(self) + self.concept.model_dump_json())

    def __eq__(self, other):
        """Return whether two classifiers are equal."""
        return hash(self) == hash(other)

    @property
    def id(self):
        """Return a neat human-readable identifier for the classifier."""
        return generate_identifier(hash(self))

    def save(self, path: Union[str, Path]):
        """
        Save the classifier to a file.

        :param Union[str, Path] path: The path to save the classifier to
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Classifier":
        """
        Load a classifier from a file.

        :param Union[str, Path] path: The path to load the classifier from
        :return Classifier: The loaded classifier
        """
        with open(path, "rb") as f:
            classifier = pickle.load(f)
        assert isinstance(classifier, Classifier)
        return classifier
