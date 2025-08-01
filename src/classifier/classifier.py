import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Sequence, Union

from typing_extensions import Self

from src.concept import Concept
from src.identifiers import Identifier, WikibaseID
from src.span import Span
from src.version import Version


class Classifier(ABC):
    """Abstract class for all classifier types."""

    concept: Concept
    version: Optional[Version]
    allowed_concept_ids: Optional[Sequence[WikibaseID]] = None

    def __init__(
        self,
        concept: Concept,
        version: Optional[Version] = None,
        **kwargs,
    ):
        self.concept = concept
        self.version = version

        if self.allowed_concept_ids:
            self._validate_concept_id(self.allowed_concept_ids)

    def _validate_concept_id(self, expected_ids: Sequence[WikibaseID]) -> None:
        """Check whether the supplied concept matches one of the expected IDs."""
        if self.concept.wikibase_id not in expected_ids:
            expected = (
                f"one of {','.join(str(id) for id in expected_ids)}"
                if len(expected_ids) > 1
                else str(expected_ids[0])
            )
            raise ValueError(
                f"The concept supplied to a {self.name} must be "
                f"{expected}, not {self.concept.wikibase_id}"
            )

    def fit(self, **kwargs) -> "Classifier":
        """
        Train the classifier on the data in the concept.

        This is a template method, which subclasses can override to implement their
        specific training logic. The base class is a no-op, allowing subclasses to
        optionally implement it, only when training is needed. Keeping the template
        method here provides a consistent interface across all classifiers while
        allowing flexibility in implementation.

        Args:
            **kwargs: Training configuration parameters. The specific parameters
                required will depend on the classifier implementation.

        Returns:
            Classifier: The trained classifier
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

    def predict_batch(self, texts: list[str]) -> list[list[Span]]:
        """
        Predict whether the supplied texts contain instances of the concept.

        :param list[str] texts: The texts to predict on
        :return list[list[Span]]: A list of spans in the texts for each text
        """
        return [self.predict(text) for text in texts]

    def get_variant_sub_classifier(self) -> Self:
        """
        Get a variant of the classifier, used for uncertainty estimation.

        Subclasses should override this method to return a variant of the classifier
        with some stochastic variation, eg a different random seed, or dropout enabled.
        """
        raise NotImplementedError

    def __repr__(self):
        """Return a string representation of the classifier."""
        return f'{self.name}("{self.concept.preferred_label}")'

    @property
    def name(self):
        """Return the name of the classifier."""
        return self.__class__.__name__

    def __eq__(self, other):
        """Return whether two classifiers are equal."""
        if not isinstance(other, Classifier):
            return False
        return self.id == other.id

    @property
    @abstractmethod
    def id(self) -> Identifier:
        """
        Return a deterministic, human-readable identifier for the classifier.

        Identifiers are generated from each classifier's core properties (name, concept,
        and behavior-driving parameters). They should be:

        - deterministic: the same classifier configuration will always produce
          the same id
        - cross-session consistent: the id remains identical across different
          python sessions, processes, and machines
        - unique: different classifier configurations will produce different ids
        - human-readable: an 8-character string using unambiguous characters

        This property should be used for:
        - Persistent storage and retrieval of classifiers
        - Tracing and debugging classifiers across different environments

        Classifier subclasses should override this method to return a unique identifier
        for the classifier according to its specific parameters and implementation.

        :return Identifier: A deterministic 8-character identifier
        """
        ...

    def __hash__(self) -> int:
        """
        Return a hash for the classifier for use in Python collections.

        This hash is derived from the classifier's id, but is only intended to be used
        for in-memory collection/comparison purposes (sets, dictionaries, equality
        checks, etc.). Unlike the classifier.id, hashes are session-dependent and
        should not be used for persistent, cross-session identification.

        :return int: A hash value for in-memory collection/comparison usage
        """
        return hash(self.id)

    def save(self, path: Union[str, Path]):
        """
        Save the classifier to a file.

        :param Union[str, Path] path: The path to save the classifier to
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Union[str, Path], model_to_cuda: bool = False) -> "Classifier":
        """
        Load a classifier from a file.

        :param Union[str, Path] path: The path to load the classifier from
        :return Classifier: The loaded classifier
        """
        with open(path, "rb") as f:
            classifier = pickle.load(f)
        assert isinstance(classifier, Classifier)
        if model_to_cuda and hasattr(classifier, "pipeline"):
            classifier.pipeline.model.to("cuda:0")  # type: ignore
            import torch  # type: ignore[import-untyped]

            classifier.pipeline.device = torch.device("cuda:0")  # type: ignore
        return classifier


class ZeroShotClassifier(ABC):
    """
    A mixin which identifies classifiers that can make predictions without training.

    Zero-shot classifiers can make predictions based only on a concept object, without
    seeing any examples of the concept appearing in real passages of text. Zero-shot
    classifiers do not require calling .fit() before running .predict().

    See: https://en.wikipedia.org/wiki/Zero-shot_learning
    """


class GPUBoundClassifier(ABC):
    """
    A mixin which identifies classifiers which should run on GPU hardware.

    GPU-bound classifiers should ideally run on GPU hardware during both training and
    inference.
    """
