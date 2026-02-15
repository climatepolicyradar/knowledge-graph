import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Protocol, Sequence, Union, overload, runtime_checkable

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)
from typing_extensions import Self

from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import ClassifierID, WikibaseID
from knowledge_graph.span import Span
from knowledge_graph.utils import iterate_batch
from knowledge_graph.version import Version


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
        self.is_fitted = False
        self.prediction_threshold: Optional[float] = None

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

    def set_prediction_threshold(self, threshold: float) -> Self:
        """
        Set a prediction threshold for the classifier.

        :param float threshold: The prediction threshold to use
        :return Self: The classifier instance.
        """

        self.prediction_threshold = threshold

        return self

    def fit(self, *args, **kwargs) -> "Classifier":
        """
        Train the classifier on the data in the concept.

        This is a template method, which subclasses can override to implement their
        specific training logic. The base class is a no-op, allowing subclasses to
        optionally implement it, only when training is needed. Keeping the template
        method here provides a consistent interface across all classifiers while
        allowing flexibility in implementation.


        :param *args: Training data or other required parameters.
        :param **kwargs: Training configuration parameters. The specific parameters
                required will depend on the classifier implementation.

        :returns Classifier: The trained classifier
        """
        self.is_fitted = True
        return self

    @overload
    def predict(
        self,
        text: str,
        batch_size: int | None = None,
        show_progress: bool = False,
        console: Console | None = None,
        threshold: float | None = None,
        **kwargs,
    ) -> list[Span]: ...

    @overload
    def predict(
        self,
        text: list[str],
        batch_size: int | None = None,
        show_progress: bool = False,
        console: Console | None = None,
        threshold: float | None = None,
        **kwargs,
    ) -> list[list[Span]]: ...

    def predict(
        self,
        text: str | list[str],
        batch_size: int | None = None,
        show_progress: bool = False,
        console: Console | None = None,
        threshold: float | None = None,
        **kwargs,
    ) -> list[Span] | list[list[Span]]:
        """
        Predict whether the supplied text contains an instance of the concept.

        :param str | list[str] text: The text to predict on
        :param int | None batch_size: Batch size to use if predicting on multiple texts.
            If not passed, defaults to predicting all texts in one batch.
        :param bool show_progress: Whether to show progress in predicting. Defaults to
            False.
        :param Console console: Optional rich console used to render the progress bar.
            This is to avoid flickering when multiple consoles are created.
        :param float | None threshold: Optional prediction threshold that overrides the
            classifier's default threshold. Predictions with confidence scores below this
            threshold will be filtered out.
        :return list[Span] | list[list[Span]]: A list of spans in the text or texts
        """

        if isinstance(text, str):
            return self._predict(text, threshold=threshold, **kwargs)

        batch_size = batch_size or len(text)
        text_batches = list(iterate_batch(text, batch_size))
        preds = []

        if show_progress:
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
                console=console,
            ) as progress:
                task = progress.add_task(
                    "Processing batches...", total=len(text_batches)
                )
                for batch in text_batches:
                    batch_preds = self._predict_batch(
                        batch, threshold=threshold, **kwargs
                    )
                    preds.extend(batch_preds)
                    progress.advance(task)
        else:
            for batch in text_batches:
                batch_preds = self._predict_batch(batch, threshold=threshold, **kwargs)
                preds.extend(batch_preds)

        return preds

    @abstractmethod
    def _predict(self, text: str, threshold: float | None = None) -> list[Span]:
        """
        Predict whether the supplied text contains an instance of the concept.

        :param str text: The text to predict on
        :param float | None threshold: Optional prediction threshold that overrides the
            classifier's default threshold
        :return list[Span]: A list of spans in the text
        """
        raise NotImplementedError

    def _predict_batch(
        self, texts: Sequence[str], threshold: float | None = None
    ) -> list[list[Span]]:
        """
        Predict whether the supplied texts contain instances of the concept.

        :param list[str] texts: The texts to predict on
        :param float | None threshold: Optional prediction threshold that overrides the
            classifier's default threshold
        :return list[list[Span]]: A list of spans in the texts for each text
        """
        return [self._predict(text, threshold=threshold) for text in texts]

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
    def id(self) -> ClassifierID:
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

        :return ClassifierID: A deterministic 8-character identifier
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


class ProbabilityCapableClassifier(ABC):
    """
    A mixin which identifies classifiers that output probabilities of predictions.

    This is useful to know when we need probabilities, or want to use a process that
    overwrites probabilities – like the VotingClassifier.
    """


@runtime_checkable
class VariantEnabledClassifier(Protocol):
    """
    Protocol for classifiers that can generate variants of themselves.

    Classes implementing this protocol can:

    1. Create variant instances of themselves (with stochastic variation)
    2. Make predictions on text

    This protocol ensures type safety when performing uncertainty estimation.
    """

    def get_variant(self) -> "VariantEnabledClassifier": ...  # noqa: D102
    def _predict(self, text: str, threshold: float | None = None) -> list[Span]: ...  # noqa: D102
