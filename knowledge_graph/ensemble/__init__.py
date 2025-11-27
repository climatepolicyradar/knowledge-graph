import logging
from typing import Any, Sequence, cast, overload

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
)

from flows.utils import iterate_batch
from knowledge_graph.classifier.classifier import (
    Classifier,
    VariantEnabledClassifier,
    ZeroShotClassifier,
)
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import ClassifierID
from knowledge_graph.span import Span

logger = logging.getLogger(__name__)


class IncompatibleSubClassifiersError(Exception):
    """Exception raised when classifiers don't share the same concept."""

    def __init__(self, reason: str):
        self.message = f"Classifiers attempting to be ensembled are incompatible.\nReason: {reason}"
        super().__init__(self.message)


class Ensemble:
    """
    A collection of classifiers.

    These can be used to improve the performance of a single classifier or measure
    its stability or uncertainty, by creating an ensemble containing slight variants
    of the same classifier.

    The `predict` and `predict_batch` methods here return lists of the same types as
    their equivalents on Classifier: one for each classifier.
    """

    def __init__(
        self,
        concept: Concept,
        classifiers: Sequence[Classifier],
    ):
        self._validate_classifiers(concept, classifiers)
        self.concept = concept
        self.classifiers = classifiers

    def _validate_classifiers(
        self,
        concept: Concept,
        classifiers: Sequence[Classifier],
    ) -> None:
        """Check that classifiers are compatible to be part of the same ensemble."""

        if invalid_concepts := {
            clf.concept for clf in classifiers if clf.concept != concept
        }:
            raise IncompatibleSubClassifiersError(
                f"All classifiers used in the ensemble must share the concept {concept}. Other concepts found: {invalid_concepts}"
            )

    @overload
    def predict(
        self,
        text: str,
        batch_size: int | None = None,
        show_progress: bool = False,
        console: Console | None = None,
        **kwargs,
    ) -> list[list[Span]]: ...

    @overload
    def predict(
        self,
        text: list[str],
        batch_size: int | None = None,
        show_progress: bool = False,
        console: Console | None = None,
        **kwargs,
    ) -> list[list[list[Span]]]: ...

    def predict(
        self,
        text: str | list[str],
        batch_size: int | None = None,
        show_progress: bool = False,
        console: Console | None = None,
        **kwargs,
    ) -> list[list[Span]] | list[list[list[Span]]]:
        """
        Predict whether the supplied text contains instances of the concept using the ensemble.

        :param str | list[str] text: The text to predict on
        :param int | None batch_size: Batch size to use if predicting on multiple texts.
            If not passed, defaults to predicting all texts in one batch.
        :param bool show_progress: Whether to show progress in predicting. Defaults to
            False.
        :param Console console: Optional rich console used to render the progress bar.
            This is to avoid flickering when multiple consoles are created.
        :return list[list[Span]] | list[list[list[Span]]]: A list of spans per classifier
            for single text, or a list of spans per classifier per text for multiple texts
        """

        if isinstance(text, str):
            return self._predict(text, **kwargs)

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
                    batch_preds = self._predict_batch(batch, **kwargs)  # type: ignore
                    preds.extend(batch_preds)
                    progress.advance(task)
        else:
            for batch in text_batches:
                batch_preds = self._predict_batch(batch, **kwargs)  # type: ignore
                preds.extend(batch_preds)

        return preds

    def _predict(self, text: str) -> list[list[Span]]:
        """
        Run prediction for each classifier in the ensemble on the input text.

        :param str text: the text to predict on
        :return list[list[Span]]: a list of spans per classifier
        """

        return [clf.predict(text) for clf in self.classifiers]

    def _predict_batch(self, texts: list[str]) -> list[list[list[Span]]]:
        """
        Run prediction for each classifier in the ensemble on the input text batch.

        Spans are returned with the outer list being batches, and the inner list being
        classifiers. This is to make the output consistent with `Ensemble().predict`.

        :param Sequence[str] texts: the text to predict on
        :return list[list[list[Span]]]: a list of spans per classifier per batch
        """

        # this is in the format classifier -> batch -> spans
        spans_per_batch_per_classifier = [
            clf.predict(texts, batch_size=len(texts)) for clf in self.classifiers
        ]

        # transpose to batch -> classifier -> spans
        return [
            [
                spans_per_batch_per_classifier[clf_idx][batch_idx]
                for clf_idx in range(len(self.classifiers))
            ]
            for batch_idx in range(len(texts))
        ]

    @property
    def name(self) -> str:
        """Return a string representation of the ensemble type, i.e. name of the class."""
        return self.__class__.__name__

    @property
    def id(self) -> ClassifierID:
        """Return a deterministic, human-readable identifier for the ensemble"""
        ensembled_classifier_ids = [clf.id for clf in self.classifiers]

        return ClassifierID.generate(
            self.name, self.concept.id, "|".join(ensembled_classifier_ids)
        )

    def __repr__(self) -> str:
        """Return a string representation of the ensemble."""
        return str(self.id)


def create_ensemble(
    classifier: Classifier,
    n_classifiers: int,
) -> Ensemble:
    """
    Create an ensemble from a classifier.

    :param classifier: A classifier. Must be a VariantEnabledClassifier, and must have
        been fit if it's fittable.
    :param n_classifiers: Total number of classifiers to include in the ensemble
    :return Ensemble: An ensemble containing the fitted classifier and its variants
    :raises ValueError: if n_classifiers < 1 or if classifier is not variant-enabled
    """
    if not isinstance(classifier, VariantEnabledClassifier):
        raise ValueError(
            f"Classifier must be variant-enabled to be part of an ensemble.\nClassifier type {classifier.name} is not."
        )

    if not classifier.is_fitted and not isinstance(classifier, ZeroShotClassifier):
        raise ValueError(
            f"Classifier must be fitted before creating an ensemble.\n"
            f"Call {classifier.name}.fit() before creating the ensemble."
        )

    if n_classifiers < 1:
        raise ValueError(f"n_classifiers must be at least 1, got {n_classifiers}")

    # cast is needed here as list is invariant, so list[Classifier] is incompatible
    # with list[VariantEnabledClassifier]
    classifiers: list[Classifier] = [
        classifier,
        *[cast(Classifier, classifier.get_variant()) for _ in range(n_classifiers - 1)],
    ]

    ensemble = Ensemble(
        concept=classifier.concept,
        classifiers=classifiers,
    )

    return ensemble
