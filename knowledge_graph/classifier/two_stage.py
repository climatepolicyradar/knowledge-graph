from datetime import datetime
from typing import Sequence, overload

from rich.console import Console

from knowledge_graph.classifier.classifier import (
    Classifier,
    GPUBoundClassifier,
    ZeroShotClassifier,
)
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import ClassifierID
from knowledge_graph.span import Span
from knowledge_graph.utils import get_logger


class TwoStageClassifier(Classifier):
    """
    A classifier which only labels a passage when two classifiers agree.

    The two classifiers run in a fixed order:

    1. The candidate classifier runs first, on every passage. It should be the
       cheap one (i.e. a KeywordClassifier).
    2. The filter classifier runs second, only on passages where the candidate
       classifier positively labelled spans. It can be the expensive one (eg. a BERT model).

    Spans are published under the candidate classifier's concept. Where the filter
    classifier gives a probability, it is used as the prediction_probability of the
    kept spans.
    """

    def __init__(
        self,
        concept: Concept,
        filter_classifier: Classifier | str,
        candidate_classifier: Classifier | str | None = None,
        filter_threshold: float | str | None = None,
    ):
        """
        Create a two-stage classifier.

        Every argument can also be given as a string, so the classifier can be built
        with `scripts/train.py --classifier-type TwoStageClassifier
        --classifier-override filter_classifier=<wandb path>`.

        :param Concept concept: The concept to publish spans under
        :param Classifier | str filter_classifier: The filter classifier, or the W&B
            path to load it from
        :param Classifier | str | None candidate_classifier: The candidate
            classifier, or the W&B path to load it from. Defaults to a
            KeywordClassifier for the concept.
        :param float | str | None filter_threshold: Optional probability threshold
            for the filter classifier
        """
        super().__init__(concept)

        # Imported here to avoid a circular import
        from knowledge_graph.classifier.keyword import KeywordClassifier
        from knowledge_graph.wandb_helpers import load_classifier_from_wandb

        if isinstance(filter_classifier, str):
            filter_classifier = load_classifier_from_wandb(filter_classifier)
        if isinstance(candidate_classifier, str):
            candidate_classifier = load_classifier_from_wandb(candidate_classifier)
        elif candidate_classifier is None:
            candidate_classifier = KeywordClassifier(concept)

        if candidate_classifier.concept.wikibase_id != concept.wikibase_id:
            raise ValueError(
                f"The candidate classifier's concept ({candidate_classifier.concept}) "
                f"must match the concept of the {self.name} ({concept})"
            )
        for sub_classifier in (candidate_classifier, filter_classifier):
            if not sub_classifier.is_fitted and not isinstance(
                sub_classifier, ZeroShotClassifier
            ):
                raise ValueError(
                    f"{sub_classifier} must be fitted before it is used in a "
                    f"{self.name}"
                )
        if isinstance(candidate_classifier, GPUBoundClassifier) and not isinstance(
            filter_classifier, GPUBoundClassifier
        ):
            get_logger().warning(
                f"The candidate classifier {candidate_classifier} is GPU-bound but the "
                f"filter classifier {filter_classifier} is not. The candidate "
                f"classifier runs on every passage, so it should be the cheaper of the "
                f"two."
            )

        self.candidate_classifier = candidate_classifier
        self.filter_classifier = filter_classifier
        # CLI kwargs only parse ints and bools, so floats arrive as strings
        self.filter_threshold = (
            float(filter_threshold) if filter_threshold is not None else None
        )
        # Both sub-classifiers are already zero-shot or fitted
        self.is_fitted = True

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

        Unlike the base class, which runs both stages on one batch at a time, this
        runs the candidate classifier over every text first, then sends all of the
        candidate passages to the filter classifier in full batches of batch_size.
        This stops an expensive filter classifier (eg. BERT on a GPU) being given
        lots of small, uneven batches.

        :param float | None threshold: Optional threshold which is passed to the
            filter classifier, overriding filter_threshold. It is never passed to the
            candidate classifier.
        """
        if isinstance(text, str):
            return self._predict(text, threshold=threshold)

        candidate_spans = self.candidate_classifier.predict(
            text, batch_size=batch_size, show_progress=show_progress, console=console
        )
        return self._filter_candidate_spans(
            text,
            candidate_spans,
            threshold=threshold,
            batch_size=batch_size,
            show_progress=show_progress,
            console=console,
        )

    def _predict(self, text: str, threshold: float | None = None) -> list[Span]:
        """Predict whether the supplied text contains an instance of the concept."""
        return self._predict_batch([text], threshold=threshold)[0]

    def _predict_batch(
        self, texts: Sequence[str], threshold: float | None = None
    ) -> list[list[Span]]:
        """
        Predict whether the supplied texts contain instances of the concept.

        :param float | None threshold: Optional threshold which is passed to the
            filter classifier, overriding filter_threshold. It is never passed to the
            candidate classifier.
        """
        candidate_spans = self.candidate_classifier._predict_batch(texts)
        return self._filter_candidate_spans(texts, candidate_spans, threshold=threshold)

    def _filter_candidate_spans(
        self,
        texts: Sequence[str],
        candidate_spans: list[list[Span]],
        threshold: float | None = None,
        batch_size: int | None = None,
        show_progress: bool = False,
        console: Console | None = None,
    ) -> list[list[Span]]:
        """
        Keep candidate spans only on passages where the filter classifier fires.

        The filter classifier is only run on passages which have candidate spans.
        """
        candidate_indices = [i for i, spans in enumerate(candidate_spans) if spans]
        if not candidate_indices:
            return [[] for _ in texts]

        filter_threshold = threshold if threshold is not None else self.filter_threshold
        filter_spans = self.filter_classifier.predict(
            [texts[i] for i in candidate_indices],
            batch_size=batch_size,
            show_progress=show_progress,
            console=console,
            threshold=filter_threshold,
        )

        now = datetime.now()
        labeller = str(self)
        results: list[list[Span]] = [[] for _ in texts]
        for i, spans_from_filter in zip(candidate_indices, filter_spans):
            if not spans_from_filter:
                continue
            filter_probability = max(
                (
                    span.prediction_probability
                    for span in spans_from_filter
                    if span.prediction_probability is not None
                ),
                default=None,
            )
            results[i] = [
                span.model_copy(
                    update={
                        "prediction_probability": filter_probability,
                        "labellers": [labeller],
                        "timestamps": [now],
                    }
                )
                for span in candidate_spans[i]
            ]

        return results

    def move_model_to_device(self, device=None) -> None:
        """Move any torch-backed sub-classifiers to the given device."""
        for sub_classifier in (self.candidate_classifier, self.filter_classifier):
            if hasattr(sub_classifier, "move_model_to_device"):
                sub_classifier.move_model_to_device(device)  # type: ignore[attr-defined]

    def __setstate__(self, state: dict) -> None:
        """
        Restore the classifier after unpickling.

        Classifier.load only moves top-level BERT models onto the best available
        device, so do the same here for any torch-backed sub-classifiers.
        """
        vars(self).update(state)
        self.move_model_to_device()

    @property
    def id(self) -> ClassifierID:
        """Return a deterministic, human-readable identifier for the classifier."""
        return ClassifierID.generate(
            self.name,
            self.concept.id,
            self.candidate_classifier.id,
            self.filter_classifier.id,
            self.filter_threshold,
        )

    def __repr__(self) -> str:
        """Return a string representation, showing the order the classifiers run in."""
        return (
            f'{self.name}("{self.concept.preferred_label}", '
            f"first={self.candidate_classifier!r}, then={self.filter_classifier!r})"
        )
