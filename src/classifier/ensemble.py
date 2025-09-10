from typing import Sequence

from src.classifier.classifier import Classifier
from src.concept import Concept
from src.identifiers import ClassifierID
from src.span import Span, group_overlapping_spans


class EnsembleClassifier(Classifier):
    """
    Classifier which combines the predictions of several classifiers.

    :param Concept concept: the concept the classifier is designed to find
    :param Sequence[Classifier] classifiers: an ordered list of classifiers. NOTE:
        assume that the behaviour of the ensemble classifier is sensitive to classifier
        order
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
            raise ValueError(
                f"All classifiers used in the ensemble must share the concept {concept}. Other concepts found: {invalid_concepts}"
            )

        unique_classifier_ids = {str(clf) for clf in classifiers}
        if len(unique_classifier_ids) < len(classifiers):
            raise ValueError("All classifiers in the ensemble must be unique")

    def _predict(self, text: str) -> list[list[Span]]:
        """
        Run prediction for each classifier in the ensemble on the input text.

        :param str text: the text to predict on
        :return list[list[Span]]: a list of spans per classifier
        """

        return [clf.predict(text) for clf in self.classifiers]

    @property
    def id(self) -> ClassifierID:
        """Return a deterministic, human-readable identifier for the classifier"""
        ensembled_classifier_ids = [clf.id for clf in self.classifiers]

        return ClassifierID.generate(
            self.name, self.concept.id, "|".join(ensembled_classifier_ids)
        )


class VotingClassifier(EnsembleClassifier):
    """
    Uses voting strategies to combine predictions from several different classifiers.

    This can be useful to estimate probabilities for predictions for classifier types
    which can't inherently output probabilities.
    """

    def predict(
        self,
        text: str,
    ) -> list[Span]:
        """
        Predict whether the text contains an instance of a concept, with probability.

        Probabilities are calculated by the proportion of models that have predicted
        each span. Prediction probabilities output by individual classifiers are ignored.

        TODO: do we want to handle passage-level probabilities?

        :param str text: The text to predict on
        :returns list[Span]: List of predictions
        """

        predictions_per_classifier = self._predict(text)
        flattened_predictions = [
            prediction
            for predictions in predictions_per_classifier
            for prediction in predictions
        ]

        span_groups = group_overlapping_spans(flattened_predictions)

        predicted_spans = []

        for span_group in span_groups:
            combined_span = Span.union(span_group)
            n_classifiers_predicted = len({span.labellers[0] for span in span_group})

            combined_span.prediction_probability = n_classifiers_predicted / len(
                self.classifiers
            )
            combined_span.labellers = [str(self)]

            predicted_spans.append(combined_span)

        return predicted_spans
