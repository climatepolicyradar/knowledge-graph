from datetime import datetime
from typing import Optional, Sequence

from src.classifier.classifier import Classifier, ProbabilityCapableClassifier
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


class VotingClassifier(EnsembleClassifier, ProbabilityCapableClassifier):
    """
    Uses voting strategies to combine predictions from several different classifiers.

    This can be useful to estimate probabilities for predictions for classifier types
    which can't inherently output probabilities.
    """

    def _combine_predictions_span_level(
        self,
        predictions_per_classifier: list[list[Span]],
    ) -> list[Span]:
        """Combine predictions from multiple classifiers, outputting span labels."""

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
            combined_span.timestamps = [datetime.now()]

            predicted_spans.append(combined_span)

        return predicted_spans

    def _combine_predictions_passage_level(
        self,
        text: str,
        predictions_per_classifier: list[list[Span]],
    ) -> Optional[Span]:
        """
        Combine predictions from multiple classifiers at passage-level.

        Outputs a single span for the entire text, or None if none of the classifiers
        predicted any spans in the text.
        """

        if all(not predictions for predictions in predictions_per_classifier):
            return None

        binary_predictions = [
            1 if predictions else 0 for predictions in predictions_per_classifier
        ]
        positive_ratio = sum(binary_predictions) / len(binary_predictions)

        return Span(
            text=text,
            start_index=0,
            end_index=len(text),
            prediction_probability=positive_ratio,
            concept_id=self.concept.wikibase_id,
            labellers=[str(self)],
            timestamps=[datetime.now()],
        )

    def predict(
        self,
        text: str,
        passage_level: bool = False,
    ) -> list[Span]:
        """
        Predict whether the text contains an instance of a concept, with probability.

        Probabilities are calculated by the proportion of models that have predicted
        each result. Prediction probabilities output by individual classifiers are ignored.

        :param str text: The text to predict on
        :param bool passage_level: Whether to combine predictions into passage level.
            Otherwise will combine predictions at the span level.
        :returns list[Span]: List of predictions
        """

        predictions_per_classifier = self._predict(text)

        if passage_level:
            prediction = self._combine_predictions_passage_level(
                text, predictions_per_classifier
            )
            predictions = [prediction] if prediction is not None else []
        else:
            predictions = self._combine_predictions_span_level(
                predictions_per_classifier
            )

        return predictions
