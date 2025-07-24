from typing import Literal, Protocol

from more_itertools import pairwise

from src.span import Span, UnitInterval, jaccard_similarity_for_span_lists


class UncertaintyCapableClassifier(Protocol):
    """
    Protocol for classifiers that support uncertainty estimation through sampling.

    This protocol defines the complete interface required for Monte Carlo uncertainty
    estimation. Classes implementing this protocol can:

    1. Create variant instances of themselves (with stochastic variation)
    2. Make predictions on text
    3. Calculate uncertainty from multiple prediction samples using different methods

    The uncertainty estimation process works by:
    - Creating multiple variant classifiers using get_variant_sub_classifier()
    - Running predictions with each variant on the same text
    - Analyzing the variance in predictions to estimate uncertainty using one of three methods:
      * Passage uncertainty: Based only on binary prediction consistency (concept present/absent)
      * Span uncertainty: Based only on overlap consistency between predicted spans
      * Combined uncertainty: Merges both binary (present/absent) and span overlap consistency

    This protocol ensures type safety when performing uncertainty estimation.
    """

    def get_variant_sub_classifier(self) -> "UncertaintyCapableClassifier": ...  # noqa: D102
    def predict(self, text: str) -> list[Span]: ...  # noqa: D102
    def _calculate_passage_uncertainty(
        self, predictions: list[list[Span]]
    ) -> "Uncertainty": ...  # noqa: D102
    def _calculate_span_uncertainty(
        self, positive_predictions: list[list[Span]]
    ) -> "Uncertainty": ...  # noqa: D102
    def _calculate_combined_uncertainty(
        self, predictions: list[list[Span]]
    ) -> "Uncertainty": ...  # noqa: D102


class Uncertainty(UnitInterval):
    """
    A validated float representing uncertainty between 0.0 and 1.0.

    Lower values represent high confidence, with 0 representing total certainty.
    Higher values represent low confidence, with 1 representing total uncertainty.
    """


class UncertaintyMixin:
    """
    Mixin class providing shared uncertainty calculation logic.

    This mixin should be used with classes that implement:
    - get_variant_sub_classifier() method for creating variants
    - predict() method for making predictions

    For type checking, classes using this mixin should be declared as:
        class MyClassifier(Classifier, UncertaintyMixin):
            # ... implementation

    The type system enforces that predict_with_uncertainty() can only be called
    on objects that implement both the classifier interface and the uncertainty methods.
    """

    def predict_with_uncertainty(
        self: UncertaintyCapableClassifier,
        text: str,
        num_samples: int = 10,
        method: Literal["combined", "passage", "span"] = "combined",
    ) -> tuple[list[list[Span]], Uncertainty]:
        """
        Predict with uncertainty estimation using multiple sampling runs.

        Args:
            text: The text to predict spans for
            num_samples: The number of sampling runs to perform
            method: The method to use to calculate uncertainty. Must be one of
                span: calculate uncertainty from span overlap consistency in positive predictions
                passage: calculate uncertainty from binary predictions (concept present/absent)
                combined: calculate uncertainty using contributions from both span and binary calculations

        Returns:
            A tuple (predictions, uncertainty) where:
            - predictions: List of span predictions from each sampling run
            - uncertainty: Uncertainty score between 0 (certain) and 1 (uncertain)
        """

        match method:
            case "combined":
                uncertainty_calculator = self._calculate_combined_uncertainty
            case "passage":
                uncertainty_calculator = self._calculate_passage_uncertainty
            case "span":
                uncertainty_calculator = self._calculate_span_uncertainty
            case _:
                raise ValueError(f"Invalid uncertainty method: {method}")

        predictions = []
        for _ in range(num_samples):
            sub_classifier = self.get_variant_sub_classifier()
            prediction = sub_classifier.predict(text)
            predictions.append(prediction)

        return predictions, uncertainty_calculator(predictions)

    def _calculate_combined_uncertainty(
        self, predictions: list[list[Span]]
    ) -> Uncertainty:
        """
        Calculate uncertainty score from multiple predictions.

        This method combines binary prediction consistency (concept present/absent) with
        span overlap consistency for positive predictions to provide a robust uncertainty
        measure.

        Args:
            predictions: List of span predictions from multiple sampling runs

        Returns:
            Uncertainty score between 0 (certain) and 1 (uncertain)
        """
        if not predictions:
            # If we have no predictions, we are certain that the concept is not present
            return Uncertainty(0.0)

        # Calculate binary prediction consistency (concept present/absent)
        binary_uncertainty = self._calculate_passage_uncertainty(predictions)

        # If all predictions are negative, uncertainty is based only on binary consistency
        positive_predictions = [spans for spans in predictions if spans]
        if len(positive_predictions) < 2:
            return binary_uncertainty

        # For positive predictions, calculate span overlap consistency
        overlap_uncertainty = self._calculate_span_uncertainty(positive_predictions)

        # Combine binary and overlap uncertainty
        binary_weight = 0.7  # binary uncertainty weighted more heavily
        overlap_weight = 0.3
        assert binary_weight + overlap_weight == 1.0, (
            f"Weights must sum to 1.0, got {binary_weight} + {overlap_weight} != 1.0"
        )

        combined_uncertainty = (binary_weight * binary_uncertainty) + (
            overlap_weight * overlap_uncertainty
        )

        return Uncertainty(combined_uncertainty)

    def _calculate_passage_uncertainty(
        self, predictions: list[list[Span]]
    ) -> Uncertainty:
        """
        Calculate uncertainty from binary predictions (concept present/absent).

        Args:
            predictions: List of span predictions from multiple sampling runs

        Returns:
            Uncertainty score between 0 and 1
        """

        if all(not spans for spans in predictions):
            return Uncertainty(0.0)

        binary_predictions = [1 if spans else 0 for spans in predictions]

        # Calculate proportion of positive predictions
        positive_ratio = sum(binary_predictions) / len(binary_predictions)

        # Uncertainty is highest when predictions are 50/50 split
        # Use 4 * p * (1-p), which is maximized at p=0.5 and equals 1.0.
        uncertainty = 4 * positive_ratio * (1 - positive_ratio)

        return Uncertainty(uncertainty)

    def _calculate_span_uncertainty(
        self, positive_predictions: list[list[Span]]
    ) -> Uncertainty:
        """
        Calculate uncertainty from span overlap consistency in positive predictions.

        This method calculates the Jaccard similarity between the sets of
        character indices covered by the spans in each pair of prediction runs.
        This provides a holistic measure of overlap that naturally handles
        multiple disjoint spans.

        Args:
            positive_predictions: List of non-empty span predictions from multiple sampling runs

        Returns:
            Uncertainty score between 0 (low uncertainty/high agreement) and 1 (high uncertainty/low agreement)
        """
        if len(positive_predictions) < 2:
            return Uncertainty(0.0)

        jaccard_scores = []
        for predictions_i, predictions_j in pairwise(positive_predictions):
            jaccard = jaccard_similarity_for_span_lists(predictions_i, predictions_j)
            jaccard_scores.append(jaccard)

        # Convert similarity to uncertainty (high similarity = low uncertainty)
        overlap_uncertainty = 1.0 - sum(jaccard_scores) / len(jaccard_scores)

        return Uncertainty(overlap_uncertainty)
