from abc import abstractmethod
from typing import TYPE_CHECKING

from more_itertools import pairwise
from typing_extensions import Self

from src.span import Span, jaccard_similarity_for_span_lists

if TYPE_CHECKING:
    pass


class Uncertainty(float):
    """
    A validated float representing uncertainty as a number between 0 and 1.

    Lower values represent high confidence, with 0 representing total certainty.
    Higher values represent low confidence, with 1 representing total uncertainty.
    """

    def __new__(cls, value: int | float) -> "Uncertainty":
        """Create a new uncertainty score"""
        if not 0 <= value <= 1:
            raise ValueError(f"Uncertainty values must be between 0 and 1. Got {value}")
        return super().__new__(cls, value)


class UncertaintyMixin:
    """
    Mixin class providing shared uncertainty calculation logic.

    This mixin can be used by any classifier that implements get_variant_sub_classifier()
    to provide consistent uncertainty calculation across different classifier types.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, "get_variant_sub_classifier"):
            raise NotImplementedError(
                f"{self.__class__.__name__} must have implemented the "
                "get_variant_sub_classifier() method in order to use the uncertainty "
                "estimation mixin."
            )

    @abstractmethod
    def get_variant_sub_classifier(self) -> "Self":
        """Get a variant of the classifier. Must be implemented by subclasses."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, text: str) -> list[Span]:
        """Predict spans in the given text. Must be implemented by subclasses."""
        raise NotImplementedError

    def predict_with_uncertainty(
        self, text: str, num_samples: int = 10
    ) -> tuple[list[list[Span]], Uncertainty]:
        """Predict with uncertainty estimation using multiple sampling runs."""
        predictions = []
        for _ in range(num_samples):
            sub_classifier = self.get_variant_sub_classifier()
            prediction = sub_classifier.predict(text)
            predictions.append(prediction)
        return predictions, self.calculate_prediction_uncertainty(predictions)

    def calculate_prediction_uncertainty(
        self, predictions: list[list[Span]]
    ) -> Uncertainty:
        """
        Calculate uncertainty score from multiple predictions using a comprehensive approach.

        This method combines binary prediction consistency (concept present/absent) with
        span overlap consistency for positive predictions to provide a robust uncertainty
        measure that works well across different types of concepts and text lengths.

        Args:
            predictions: List of span predictions from multiple sampling runs

        Returns:
            Uncertainty score between 0 (certain) and 1 (uncertain)
        """
        if not predictions:
            # If we have no predictions, we are certain that the concept is not present
            return Uncertainty(1.0)

        # Calculate binary prediction consistency (concept present/absent)
        binary_uncertainty = self._calculate_binary_uncertainty(predictions)

        # If all predictions are negative, uncertainty is based only on binary consistency
        positive_predictions = [spans for spans in predictions if spans]
        if len(positive_predictions) < 2:
            return binary_uncertainty

        # For positive predictions, calculate span overlap consistency
        overlap_uncertainty = self._calculate_span_overlap_uncertainty(
            positive_predictions
        )

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

    def _calculate_binary_uncertainty(
        self, predictions: list[list[Span]]
    ) -> Uncertainty:
        """
        Calculate uncertainty from binary predictions (concept present/absent).

        Args:
            predictions: List of span predictions from multiple sampling runs

        Returns:
            Uncertainty score between 0 and 1
        """

        binary_predictions = [1 if spans else 0 for spans in predictions]

        if len(set(binary_predictions)) <= 1:
            return Uncertainty(0.0)

        # Calculate proportion of positive predictions
        positive_ratio = sum(binary_predictions) / len(binary_predictions)

        # Uncertainty is highest when predictions are 50/50 split
        # Use 4 * p * (1-p), which is maximized at p=0.5 and equals 1.0.
        uncertainty = 4 * positive_ratio * (1 - positive_ratio)

        return Uncertainty(uncertainty)

    def _calculate_span_overlap_uncertainty(
        self, positive_predictions: list[list[Span]]
    ) -> Uncertainty:
        """
        Calculate uncertainty from span overlap consistency in positive predictions.

        This method calculates the Jaccard similarity between the sets of
        character indices covered by the spans in each pair of prediction runs.
        This provides a holistic measure of overlap that naturally handles
        multiple disjoint spans.

        Args:
            positive_predictions: List of non-empty span predictions. Each element
                                  should be a list of spans from one prediction run.

        Returns:
            Uncertainty score between 0 (low uncertainty/high agreement) and 1
            (high uncertainty/low agreement).
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
