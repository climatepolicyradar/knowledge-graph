from typing import Literal, Protocol

from more_itertools import pairwise

from src.span import Span, jaccard_similarity_for_span_lists


class HasUncertaintyMethods(Protocol):
    """
    Protocol defining the uncertainty calculation methods provided by UncertaintyMixin.

    This protocol specifies the interface for uncertainty calculation methods that are
    implemented by the UncertaintyMixin class. It defines three different approaches
    to calculating uncertainty from multiple prediction samples:

    - Passage uncertainty: Based only on binary prediction consistency (concept present/absent)
    - Span uncertainty: Based only on overlap consistency between predicted spans
    - Combined uncertainty: Merges both binary (present/absent) and span overlap consistency

    Classes that inherit from UncertaintyMixin will automatically implement these methods
    and can be type-checked against this protocol.
    """

    def _calculate_passage_uncertainty(
        self, predictions: list[list[Span]]
    ) -> "Uncertainty": ...  # noqa: D102
    def _calculate_span_uncertainty(
        self, positive_predictions: list[list[Span]]
    ) -> "Uncertainty": ...  # noqa: D102
    def _calculate_combined_uncertainty(
        self, predictions: list[list[Span]]
    ) -> "Uncertainty": ...  # noqa: D102


class SupportsUncertainty(Protocol):
    """
    Protocol for classes that support uncertainty estimation through sampling.

    This protocol defines the core interface required for Monte Carlo uncertainty
    estimation. Classes implementing this protocol can:

    1. Create variant instances of themselves (with stochastic variation)
    2. Make predictions on text

    The uncertainty estimation process works by:
    - Creating multiple variant classifiers using get_variant_sub_classifier()
    - Running predictions with each variant on the same text
    - Analyzing the variance in predictions to estimate uncertainty

    Different classifier types may implement their variant subclassifiers differently.

    This protocol ensures type safety when combining classifiers with uncertainty
    estimation capabilities.
    """

    def get_variant_sub_classifier(self) -> "SupportsUncertainty": ...  # noqa: D102
    def predict(self, text: str) -> list[Span]: ...  # noqa: D102


class UncertaintyCapableClassifier(
    SupportsUncertainty, HasUncertaintyMethods, Protocol
):
    """
    Protocol combining classifier prediction and uncertainty calculation capabilities.

    This protocol represents the complete interface for classifiers that can perform
    uncertainty estimation. It combines:

    1. SupportsUncertainty: Ability to create variants and make predictions
    2. HasUncertaintyMethods: Uncertainty calculation algorithms

    The purpose of this protocol is to enable type-safe uncertainty estimation.
    The UncertaintyMixin.predict_with_uncertainty() method uses this protocol
    as a type constraint, ensuring it can only be called on objects that
    implement both sets of required methods.

    The design prevents runtime errors where uncertainty estimation might be
    attempted on classifiers that don't properly support the required operations.

    Example:
        class MyClassifier(Classifier, UncertaintyMixin):
            # This automatically implements UncertaintyCapableClassifier protocol,
            # allowing us to safely call predict_with_uncertainty()
            pass
    """


class Uncertainty(RootModel[Annotated[float, Field(ge=0.0, le=1.0)]])::
    """
    A validated float representing uncertainty as a number, inclusive of 0.0 and 1.0.

    Lower values represent high confidence, with 0 representing total certainty.
    Higher values represent low confidence, with 1 representing total uncertainty.
    """

    def __str__(self) -> str:
         return str(self.root)
         
    def __repr__(self) -> str:
        return f"Uncertainty({self.root})"
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

        binary_predictions = [1 if spans else 0 for spans in predictions]

        if len(set(binary_predictions)) <= 1:
            return Uncertainty(0.0)

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
