import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.classifier.uncertainty_mixin import Uncertainty, UncertaintyMixin
from src.span import Span
from tests.common_strategies import text_strategy


class MixinTester(UncertaintyMixin):
    """Test class to access UncertaintyMixin methods."""

    pass


# Module-level instance for Hypothesis tests (avoids fixture scope issues)
mixin_tester = MixinTester()


@pytest.fixture
def sample_text():
    """Fixture providing sample text for span creation."""
    return "This is a sample text for testing uncertainty calculations."


@pytest.fixture
def sample_spans(sample_text):
    """Fixture providing various sample spans for testing."""
    return {
        "span_a": Span(text=sample_text, start_index=0, end_index=4),
        "span_b": Span(text=sample_text, start_index=10, end_index=16),
        "span_c": Span(text=sample_text, start_index=17, end_index=21),
        "overlapping": Span(text=sample_text, start_index=2, end_index=8),
    }


@st.composite
def consistent_negative_predictions_strategy(draw):
    """Generates a list of empty list, representing consistent negative predictions."""
    num_predictions = draw(st.integers(min_value=2, max_value=10))
    return [[] for _ in range(num_predictions)]


@st.composite
def consistent_positive_predictions_high_agreement_strategy(draw):
    """Generates predictions that are all positive and in perfect agreement."""
    text = draw(text_strategy)
    start_index = draw(st.integers(min_value=0, max_value=len(text) - 2))
    end_index = draw(st.integers(min_value=start_index + 1, max_value=len(text) - 1))
    span = Span(text=text, start_index=start_index, end_index=end_index)
    num_predictions = draw(st.integers(min_value=2, max_value=10))
    return [[span] for _ in range(num_predictions)]


@st.composite
def inconsistent_passage_predictions_strategy(draw):
    """Generates a mix of positive and negative predictions."""
    num_predictions = draw(st.integers(min_value=2, max_value=20))
    num_positive = draw(st.integers(min_value=1, max_value=num_predictions - 1))

    text = draw(text_strategy)
    start_index = draw(st.integers(min_value=0, max_value=len(text) - 2))
    end_index = draw(st.integers(min_value=start_index + 1, max_value=len(text) - 1))
    positive_prediction = [
        Span(text=text, start_index=start_index, end_index=end_index)
    ]

    predictions = [positive_prediction] * num_positive + [[]] * (
        num_predictions - num_positive
    )
    draw(st.randoms()).shuffle(predictions)
    return predictions


@st.composite
def consistent_positive_predictions_low_agreement_strategy(draw):
    """Generates positive predictions with low agreement on span location."""
    text = draw(text_strategy.filter(lambda t: len(t) >= 100))
    num_predictions = draw(st.integers(min_value=2, max_value=10))

    # Create two non-overlapping spans
    start_a = draw(st.integers(min_value=0, max_value=len(text) // 2 - 1))
    end_a = draw(st.integers(min_value=start_a + 1, max_value=len(text) // 2))
    span_a = Span(text=text, start_index=start_a, end_index=end_a)

    start_b = draw(st.integers(min_value=end_a, max_value=len(text) - 1))
    end_b = draw(st.integers(min_value=start_b + 1, max_value=len(text)))
    span_b = Span(text=text, start_index=start_b, end_index=end_b)

    # Alternate between the two spans for low agreement
    predictions = []
    for i in range(num_predictions):
        if i % 2 == 0:
            predictions.append([span_a])
        else:
            predictions.append([span_b])
    return predictions


@st.composite
def overlapping_spans_predictions_strategy(draw):
    """Generates positive predictions with partially overlapping spans."""
    text = draw(text_strategy.filter(lambda t: len(t) >= 50))
    num_predictions = draw(st.integers(min_value=2, max_value=5))

    # Create overlapping spans
    base_start = draw(st.integers(min_value=0, max_value=len(text) // 2))
    base_end = draw(
        st.integers(min_value=base_start + 10, max_value=len(text) // 2 + 20)
    )

    predictions = []
    for i in range(num_predictions):
        # Create slightly different overlapping spans
        offset = draw(st.integers(min_value=-5, max_value=5))
        start = max(0, base_start + offset)
        end = min(len(text), base_end + offset)
        if start < end:
            span = Span(text=text, start_index=start, end_index=end)
            predictions.append([span])
        else:
            predictions.append([])

    return [p for p in predictions if p]  # Filter out empty predictions


@st.composite
def multiple_spans_per_prediction_strategy(draw):
    """Generates predictions where each has multiple spans."""
    text = draw(text_strategy.filter(lambda t: len(t) >= 100))
    num_predictions = draw(st.integers(min_value=2, max_value=5))

    predictions = []
    for _ in range(num_predictions):
        # Create 2-3 spans per prediction
        num_spans = draw(st.integers(min_value=1, max_value=3))
        spans = []
        for _ in range(num_spans):
            start = draw(st.integers(min_value=0, max_value=len(text) - 10))
            end = draw(
                st.integers(min_value=start + 1, max_value=min(start + 20, len(text)))
            )
            spans.append(Span(text=text, start_index=start, end_index=end))
        predictions.append(spans)

    return predictions


# Test the Uncertainty class itself
@pytest.mark.parametrize(
    "value,expected",
    [
        (0.0, 0.0),
        (0.5, 0.5),
        (1.0, 1.0),
        (0, 0.0),
        (1, 1.0),
    ],
)
def test_uncertainty_validation_accepts_valid_values(value, expected):
    """Test that Uncertainty accepts valid values between 0 and 1."""
    assert Uncertainty(value) == expected


@pytest.mark.parametrize("invalid_value", [-0.1, 1.1, 2.0, -1.0, 10.0])
def test_uncertainty_validation_rejects_invalid_values(invalid_value):
    """Test that Uncertainty rejects values outside the valid range."""
    with pytest.raises(
        ValueError,
        match=f"Values must be between 0 and 1. Got {invalid_value}",
    ):
        Uncertainty(invalid_value)


# Test passage uncertainty methods
@given(predictions=consistent_negative_predictions_strategy())
def test_passage_uncertainty_is_low_for_consistent_negative_predictions(predictions):
    uncertainty = mixin_tester._calculate_passage_uncertainty(predictions)
    assert uncertainty == Uncertainty(0.0)


@given(predictions=consistent_positive_predictions_high_agreement_strategy())
def test_passage_uncertainty_is_low_for_consistent_positive_predictions(predictions):
    uncertainty = mixin_tester._calculate_passage_uncertainty(predictions)
    assert uncertainty == Uncertainty(0.0)


@given(predictions=inconsistent_passage_predictions_strategy())
def test_passage_uncertainty_is_high_for_inconsistent_predictions(predictions):
    uncertainty = mixin_tester._calculate_passage_uncertainty(predictions)
    assert uncertainty > 0.0
    # check the math: uncertainty = 4 * p * (1-p)
    positive_ratio = sum(1 for p in predictions if p) / len(predictions)
    expected_uncertainty = 4 * positive_ratio * (1 - positive_ratio)
    assert uncertainty == pytest.approx(Uncertainty(expected_uncertainty))


@pytest.mark.parametrize(
    "positive_ratio,expected_uncertainty",
    [
        (0.5, 1.0),  # 50/50 split gives maximum uncertainty
        (0.75, 0.75),  # 75/25 split: 4 * 0.75 * 0.25 = 0.75
        (0.25, 0.75),  # 25/75 split: 4 * 0.25 * 0.75 = 0.75
        (0.9, 0.36),  # 90/10 split: 4 * 0.9 * 0.1 = 0.36
    ],
)
def test_passage_uncertainty_specific_ratios(
    sample_spans, positive_ratio, expected_uncertainty
):
    """Test passage uncertainty calculation for specific positive/negative ratios."""
    total_predictions = 20
    num_positive = int(total_predictions * positive_ratio)
    num_negative = total_predictions - num_positive

    predictions = [[sample_spans["span_a"]]] * num_positive + [[]] * num_negative
    uncertainty = mixin_tester._calculate_passage_uncertainty(predictions)
    assert uncertainty == pytest.approx(Uncertainty(expected_uncertainty))


# Test span uncertainty methods
@given(predictions=consistent_positive_predictions_high_agreement_strategy())
def test_span_uncertainty_is_low_for_high_agreement_predictions(predictions):
    uncertainty = mixin_tester._calculate_span_uncertainty(predictions)
    assert uncertainty == Uncertainty(0.0)


@given(predictions=consistent_positive_predictions_low_agreement_strategy())
def test_span_uncertainty_is_high_for_low_agreement_predictions(predictions):
    uncertainty = mixin_tester._calculate_span_uncertainty(predictions)
    assert uncertainty > 0.9


@given(predictions=overlapping_spans_predictions_strategy())
def test_span_uncertainty_with_partial_overlap(predictions):
    """Test span uncertainty with partially overlapping spans."""
    if len(predictions) >= 2:
        uncertainty = mixin_tester._calculate_span_uncertainty(predictions)
        # Should be between 0 and 1, with some uncertainty due to partial overlap
        assert 0.0 <= uncertainty <= 1.0


@given(predictions=multiple_spans_per_prediction_strategy())
def test_span_uncertainty_with_multiple_spans_per_prediction(predictions):
    """Test span uncertainty when predictions contain multiple spans."""
    uncertainty = mixin_tester._calculate_span_uncertainty(predictions)
    assert 0.0 <= uncertainty <= 1.0


def test_single_prediction_span_uncertainty(sample_spans):
    """Test span uncertainty with only one positive prediction."""
    uncertainty = mixin_tester._calculate_span_uncertainty([[sample_spans["span_a"]]])
    assert uncertainty == Uncertainty(0.0)


# Test combined uncertainty methods
@given(
    predictions=st.one_of(
        consistent_negative_predictions_strategy(),
        consistent_positive_predictions_high_agreement_strategy(),
    )
)
def test_combined_uncertainty_is_low_for_consistent_predictions(predictions):
    uncertainty = mixin_tester._calculate_combined_uncertainty(predictions)
    assert uncertainty == Uncertainty(0.0)


@given(predictions=inconsistent_passage_predictions_strategy())
def test_combined_uncertainty_is_high_for_inconsistent_passage_predictions(predictions):
    uncertainty = mixin_tester._calculate_combined_uncertainty(predictions)
    assert uncertainty > 0.0


@given(predictions=consistent_positive_predictions_low_agreement_strategy())
def test_combined_uncertainty_is_high_for_low_agreement_span_predictions(predictions):
    uncertainty = mixin_tester._calculate_combined_uncertainty(predictions)
    # When passage uncertainty is 0, combined is 0.3 * span_uncertainty
    # and span uncertainty should be high
    assert uncertainty > 0.25


def test_empty_predictions_list():
    """Test behavior with empty predictions list."""
    uncertainty = mixin_tester._calculate_combined_uncertainty([])
    assert uncertainty == Uncertainty(0.0)


def test_combined_uncertainty_with_single_positive_prediction(sample_spans):
    """Test combined uncertainty when there's only one positive prediction."""
    predictions = [[], [sample_spans["span_a"]], []]  # mixed with mostly negative
    uncertainty = mixin_tester._calculate_combined_uncertainty(predictions)
    # Should be based only on binary uncertainty since <2 positive predictions
    expected_binary = mixin_tester._calculate_passage_uncertainty(predictions)
    assert uncertainty == expected_binary


def test_combined_uncertainty_weights_sum_to_one(sample_spans):
    """Test that the binary and overlap weights sum to 1.0."""
    # This is tested in the assertion in the code, but let's verify it explicitly
    # by checking that the method doesn't raise an AssertionError
    predictions = [
        [sample_spans["span_a"]],
        [sample_spans["span_b"]],
    ]  # Different spans for both predictions

    try:
        uncertainty = mixin_tester._calculate_combined_uncertainty(predictions)
        # If we get here, the weights sum correctly
        assert 0.0 <= uncertainty <= 1.0
    except AssertionError as e:
        if "Weights must sum to 1.0" in str(e):
            pytest.fail("Binary and overlap weights do not sum to 1.0")
        else:
            raise


# Mock classifier for testing predict_with_uncertainty
class MockClassifier(UncertaintyMixin):
    """Mock classifier for testing predict_with_uncertainty."""

    def __init__(self, predictions_sequence: list[list[Span]]):
        self.predictions_sequence = predictions_sequence
        self.call_count = 0

    def get_variant_sub_classifier(self):
        """Return a variant that cycles through the prediction sequence."""
        variant = MockClassifier(self.predictions_sequence)
        variant.call_count = self.call_count
        return variant

    def predict(self, text: str) -> list[Span]:
        """Return the next prediction in the sequence."""
        if self.call_count < len(self.predictions_sequence):
            result = self.predictions_sequence[self.call_count]
            self.call_count += 1
            return result
        return []


@pytest.fixture
def mock_classifier_with_mixed_predictions(sample_spans):
    """Fixture providing a mock classifier with mixed prediction results."""
    predictions_sequence = [[sample_spans["span_a"]], [], [sample_spans["span_a"]]]
    return MockClassifier(predictions_sequence)


@pytest.fixture
def mock_classifier_with_different_spans(sample_spans):
    """Fixture providing a mock classifier with different span predictions."""
    predictions_sequence = [[sample_spans["span_a"]], [sample_spans["span_b"]]]
    return MockClassifier(predictions_sequence)


@pytest.mark.parametrize(
    "method,expected_samples",
    [
        ("combined", 3),
        ("passage", 2),
        ("span", 2),
    ],
)
def test_predict_with_uncertainty_methods(
    sample_text, mock_classifier_with_mixed_predictions, method, expected_samples
):
    """Test predict_with_uncertainty with different methods."""
    predictions, uncertainty = (
        mock_classifier_with_mixed_predictions.predict_with_uncertainty(
            sample_text, num_samples=expected_samples, method=method
        )
    )

    assert len(predictions) == expected_samples
    assert isinstance(uncertainty, Uncertainty)
    assert 0.0 <= uncertainty <= 1.0


def test_predict_with_uncertainty_invalid_method():
    """Test predict_with_uncertainty with invalid method raises ValueError."""
    mock_classifier = MockClassifier([])

    with pytest.raises(ValueError, match="Invalid uncertainty method: invalid"):
        mock_classifier.predict_with_uncertainty(
            "test",
            num_samples=1,
            method="invalid",  # type: ignore
        )


# Boundary condition tests
@given(
    num_positive=st.integers(min_value=0, max_value=20),
    num_negative=st.integers(min_value=0, max_value=20),
)
def test_passage_uncertainty_boundary_conditions(num_positive, num_negative):
    """Test passage uncertainty across different positive/negative ratios."""
    if num_positive + num_negative == 0:
        return  # Skip empty case

    text = "test text"
    span = Span(text=text, start_index=0, end_index=4)
    predictions = [[span]] * num_positive + [[]] * num_negative

    uncertainty = mixin_tester._calculate_passage_uncertainty(predictions)

    if num_positive == 0 or num_negative == 0:
        # All same prediction type should give 0 uncertainty
        assert uncertainty == Uncertainty(0.0)
    else:
        # Mixed should give some uncertainty
        assert uncertainty > 0.0
        assert uncertainty <= 1.0

        # Verify the math: 4 * p * (1-p)
        p = num_positive / (num_positive + num_negative)
        expected = 4 * p * (1 - p)
        assert uncertainty == pytest.approx(Uncertainty(expected))
