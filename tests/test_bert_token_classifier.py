"""Tests for BertTokenClassifier alignment and span reconstruction utilities."""

import numpy as np
import pytest
from transformers import EvalPrediction

from knowledge_graph.classifier.bert_token_classifier import (
    B_LABEL,
    I_LABEL,
    IGNORE_LABEL,
    O_LABEL,
    _align_labels_with_tokens,
    _compute_token_metrics,
    _reconstruct_spans_from_predictions,
)
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.span import Span


@pytest.fixture
def concept_id():
    return WikibaseID("Q42")


@pytest.fixture
def sample_text():
    return "Climate change affects biodiversity in many regions."


def _make_span(text: str, start: int, end: int, concept_id: WikibaseID) -> Span:
    return Span(
        text=text,
        start_index=start,
        end_index=end,
        concept_id=concept_id,
    )


class TestAlignLabelsWithTokens:
    """Tests for _align_labels_with_tokens"""

    def test_no_spans_all_outside(self, concept_id):
        """All real tokens should be O when there are no gold spans."""
        offset_mapping = [(0, 0), (0, 7), (8, 14), (0, 0)]  # CLS, 2 tokens, SEP
        labels = _align_labels_with_tokens(offset_mapping, [], concept_id)
        assert labels == [IGNORE_LABEL, O_LABEL, O_LABEL, IGNORE_LABEL]

    def test_single_span_bio_labels(self, sample_text, concept_id):
        """A span covering multiple tokens should produce B then I labels."""
        span = _make_span(sample_text, 0, 14, concept_id)  # "Climate change"

        # Simulated tokens: [CLS] "Climate" " change" " affects" [SEP]
        offset_mapping = [(0, 0), (0, 7), (7, 14), (14, 22), (0, 0)]
        labels = _align_labels_with_tokens(offset_mapping, [span], concept_id)
        assert labels == [IGNORE_LABEL, B_LABEL, I_LABEL, O_LABEL, IGNORE_LABEL]

    def test_single_token_span(self, sample_text, concept_id):
        """A span covering exactly one token should produce just B."""

        span = _make_span(sample_text, 0, 7, concept_id)  # "Climate"
        offset_mapping = [(0, 0), (0, 7), (7, 14), (0, 0)]
        labels = _align_labels_with_tokens(offset_mapping, [span], concept_id)
        assert labels == [IGNORE_LABEL, B_LABEL, O_LABEL, IGNORE_LABEL]

    def test_multiple_separate_spans(self, concept_id):
        """Two separate spans should each start with B."""

        text = "Climate change affects biodiversity and ecosystems."
        span1 = _make_span(text, 0, 14, concept_id)  # "Climate change"
        span2 = _make_span(text, 23, 36, concept_id)  # "biodiversity"

        # [CLS] "Climate" " change" " affects" " biodiversity" " and" [SEP]
        offset_mapping = [(0, 0), (0, 7), (7, 14), (14, 23), (23, 36), (36, 40), (0, 0)]
        labels = _align_labels_with_tokens(offset_mapping, [span1, span2], concept_id)
        assert labels == [
            IGNORE_LABEL,
            B_LABEL,
            I_LABEL,
            O_LABEL,
            B_LABEL,
            O_LABEL,
            IGNORE_LABEL,
        ]

    def test_adjacent_spans_get_separate_b_labels(self, concept_id):
        """Two adjacent spans should each start with B (BIO separates them)."""

        text = "AB"
        span1 = _make_span(text, 0, 1, concept_id)  # "A"
        span2 = _make_span(text, 1, 2, concept_id)  # "B"
        offset_mapping = [(0, 0), (0, 1), (1, 2), (0, 0)]
        labels = _align_labels_with_tokens(offset_mapping, [span1, span2], concept_id)
        assert labels == [IGNORE_LABEL, B_LABEL, B_LABEL, IGNORE_LABEL]

    def test_partial_token_overlap(self, concept_id):
        """A token partially overlapping a span should still be labelled."""

        text = "abcdefghij"
        span = _make_span(text, 2, 5, concept_id)  # "cde"
        # Token covers chars 1-6, overlapping the span
        offset_mapping = [(0, 0), (0, 2), (2, 6), (6, 10), (0, 0)]
        labels = _align_labels_with_tokens(offset_mapping, [span], concept_id)
        assert labels == [IGNORE_LABEL, O_LABEL, B_LABEL, O_LABEL, IGNORE_LABEL]

    def test_different_concept_id_ignored(self, concept_id):
        """Spans with a different concept_id should be ignored."""

        text = "Climate change"
        other_concept = WikibaseID("Q999")
        span = _make_span(text, 0, 7, other_concept)
        offset_mapping = [(0, 0), (0, 7), (7, 14), (0, 0)]
        labels = _align_labels_with_tokens(offset_mapping, [span], concept_id)
        assert labels == [IGNORE_LABEL, O_LABEL, O_LABEL, IGNORE_LABEL]

    def test_empty_offset_mapping(self, concept_id):
        """Empty offset mapping should return empty labels."""

        labels = _align_labels_with_tokens([], [], concept_id)
        assert labels == []


class TestReconstructSpans:
    """Tests for _reconstruct_spans_from_predictions"""

    def test_no_positive_predictions(self, sample_text, concept_id):
        """All O predictions should produce no spans."""

        token_labels = [O_LABEL, O_LABEL, O_LABEL]
        token_probs = [0.95, 0.92, 0.90]
        offset_mapping = [(0, 0), (0, 7), (7, 14)]
        spans = _reconstruct_spans_from_predictions(
            token_labels,
            token_probs,
            offset_mapping,
            sample_text,
            concept_id,
            "test",
        )
        assert spans == []

    def test_single_b_token_span(self, sample_text, concept_id):
        """A single B token should create a span."""

        token_labels = [O_LABEL, B_LABEL, O_LABEL]
        token_probs = [0.95, 0.85, 0.90]
        offset_mapping = [(0, 0), (0, 7), (7, 14)]
        spans = _reconstruct_spans_from_predictions(
            token_labels,
            token_probs,
            offset_mapping,
            sample_text,
            concept_id,
            "test",
        )
        assert len(spans) == 1
        assert spans[0].start_index == 0
        assert spans[0].end_index == 7
        assert spans[0].labelled_text == "Climate"

    def test_b_then_i_merged(self, sample_text, concept_id):
        """B followed by I tokens should merge into one span."""

        # [CLS] "Climate" " change" " affects" [SEP]
        token_labels = [O_LABEL, B_LABEL, I_LABEL, O_LABEL, O_LABEL]
        token_probs = [0.9, 0.85, 0.80, 0.95, 0.9]
        offset_mapping = [(0, 0), (0, 7), (7, 14), (14, 22), (0, 0)]
        spans = _reconstruct_spans_from_predictions(
            token_labels,
            token_probs,
            offset_mapping,
            sample_text,
            concept_id,
            "test",
        )
        assert len(spans) == 1
        assert spans[0].start_index == 0
        assert spans[0].end_index == 14
        assert spans[0].labelled_text == "Climate change"
        assert spans[0].prediction_probability == pytest.approx(0.825)

    def test_two_separate_spans(self, concept_id):
        """Two B tokens separated by O should produce two spans."""

        text = "Climate change affects biodiversity significantly."
        # [CLS] "Climate" " change" " affects" " biodiversity" " significantly" [SEP]
        token_labels = [O_LABEL, B_LABEL, I_LABEL, O_LABEL, B_LABEL, O_LABEL, O_LABEL]
        token_probs = [0.9, 0.85, 0.80, 0.95, 0.90, 0.88, 0.9]
        offset_mapping = [(0, 0), (0, 7), (7, 14), (14, 23), (23, 37), (37, 51), (0, 0)]
        spans = _reconstruct_spans_from_predictions(
            token_labels,
            token_probs,
            offset_mapping,
            text,
            concept_id,
            "test",
        )
        assert len(spans) == 2
        assert spans[0].labelled_text == "Climate change"
        assert spans[1].start_index == 23

    def test_orphaned_i_label_ignored(self, sample_text, concept_id):
        """An I label not preceded by B should not create a span."""

        token_labels = [O_LABEL, O_LABEL, I_LABEL, O_LABEL]
        token_probs = [0.9, 0.95, 0.80, 0.90]
        offset_mapping = [(0, 0), (0, 7), (7, 14), (14, 22)]
        spans = _reconstruct_spans_from_predictions(
            token_labels,
            token_probs,
            offset_mapping,
            sample_text,
            concept_id,
            "test",
        )
        assert spans == []

    def test_short_spans_filtered(self, concept_id):
        """Spans shorter than min_span_chars should be filtered out."""

        text = "A B C D"
        token_labels = [B_LABEL, O_LABEL]
        token_probs = [0.85, 0.90]
        offset_mapping = [(0, 1), (1, 3)]
        spans = _reconstruct_spans_from_predictions(
            token_labels,
            token_probs,
            offset_mapping,
            text,
            concept_id,
            "test",
            min_span_chars=2,
        )
        assert spans == []

    def test_span_at_end_of_sequence(self, concept_id):
        """A span at the end of the sequence should finish at the appropriate place."""

        text = "affects biodiversity"
        token_labels = [O_LABEL, B_LABEL]
        token_probs = [0.90, 0.85]
        offset_mapping = [(0, 7), (8, 20)]
        spans = _reconstruct_spans_from_predictions(
            token_labels,
            token_probs,
            offset_mapping,
            text,
            concept_id,
            "test",
        )
        assert len(spans) == 1
        assert spans[0].start_index == 8
        assert spans[0].end_index == 20

    def test_empty_inputs(self, concept_id):
        """Empty inputs should produce no spans."""

        spans = _reconstruct_spans_from_predictions(
            [],
            [],
            [],
            "some text",
            concept_id,
            "test",
        )
        assert spans == []


def _make_eval_prediction(
    predictions: list[list[list[float]]], labels: list[list[int]]
) -> EvalPrediction:
    """Helper: build an EvalPrediction from raw logits and label IDs."""
    return EvalPrediction(
        predictions=np.array(predictions, dtype=np.float32),
        label_ids=np.array(labels, dtype=np.int64),
    )


class TestComputeTokenMetrics:
    """Tests for _compute_token_metrics"""

    def test_perfect_predictions(self):
        """All correct predictions should give F1=1, accuracy=1."""
        # 1 sequence: [O, B, I] with matching gold labels
        preds = [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        ]
        labels = [[O_LABEL, B_LABEL, I_LABEL]]
        result = _compute_token_metrics(_make_eval_prediction(preds, labels))
        assert result["accuracy"] == pytest.approx(1.0)
        assert result["f1"] == pytest.approx(1.0)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)

    def test_all_wrong_predictions(self):
        """Predicting all O when gold has B/I should give recall=0, F1=0."""
        preds = [
            [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        ]
        labels = [[O_LABEL, B_LABEL, I_LABEL]]
        result = _compute_token_metrics(_make_eval_prediction(preds, labels))
        assert result["recall"] == pytest.approx(0.0)
        assert result["f1"] == pytest.approx(0.0)

    def test_ignore_label_positions_excluded(self):
        """IGNORE_LABEL positions (CLS/SEP/PAD) should not affect metrics."""
        # IGNORE_LABEL at position 0 and 3 (CLS/SEP), real tokens at 1 and 2
        preds = [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        ]
        labels = [[IGNORE_LABEL, B_LABEL, I_LABEL, IGNORE_LABEL]]
        result = _compute_token_metrics(_make_eval_prediction(preds, labels))
        # Only positions 1 and 2 count: both predicted correctly
        assert result["accuracy"] == pytest.approx(1.0)
        assert result["f1"] == pytest.approx(1.0)

    def test_token_level_not_span_level(self):
        """A 3-token entity contributes 3 TPs, not 1 (token-level semantics)."""
        # Gold: B I I (one 3-token entity). Predict: B I I (perfect).
        preds = [
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        ]
        labels = [[B_LABEL, I_LABEL, I_LABEL]]
        result = _compute_token_metrics(_make_eval_prediction(preds, labels))
        # 3 TP, 0 FP, 0 FN → precision=1, recall=1, f1=1
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)

    def test_partial_overlap(self):
        """Predicting only part of an entity: precision=1, recall=0.5."""
        # Gold: B I. Predict: B O (miss the I token).
        preds = [
            [[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]],
        ]
        labels = [[B_LABEL, I_LABEL]]
        result = _compute_token_metrics(_make_eval_prediction(preds, labels))
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(0.5)
        assert result["f1"] == pytest.approx(2 / 3)
