from datetime import datetime
from typing import Sequence
from unittest.mock import patch

import pytest

from knowledge_graph.classifier.classifier import (
    Classifier,
    GPUBoundClassifier,
    ZeroShotClassifier,
)
from knowledge_graph.classifier.keyword import KeywordClassifier
from knowledge_graph.classifier.two_stage import TwoStageClassifier
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import ClassifierID, WikibaseID
from knowledge_graph.span import Span

GRANT = Concept(preferred_label="grant", wikibase_id=WikibaseID("Q1272"))
FINANCE = Concept(
    preferred_label="finance",
    alternative_labels=["funding"],
    wikibase_id=WikibaseID("Q1829"),
)


class RecordingFilter(Classifier, ZeroShotClassifier):
    """Fires on whole passages containing 'money', recording what it was shown."""

    def __init__(self, concept: Concept):
        super().__init__(concept)
        self.seen: list[str] = []
        self.thresholds: list[float | None] = []
        self.batches: list[list[str]] = []

    def _predict(self, text: str, threshold: float | None = None) -> list[Span]:
        self.seen.append(text)
        self.thresholds.append(threshold)
        if "money" not in text:
            return []
        return [
            Span(
                text=text,
                start_index=0,
                end_index=len(text),
                concept_id=self.concept.wikibase_id,
                prediction_probability=0.9,
                labellers=["filter"],
                timestamps=[datetime.now()],
            )
        ]

    def _predict_batch(
        self, texts: Sequence[str], threshold: float | None = None
    ) -> list[list[Span]]:
        self.batches.append(list(texts))
        return super()._predict_batch(texts, threshold=threshold)

    @property
    def id(self) -> ClassifierID:
        """Return a deterministic identifier for the test classifier."""
        return ClassifierID.generate(self.name, self.concept.id)


class GPUBoundCandidate(KeywordClassifier, GPUBoundClassifier):
    """A keyword classifier marked as GPU-bound, standing in for a BERT model."""


def test_only_labels_when_both_classifiers_fire():
    clf = TwoStageClassifier(GRANT, filter_classifier=KeywordClassifier(FINANCE))
    results = clf.predict(
        [
            "The grant provides finance for adaptation.",
            "The court may grant permission.",
            "Climate finance is growing.",
        ]
    )
    assert [bool(r) for r in results] == [True, False, False]
    span = results[0][0]
    assert span.labelled_text.lower() == "grant"
    assert span.concept_id == GRANT.wikibase_id
    assert len(span.labellers) == len(span.timestamps) == 1


def test_candidate_defaults_to_keyword_classifier():
    clf = TwoStageClassifier(GRANT, filter_classifier=KeywordClassifier(FINANCE))
    assert clf.candidate_classifier == KeywordClassifier(GRANT)


def test_filter_only_sees_candidate_passages_and_gets_threshold():
    filter_classifier = RecordingFilter(FINANCE)
    clf = TwoStageClassifier(
        GRANT, filter_classifier=filter_classifier, filter_threshold="0.7"
    )
    results = clf.predict(["a grant of money", "no keyword here", "grant me patience"])
    assert filter_classifier.seen == ["a grant of money", "grant me patience"]
    assert filter_classifier.thresholds == [0.7, 0.7]
    assert [bool(r) for r in results] == [True, False, False]
    assert results[0][0].prediction_probability == 0.9

    filter_classifier.thresholds.clear()
    clf.predict(["a grant of money"], threshold=0.2)
    assert filter_classifier.thresholds == [0.2]


def test_id_is_deterministic_and_depends_on_sub_classifiers():
    a = TwoStageClassifier(GRANT, filter_classifier=KeywordClassifier(FINANCE))
    b = TwoStageClassifier(GRANT, filter_classifier=KeywordClassifier(FINANCE))
    c = TwoStageClassifier(GRANT, filter_classifier=RecordingFilter(FINANCE))
    assert a.id == b.id
    assert a.id != c.id
    assert a.id != a.candidate_classifier.id


def test_candidate_concept_must_match():
    with pytest.raises(ValueError):
        TwoStageClassifier(
            GRANT,
            filter_classifier=KeywordClassifier(FINANCE),
            candidate_classifier=KeywordClassifier(FINANCE),
        )


def test_save_and_load_round_trip(tmp_path):
    clf = TwoStageClassifier(GRANT, filter_classifier=KeywordClassifier(FINANCE))
    path = tmp_path / "model.pickle"
    clf.save(path)
    loaded = Classifier.load(path)
    assert isinstance(loaded, TwoStageClassifier)
    assert loaded.id == clf.id
    assert loaded.predict("The grant provides finance.") != []


def test_filter_gets_full_batches_of_candidate_passages():
    filter_classifier = RecordingFilter(FINANCE)
    clf = TwoStageClassifier(GRANT, filter_classifier=filter_classifier)
    texts = ["a grant of money", "nothing here", "nothing here either", "grant money"]

    clf.predict(texts, batch_size=2)

    # Both hits arrive in one full batch, rather than one per input batch
    assert filter_classifier.batches == [["a grant of money", "grant money"]]


def test_warns_when_candidate_is_gpu_bound_and_filter_is_not():
    # Patch the module's logger directly rather than relying on caplog: in some
    # contexts get_logger() returns a Prefect run logger that does not propagate
    # to caplog's root handler.
    with patch("knowledge_graph.classifier.two_stage.get_logger") as mock_get_logger:
        TwoStageClassifier(
            GRANT,
            filter_classifier=KeywordClassifier(FINANCE),
            candidate_classifier=GPUBoundCandidate(GRANT),
        )
    warnings = [str(call.args[0]) for call in mock_get_logger().warning.call_args_list]
    assert any("cheaper of the two" in message for message in warnings)
