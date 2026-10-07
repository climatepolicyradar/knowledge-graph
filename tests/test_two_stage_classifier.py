from datetime import datetime

import pytest

from knowledge_graph.classifier.classifier import Classifier, ZeroShotClassifier
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

    @property
    def id(self) -> ClassifierID:
        """Return a deterministic identifier for the test classifier."""
        return ClassifierID.generate(self.name, self.concept.id)


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
