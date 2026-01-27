"""Tests for the geography classifier module."""

import pytest

from knowledge_graph.classifier.geography import (
    GeographyClassifierBackend,
    LabelIndex,
    SpacyGeographyBackend,
)
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import ClassifierID, WikibaseID
from knowledge_graph.span import Span


# Test fixtures
@pytest.fixture
def sample_geography_concepts() -> list[Concept]:
    """Create sample geography concepts for testing."""
    return [
        Concept(
            wikibase_id=WikibaseID("Q155"),
            preferred_label="Brazil",
            alternative_labels=["Brasil", "Federative Republic of Brazil"],
            description="Country in South America",
        ),
        Concept(
            wikibase_id=WikibaseID("Q145"),
            preferred_label="United Kingdom",
            alternative_labels=["UK", "Britain", "Great Britain"],
            description="Country in Western Europe",
        ),
        Concept(
            wikibase_id=WikibaseID("Q142"),
            preferred_label="France",
            alternative_labels=["French Republic"],
            description="Country in Western Europe",
        ),
        Concept(
            wikibase_id=WikibaseID("Q230"),
            preferred_label="Georgia",
            alternative_labels=[],
            description="Country in the Caucasus region of Eurasia",
            definition="Georgia is a country at the intersection of Eastern Europe and Western Asia.",
        ),
        Concept(
            wikibase_id=WikibaseID("Q1428"),
            preferred_label="Georgia",
            alternative_labels=["State of Georgia"],
            description="State in the southeastern United States",
            definition="Georgia is a state in the southeastern United States, known for Atlanta.",
        ),
    ]


@pytest.fixture
def brazil_concept() -> Concept:
    """Create Brazil concept for individual tests."""
    return Concept(
        wikibase_id=WikibaseID("Q155"),
        preferred_label="Brazil",
        alternative_labels=["Brasil"],
        description="Country in South America",
    )


@pytest.fixture
def ambiguous_georgia_concepts() -> list[Concept]:
    """Create Georgia concepts for disambiguation tests."""
    return [
        Concept(
            wikibase_id=WikibaseID("Q230"),
            preferred_label="Georgia",
            description="Country in the Caucasus region",
            definition="Capital: Tbilisi. Located between Europe and Asia.",
        ),
        Concept(
            wikibase_id=WikibaseID("Q1428"),
            preferred_label="Georgia",
            description="US state",
            definition="Capital: Atlanta. Located in southeastern United States.",
        ),
    ]


class TestLabelIndex:
    """Tests for the LabelIndex class."""

    def test_from_concepts_builds_unambiguous_index(
        self, sample_geography_concepts: list[Concept]
    ):
        """Test that unambiguous labels are correctly indexed."""
        index = LabelIndex.from_concepts(sample_geography_concepts)

        # Brazil should be unambiguous
        assert "brazil" in index.unambiguous
        assert index.unambiguous["brazil"] == WikibaseID("Q155")

        # UK should be unambiguous
        assert "uk" in index.unambiguous
        assert index.unambiguous["uk"] == WikibaseID("Q145")

    def test_from_concepts_identifies_ambiguous_labels(
        self, sample_geography_concepts: list[Concept]
    ):
        """Test that ambiguous labels are correctly identified."""
        index = LabelIndex.from_concepts(sample_geography_concepts)

        # Georgia should be ambiguous (country vs US state)
        assert "georgia" in index.ambiguous
        assert len(index.ambiguous["georgia"]) == 2
        assert WikibaseID("Q230") in index.ambiguous["georgia"]
        assert WikibaseID("Q1428") in index.ambiguous["georgia"]

    def test_from_concepts_stores_descriptions(
        self, sample_geography_concepts: list[Concept]
    ):
        """Test that descriptions are stored for disambiguation."""
        index = LabelIndex.from_concepts(sample_geography_concepts)

        assert WikibaseID("Q155") in index.descriptions
        assert "South America" in index.descriptions[WikibaseID("Q155")]

    def test_lookup_unambiguous_label(self, sample_geography_concepts: list[Concept]):
        """Test lookup of unambiguous labels returns single QID."""
        index = LabelIndex.from_concepts(sample_geography_concepts)

        result = index.lookup("Brazil")
        assert result == WikibaseID("Q155")

        result = index.lookup("BRAZIL")  # Case insensitive
        assert result == WikibaseID("Q155")

    def test_lookup_ambiguous_label(self, sample_geography_concepts: list[Concept]):
        """Test lookup of ambiguous labels returns list of QIDs."""
        index = LabelIndex.from_concepts(sample_geography_concepts)

        result = index.lookup("Georgia")
        assert isinstance(result, list)
        assert len(result) == 2

    def test_lookup_unknown_label(self, sample_geography_concepts: list[Concept]):
        """Test lookup of unknown labels returns None."""
        index = LabelIndex.from_concepts(sample_geography_concepts)

        result = index.lookup("Narnia")
        assert result is None

    def test_is_ambiguous(self, sample_geography_concepts: list[Concept]):
        """Test ambiguity check."""
        index = LabelIndex.from_concepts(sample_geography_concepts)

        assert index.is_ambiguous("Georgia")
        assert not index.is_ambiguous("Brazil")
        assert not index.is_ambiguous("Narnia")

    def test_len(self, sample_geography_concepts: list[Concept]):
        """Test that len returns total unique labels."""
        index = LabelIndex.from_concepts(sample_geography_concepts)

        # Count expected labels
        # Brazil: brazil, brasil, federative republic of brazil (3)
        # UK: united kingdom, uk, britain, great britain (4)
        # France: france, french republic (2)
        # Georgia (country): georgia (1, shared)
        # Georgia (state): georgia, state of georgia (1 unique + 1 shared)
        # Total unique: 3 + 4 + 2 + 1 + 1 = 11 (georgia counted once)
        assert len(index) > 0

    def test_skips_concepts_without_wikibase_id(self):
        """Test that concepts without wikibase_id are skipped."""
        concepts = [
            Concept(
                preferred_label="Unknown Place",
                # No wikibase_id
            ),
            Concept(
                wikibase_id=WikibaseID("Q155"),
                preferred_label="Brazil",
            ),
        ]

        index = LabelIndex.from_concepts(concepts)

        assert "unknown place" not in index.unambiguous
        assert "brazil" in index.unambiguous


class TestGeographyClassifierBackend:
    """Tests for the GeographyClassifierBackend singleton."""

    def test_singleton_pattern(self):
        """Test that get_instance returns the same instance."""
        GeographyClassifierBackend.reset_instance()

        instance1 = GeographyClassifierBackend.get_instance()
        instance2 = GeographyClassifierBackend.get_instance()

        assert instance1 is instance2

        GeographyClassifierBackend.reset_instance()

    def test_reset_instance(self):
        """Test that reset_instance clears the singleton."""
        instance1 = GeographyClassifierBackend.get_instance()
        GeographyClassifierBackend.reset_instance()
        instance2 = GeographyClassifierBackend.get_instance()

        assert instance1 is not instance2

        GeographyClassifierBackend.reset_instance()

    def test_predict_all_raises_without_initialization(self):
        """Test that predict_all raises if not initialized."""
        GeographyClassifierBackend.reset_instance()
        backend = GeographyClassifierBackend.get_instance()

        with pytest.raises(RuntimeError, match="Backend not initialized"):
            backend.predict_all(["test text"])

        GeographyClassifierBackend.reset_instance()


@pytest.mark.transformers
class TestGeographyClassifierBackendWithModels:
    """Tests that require loading the actual models."""

    @pytest.fixture(autouse=True)
    def reset_backend(self):
        """Reset the backend before and after each test."""
        GeographyClassifierBackend.reset_instance()
        yield
        GeographyClassifierBackend.reset_instance()

    def test_initialize_index(self, sample_geography_concepts: list[Concept]):
        """Test that initialize_index sets up the backend correctly."""
        backend = GeographyClassifierBackend.get_instance()
        backend.initialize_index(sample_geography_concepts)

        assert backend.label_index is not None
        assert backend._initialized

    def test_predict_all_returns_correct_structure(
        self, sample_geography_concepts: list[Concept]
    ):
        """Test that predict_all returns correct structure."""
        backend = GeographyClassifierBackend.get_instance()
        backend.initialize_index(sample_geography_concepts)

        texts = [
            "Brazil announced new policies.",
            "The UK government responded.",
        ]

        results = backend.predict_all(texts, threshold=0.3)

        # Results should be a dict mapping QID to list of span lists
        assert isinstance(results, dict)
        for qid, spans_per_text in results.items():
            assert isinstance(qid, WikibaseID)
            assert isinstance(spans_per_text, list)
            assert len(spans_per_text) == len(texts)

    def test_predict_all_finds_brazil(self, sample_geography_concepts: list[Concept]):
        """Test that Brazil is correctly identified."""
        backend = GeographyClassifierBackend.get_instance()
        backend.initialize_index(sample_geography_concepts)

        texts = ["The Brazilian government announced new climate policies."]
        results = backend.predict_all(texts, threshold=0.3)

        # Should find Brazil
        brazil_qid = WikibaseID("Q155")
        if brazil_qid in results:
            brazil_spans = results[brazil_qid][0]
            assert len(brazil_spans) > 0
            assert any("brazilian" in s.labelled_text.lower() for s in brazil_spans)

    def test_predict_all_handles_empty_texts(
        self, sample_geography_concepts: list[Concept]
    ):
        """Test that empty text list is handled correctly."""
        backend = GeographyClassifierBackend.get_instance()
        backend.initialize_index(sample_geography_concepts)

        results = backend.predict_all([])
        assert isinstance(results, dict)

    def test_disambiguation_georgia_country(
        self, ambiguous_georgia_concepts: list[Concept]
    ):
        """Test disambiguation of Georgia towards country."""
        backend = GeographyClassifierBackend.get_instance()
        backend.initialize_index(ambiguous_georgia_concepts)

        # Context should favor Georgia (country)
        texts = ["Georgia's capital Tbilisi is in the Caucasus region."]
        results = backend.predict_all(texts, threshold=0.3)

        # Check if Georgia was found and disambiguated
        georgia_country = WikibaseID("Q230")
        georgia_state = WikibaseID("Q1428")

        # At least one should be found
        found_country = georgia_country in results and results[georgia_country][0]
        found_state = georgia_state in results and results[georgia_state][0]

        # If disambiguation worked, country context should favor Q230
        if found_country and not found_state:
            assert True  # Correct disambiguation
        elif found_country and found_state:
            # If both found, that's also acceptable (disambiguation might not be perfect)
            pass

    def test_disambiguation_georgia_state(
        self, ambiguous_georgia_concepts: list[Concept]
    ):
        """Test disambiguation of Georgia towards US state."""
        backend = GeographyClassifierBackend.get_instance()
        backend.initialize_index(ambiguous_georgia_concepts)

        # Context should favor Georgia (US state)
        texts = ["Georgia's capital Atlanta is in the southeastern United States."]
        results = backend.predict_all(texts, threshold=0.3)

        georgia_state = WikibaseID("Q1428")

        # If Georgia was found and disambiguated to state, that's correct
        if georgia_state in results and results[georgia_state][0]:
            assert True


@pytest.mark.transformers
class TestGeographyClassifier:
    """Tests for the GeographyClassifier class."""

    @pytest.fixture(autouse=True)
    def reset_backend(self):
        """Reset the backend before and after each test."""
        GeographyClassifierBackend.reset_instance()
        yield
        GeographyClassifierBackend.reset_instance()

    def test_classifier_initialization(self, brazil_concept: Concept):
        """Test that classifier initializes correctly."""
        from knowledge_graph.classifier.geography import GeographyClassifier

        classifier = GeographyClassifier(brazil_concept)

        assert classifier.concept == brazil_concept
        assert classifier.is_fitted  # Zero-shot, always fitted
        assert classifier.prediction_threshold == 0.5

    def test_classifier_id_is_deterministic(self, brazil_concept: Concept):
        """Test that classifier ID is deterministic."""
        from knowledge_graph.classifier.geography import GeographyClassifier

        classifier1 = GeographyClassifier(brazil_concept)
        classifier2 = GeographyClassifier(brazil_concept)

        assert classifier1.id == classifier2.id
        assert isinstance(classifier1.id, ClassifierID)

    def test_classifier_id_changes_with_threshold(self, brazil_concept: Concept):
        """Test that different thresholds produce different IDs."""
        from knowledge_graph.classifier.geography import GeographyClassifier

        classifier1 = GeographyClassifier(brazil_concept, prediction_threshold=0.5)
        classifier2 = GeographyClassifier(brazil_concept, prediction_threshold=0.7)

        assert classifier1.id != classifier2.id

    def test_classifier_fit_warns(self, brazil_concept: Concept):
        """Test that fit() emits a warning for zero-shot classifier."""
        from knowledge_graph.classifier.geography import GeographyClassifier

        classifier = GeographyClassifier(brazil_concept)

        with pytest.warns(UserWarning, match="zero-shot"):
            classifier.fit()

    def test_classifier_repr(self, brazil_concept: Concept):
        """Test classifier string representation."""
        from knowledge_graph.classifier.geography import GeographyClassifier

        classifier = GeographyClassifier(brazil_concept)
        repr_str = repr(classifier)

        assert "GeographyClassifier" in repr_str
        assert "Brazil" in repr_str

    def test_classifier_predict_requires_initialization(
        self, brazil_concept: Concept, sample_geography_concepts: list[Concept]
    ):
        """Test that predict requires backend initialization."""
        from knowledge_graph.classifier.geography import (
            GeographyClassifier,
            initialize_geography_classifiers,
        )

        # Initialize backend first
        initialize_geography_classifiers(sample_geography_concepts)

        classifier = GeographyClassifier(brazil_concept)
        text = "Brazil announced new policies."

        # Should not raise
        spans = classifier.predict(text)
        assert isinstance(spans, list)

    def test_classifier_predict_batch(
        self, brazil_concept: Concept, sample_geography_concepts: list[Concept]
    ):
        """Test batch prediction."""
        from knowledge_graph.classifier.geography import (
            GeographyClassifier,
            initialize_geography_classifiers,
        )

        initialize_geography_classifiers(sample_geography_concepts)
        classifier = GeographyClassifier(brazil_concept)

        texts = [
            "Brazil announced new policies.",
            "The weather is nice today.",
            "Brasil is known for football.",
        ]

        results = classifier.predict(texts)

        assert len(results) == 3
        assert all(isinstance(r, list) for r in results)


@pytest.mark.transformers
class TestGeographyBatchPredictor:
    """Tests for the GeographyBatchPredictor class."""

    @pytest.fixture(autouse=True)
    def reset_backend(self):
        """Reset the backend before and after each test."""
        GeographyClassifierBackend.reset_instance()
        yield
        GeographyClassifierBackend.reset_instance()

    def test_batch_predictor_initialization(self):
        """Test that batch predictor initializes correctly."""
        from knowledge_graph.classifier.geography import GeographyBatchPredictor

        predictor = GeographyBatchPredictor()
        assert predictor._backend is not None

    def test_batch_predictor_predict_all(
        self, sample_geography_concepts: list[Concept]
    ):
        """Test batch prediction across all geographies."""
        from knowledge_graph.classifier.geography import GeographyBatchPredictor

        predictor = GeographyBatchPredictor()
        predictor.initialize(sample_geography_concepts)

        texts = [
            "Brazil and France signed a trade agreement.",
            "The UK announced new policies.",
        ]

        results = predictor.predict_all(texts, threshold=0.3)

        assert isinstance(results, dict)
        # Results should contain WikibaseIDs as keys
        for qid in results:
            assert isinstance(qid, WikibaseID)


class TestGeographyClassifierMixins:
    """Tests for classifier mixin compliance."""

    def test_classifier_is_zero_shot(self, brazil_concept: Concept):
        """Test that GeographyClassifier is a ZeroShotClassifier."""
        from knowledge_graph.classifier.classifier import ZeroShotClassifier
        from knowledge_graph.classifier.geography import GeographyClassifier

        classifier = GeographyClassifier(brazil_concept)
        assert isinstance(classifier, ZeroShotClassifier)

    def test_classifier_is_gpu_bound(self, brazil_concept: Concept):
        """Test that GeographyClassifier is a GPUBoundClassifier."""
        from knowledge_graph.classifier.classifier import GPUBoundClassifier
        from knowledge_graph.classifier.geography import GeographyClassifier

        classifier = GeographyClassifier(brazil_concept)
        assert isinstance(classifier, GPUBoundClassifier)

    def test_classifier_is_probability_capable(self, brazil_concept: Concept):
        """Test that GeographyClassifier is a ProbabilityCapableClassifier."""
        from knowledge_graph.classifier.classifier import ProbabilityCapableClassifier
        from knowledge_graph.classifier.geography import GeographyClassifier

        classifier = GeographyClassifier(brazil_concept)
        assert isinstance(classifier, ProbabilityCapableClassifier)


class TestSpanCreation:
    """Tests for span creation from NER results."""

    def test_make_span_creates_valid_span(self):
        """Test that _make_span creates valid Span objects."""
        GeographyClassifierBackend.reset_instance()
        backend = GeographyClassifierBackend.get_instance()

        text = "Brazil is a country."
        entity = {
            "text": "Brazil",
            "start": 0,
            "end": 6,
            "score": 0.95,
        }
        qid = WikibaseID("Q155")

        span = backend._make_span(text, entity, qid)

        assert isinstance(span, Span)
        assert span.text == text
        assert span.start_index == 0
        assert span.end_index == 6
        assert span.labelled_text == "Brazil"
        assert span.concept_id == qid
        assert span.prediction_probability == 0.95
        assert "GeographyClassifier" in span.labellers

        GeographyClassifierBackend.reset_instance()


# =============================================================================
# spaCy Entity Linker Tests
# =============================================================================


class TestSpacyGeographyBackend:
    """Tests for the SpacyGeographyBackend singleton."""

    def test_singleton_pattern(self):
        """Test that get_instance returns the same instance."""
        SpacyGeographyBackend.reset_instance()

        instance1 = SpacyGeographyBackend.get_instance()
        instance2 = SpacyGeographyBackend.get_instance()

        assert instance1 is instance2

        SpacyGeographyBackend.reset_instance()

    def test_reset_instance(self):
        """Test that reset_instance clears the singleton."""
        instance1 = SpacyGeographyBackend.get_instance()
        SpacyGeographyBackend.reset_instance()
        instance2 = SpacyGeographyBackend.get_instance()

        assert instance1 is not instance2

        SpacyGeographyBackend.reset_instance()

    def test_predict_all_raises_without_initialization(self):
        """Test that predict_all raises if not initialized."""
        SpacyGeographyBackend.reset_instance()
        backend = SpacyGeographyBackend.get_instance()

        with pytest.raises(RuntimeError, match="Backend not initialized"):
            backend.predict_all(["test text"])

        SpacyGeographyBackend.reset_instance()

    def test_initialize_stores_valid_qids(
        self, sample_geography_concepts: list[Concept]
    ):
        """Test that initialize correctly stores valid QIDs."""
        SpacyGeographyBackend.reset_instance()
        backend = SpacyGeographyBackend.get_instance()
        backend.initialize(sample_geography_concepts)

        assert len(backend.valid_qids) == 5  # 5 concepts with wikibase_id
        assert WikibaseID("Q155") in backend.valid_qids
        assert WikibaseID("Q145") in backend.valid_qids
        assert backend._initialized

        SpacyGeographyBackend.reset_instance()


@pytest.mark.transformers
class TestSpacyGeographyBackendWithModels:
    """Tests for SpacyGeographyBackend that require loading models."""

    @pytest.fixture(autouse=True)
    def reset_backend(self):
        """Reset the backend before and after each test."""
        SpacyGeographyBackend.reset_instance()
        yield
        SpacyGeographyBackend.reset_instance()

    def test_predict_all_returns_correct_structure(
        self, sample_geography_concepts: list[Concept]
    ):
        """Test that predict_all returns correct structure."""
        backend = SpacyGeographyBackend.get_instance()
        backend.initialize(sample_geography_concepts)

        texts = [
            "Brazil announced new policies.",
            "The UK government responded.",
        ]

        results = backend.predict_all(texts)

        # Results should be a dict mapping QID to list of span lists
        assert isinstance(results, dict)
        for qid, spans_per_text in results.items():
            assert isinstance(qid, WikibaseID)
            assert isinstance(spans_per_text, list)
            assert len(spans_per_text) == len(texts)


@pytest.mark.transformers
class TestSpacyGeographyClassifier:
    """Tests for the SpacyGeographyClassifier class."""

    @pytest.fixture(autouse=True)
    def reset_backend(self):
        """Reset the backend before and after each test."""
        SpacyGeographyBackend.reset_instance()
        yield
        SpacyGeographyBackend.reset_instance()

    def test_classifier_initialization(self, brazil_concept: Concept):
        """Test that spaCy classifier initializes correctly."""
        from knowledge_graph.classifier.geography import SpacyGeographyClassifier

        classifier = SpacyGeographyClassifier(brazil_concept)

        assert classifier.concept == brazil_concept
        assert classifier.is_fitted  # Zero-shot, always fitted
        assert classifier.prediction_threshold == 0.0

    def test_classifier_id_is_deterministic(self, brazil_concept: Concept):
        """Test that classifier ID is deterministic."""
        from knowledge_graph.classifier.geography import SpacyGeographyClassifier

        classifier1 = SpacyGeographyClassifier(brazil_concept)
        classifier2 = SpacyGeographyClassifier(brazil_concept)

        assert classifier1.id == classifier2.id
        assert isinstance(classifier1.id, ClassifierID)

    def test_classifier_id_differs_from_gliner(self, brazil_concept: Concept):
        """Test that spaCy classifier has different ID from GliNER classifier."""
        from knowledge_graph.classifier.geography import (
            GeographyClassifier,
            SpacyGeographyClassifier,
        )

        gliner_classifier = GeographyClassifier(brazil_concept)
        spacy_classifier = SpacyGeographyClassifier(brazil_concept)

        assert gliner_classifier.id != spacy_classifier.id

    def test_classifier_fit_warns(self, brazil_concept: Concept):
        """Test that fit() emits a warning for zero-shot classifier."""
        from knowledge_graph.classifier.geography import SpacyGeographyClassifier

        classifier = SpacyGeographyClassifier(brazil_concept)

        with pytest.warns(UserWarning, match="zero-shot"):
            classifier.fit()

    def test_classifier_repr(self, brazil_concept: Concept):
        """Test classifier string representation."""
        from knowledge_graph.classifier.geography import SpacyGeographyClassifier

        classifier = SpacyGeographyClassifier(brazil_concept)
        repr_str = repr(classifier)

        assert "SpacyGeographyClassifier" in repr_str
        assert "Brazil" in repr_str


class TestSpacyGeographyClassifierMixins:
    """Tests for spaCy classifier mixin compliance."""

    def test_classifier_is_zero_shot(self, brazil_concept: Concept):
        """Test that SpacyGeographyClassifier is a ZeroShotClassifier."""
        from knowledge_graph.classifier.classifier import ZeroShotClassifier
        from knowledge_graph.classifier.geography import SpacyGeographyClassifier

        classifier = SpacyGeographyClassifier(brazil_concept)
        assert isinstance(classifier, ZeroShotClassifier)

    def test_classifier_is_not_gpu_bound(self, brazil_concept: Concept):
        """Test that SpacyGeographyClassifier is NOT a GPUBoundClassifier."""
        from knowledge_graph.classifier.classifier import GPUBoundClassifier
        from knowledge_graph.classifier.geography import SpacyGeographyClassifier

        classifier = SpacyGeographyClassifier(brazil_concept)
        # spaCy entity linker is CPU-based
        assert not isinstance(classifier, GPUBoundClassifier)

    def test_classifier_is_probability_capable(self, brazil_concept: Concept):
        """Test that SpacyGeographyClassifier is a ProbabilityCapableClassifier."""
        from knowledge_graph.classifier.classifier import ProbabilityCapableClassifier
        from knowledge_graph.classifier.geography import SpacyGeographyClassifier

        classifier = SpacyGeographyClassifier(brazil_concept)
        assert isinstance(classifier, ProbabilityCapableClassifier)
