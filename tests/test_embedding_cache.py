import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from knowledge_graph.classifier.embedding import EmbeddingClassifier, _EmbeddingCache
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import WikibaseID


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def simple_concept():
    """Create a simple concept for testing."""
    return Concept(
        wikibase_id=WikibaseID("Q123"),
        preferred_label="Climate Finance",
        description="$$$$$$",
    )


def test_cache_initialization(temp_cache_dir):
    """Test that cache is initialized correctly."""
    cache = _EmbeddingCache(temp_cache_dir)
    assert cache.db_path.exists()
    assert cache.hits == 0
    assert cache.misses == 0


def test_cache_set_and_get(temp_cache_dir):
    """Test storing and retrieving an embedding."""
    cache = _EmbeddingCache(temp_cache_dir)

    embedding = np.random.rand(768).astype(np.float32)
    cache_key = "test_key_123"

    cache.set(cache_key, embedding)

    retrieved = cache.get(cache_key)

    assert retrieved is not None
    assert np.allclose(retrieved, embedding, atol=1e-6)
    assert cache.hits == 1
    assert cache.misses == 0


def test_cache_miss(temp_cache_dir):
    """Test cache miss behavior."""
    cache = _EmbeddingCache(temp_cache_dir)

    result = cache.get("nonexistent_key")

    assert result is None
    assert cache.misses == 1
    assert cache.hits == 0


def test_cache_batch_operations(temp_cache_dir):
    """Test batch set and get operations."""
    cache = _EmbeddingCache(temp_cache_dir)

    embeddings = {
        "key1": np.random.rand(768).astype(np.float32),
        "key2": np.random.rand(768).astype(np.float32),
        "key3": np.random.rand(768).astype(np.float32),
    }

    cache.set_batch(embeddings)

    keys = list(embeddings.keys()) + ["nonexistent"]
    results = cache.get_batch(keys)

    assert len(results) == 3  # Only existing keys
    assert "nonexistent" not in results
    assert cache.hits == 3
    assert cache.misses == 1

    for key in embeddings:
        assert np.allclose(results[key], embeddings[key], atol=1e-6)


def test_cache_stats(temp_cache_dir):
    """Test cache statistics."""
    cache = _EmbeddingCache(temp_cache_dir)

    embedding = np.random.rand(768).astype(np.float32)
    cache.set("key1", embedding)

    cache.get("key1")  # Hit
    cache.get("key1")  # Hit
    cache.get("key2")  # Miss

    stats = cache.get_stats()

    assert stats["hits"] == 2
    assert stats["misses"] == 1
    assert stats["total_requests"] == 3
    assert stats["hit_rate"] == pytest.approx(2 / 3)


@pytest.mark.transformers
def test_classifier_cache_integration(temp_cache_dir, simple_concept, monkeypatch):
    """Test that classifier uses cache correctly."""
    # Mock the config to use temp cache dir
    import knowledge_graph.config as config

    monkeypatch.setattr(config, "embedding_cache_dir", temp_cache_dir)

    classifier = EmbeddingClassifier(
        simple_concept,
        embedding_model_name="ibm-granite/granite-embedding-107m-multilingual",
        use_cache=True,
    )
    assert classifier._cache is not None  # for type checker
    text = "This is a test about climate finance"

    # First prediction should be a cache miss
    classifier.predict(text)
    stats1 = classifier._cache.get_stats()
    assert stats1["misses"] == 1
    assert stats1["hits"] == 0

    # Second prediction should be a cache hit
    classifier.predict(text)
    stats2 = classifier._cache.get_stats()
    assert stats2["misses"] == 1
    assert stats2["hits"] == 1


@pytest.mark.transformers
def test_classifier_batch_cache(temp_cache_dir, simple_concept, monkeypatch):
    """Test that batch predictions use cache efficiently."""
    import knowledge_graph.config as config

    monkeypatch.setattr(config, "embedding_cache_dir", temp_cache_dir)

    classifier = EmbeddingClassifier(
        simple_concept,
        embedding_model_name="ibm-granite/granite-embedding-107m-multilingual",
        use_cache=True,
    )

    texts = [
        "Climate finance is important",
        "This is about adaptation",
        "Climate finance is important",  # Duplicate
    ]

    # First batch should cache all unique texts
    classifier.predict(texts)
    stats1 = classifier._cache.get_stats()
    assert stats1["misses"] == 2  # Only 2 unique texts
    assert stats1["hits"] == 1  # One duplicate

    # Second batch should be all hits
    classifier.predict(texts)
    stats2 = classifier._cache.get_stats()
    assert stats2["misses"] == 2  # No new misses
    assert stats2["hits"] == 4  # 3 more hits


@pytest.mark.transformers
def test_classifier_cache_disabled(temp_cache_dir, simple_concept, monkeypatch):
    """Test that cache can be disabled."""
    import knowledge_graph.config as config

    monkeypatch.setattr(config, "embedding_cache_dir", temp_cache_dir)

    classifier = EmbeddingClassifier(
        simple_concept,
        embedding_model_name="ibm-granite/granite-embedding-107m-multilingual",
        use_cache=False,
    )

    text = "This is a test"

    # Predict twice
    classifier.predict(text)
    classifier.predict(text)

    # Cache should be None when disabled
    assert classifier._cache is None


@pytest.mark.transformers
def test_cache_key_generation(simple_concept, temp_cache_dir, monkeypatch):
    """Test that cache keys are generated consistently."""
    import knowledge_graph.config as config

    monkeypatch.setattr(config, "embedding_cache_dir", temp_cache_dir)

    with patch("sentence_transformers.SentenceTransformer"):
        classifier1 = EmbeddingClassifier(
            simple_concept,
            embedding_model_name="test-model",
            query_prefix="Query: ",
            document_prefix="Doc: ",
        )
        classifier2 = EmbeddingClassifier(
            simple_concept,
            embedding_model_name="test-model",
            query_prefix="Query: ",
            document_prefix="Doc: ",
        )

    text = "Test text"
    key1 = classifier1._generate_cache_key(text)
    key2 = classifier2._generate_cache_key(text)

    # Same params should generate same key
    assert key1 == key2
    assert len(key1) == 8  # Identifier length

    # Different text should generate different key
    key3 = classifier1._generate_cache_key("Different text")
    assert key1 != key3
