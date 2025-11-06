import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from knowledge_graph import config
from knowledge_graph.classifier.classifier import Classifier, ZeroShotClassifier
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import ClassifierID, Identifier
from knowledge_graph.span import Span

logger = logging.getLogger(__name__)


class _EmbeddingCache:
    """
    An SQLite cache for storing text embeddings.

    This cache stores embeddings as binary blobs indexed by a cache key. The cache key
    is generated from the embedding model name, query and document prefixes prefixes,
    and input text.
    """

    def __init__(self, cache_dir: Path):
        """
        Initialize the embedding cache.

        :param cache_dir: Directory to store the cache database
        """
        self.cache_dir = cache_dir
        self.hits = 0
        self.misses = 0

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "embeddings.db"

        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database with the required schema."""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    cache_key TEXT PRIMARY KEY,
                    embedding BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.commit()

    def get(self, cache_key: str) -> Optional[np.ndarray]:
        """
        Retrieve an embedding from the cache.

        :param cache_key: The cache key to lookup
        :return: The cached embedding as a numpy array, or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT embedding FROM embeddings WHERE cache_key = ?",
                (cache_key,),
            )
            row = cursor.fetchone()

        if row is None:
            self.misses += 1
            return None

        self.hits += 1

        return np.frombuffer(row[0], dtype=np.float32)

    def get_batch(self, cache_keys: list[str]) -> dict[str, np.ndarray]:
        """
        Retrieve multiple embeddings from the cache in a single query.

        :param cache_keys: List of cache keys to lookup
        :return: Dictionary mapping cache keys to their embeddings (only for hits)
        """
        if not cache_keys:
            return {}

        placeholders = ",".join("?" * len(cache_keys))
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                f"SELECT cache_key, embedding FROM embeddings WHERE cache_key IN ({placeholders})",
                cache_keys,
            )
            rows = cursor.fetchall()

        results = {}
        for cache_key, embedding_bytes in rows:
            results[cache_key] = np.frombuffer(embedding_bytes, dtype=np.float32)

        unique_keys = len(set(cache_keys))

        # misses = unique keys not found in cache
        cache_hits = len(results)
        cache_misses = unique_keys - cache_hits
        self.misses += cache_misses

        # hits = keys found in cache + duplicate keys in the request. although duplicate
        # keys aren't found in the SQlite database, it still makes sense to count them
        # as hits as we don't need to recompute the embeddings
        self.hits += len(cache_keys) - cache_misses

        return results

    def set(self, cache_key: str, embedding: np.ndarray):
        """
        Store an embedding in the cache.

        :param cache_key: The cache key to store under
        :param embedding: The embedding to store
        """

        embedding_bytes = embedding.astype(np.float32).tobytes()

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO embeddings (cache_key, embedding) VALUES (?, ?)",
                (cache_key, embedding_bytes),
            )
            conn.commit()

    def set_batch(self, items: dict[str, np.ndarray]):
        """
        Store multiple embeddings in the cache in a single transaction.

        :param items: Dictionary mapping cache keys to embeddings
        """
        if not items:
            return

        data = [
            (cache_key, embedding.astype(np.float32).tobytes())
            for cache_key, embedding in items.items()
        ]

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings (cache_key, embedding) VALUES (?, ?)",
                data,
            )
            conn.commit()

    def get_stats(self) -> dict:
        """
        Get cache statistics.

        :return: Dictionary with cache hits, misses, and hit rate
        """
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "total_requests": total,
            "hit_rate": hit_rate,
        }


class EmbeddingClassifier(Classifier, ZeroShotClassifier):
    """
    A classifier that uses an embedding model to identify concepts in text.

    This classifier uses a SentenceTransformer model to create a vector representation
    of a concept. When .predict() is called, the classifier will encode the input text
    and compare it to the concept's embedding. If the similarity is above a given
    threshold, the classifier will return a Span object with the concept's Wikibase ID.
    """

    def __init__(
        self,
        concept: Concept,
        embedding_model_name: str = "ibm-granite/granite-embedding-107m-multilingual",
        threshold: float = 0.65,
        document_prefix: str = "",
        query_prefix: str = "",
        device: str | None = None,
        use_cache: bool = True,
    ):
        super().__init__(concept)
        try:
            from sentence_transformers import (
                SentenceTransformer,  # type: ignore[import-untyped]
            )

            if (
                embedding_model_name.startswith("Snowflake/")
                or embedding_model_name.startswith("nomic-ai/")
                or embedding_model_name.startswith("Alibaba-NLP/")
            ):
                logger.warning(
                    "EmbeddingClassifier has been set to trust remote code. Don't use this in production – create a fork of the repo, or pin a commit to load from."
                )
                _trust_remote_code = True
            else:
                _trust_remote_code = False

            self.embedding_model = SentenceTransformer(
                embedding_model_name,
                device=device,
                trust_remote_code=_trust_remote_code,
            )
        except ImportError:
            raise ImportError(
                f"The `sentence-transformers` library is required to run {self.name}s. "
                "Install it with 'uv install --extra transformers'"
            )

        self.embedding_model_name = embedding_model_name
        self.threshold = threshold
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix

        self._cache = _EmbeddingCache(config.embedding_cache_dir) if use_cache else None

        self.concept_text = self.concept.to_markdown(
            include_alternative_labels=True,
            include_definition=True,
            include_description=True,
            include_concept_neighbourhood=False,
            include_example_passages=False,
            use_markdown_headers=True,
        )
        concept_text_with_prefix = f"{self.document_prefix}{self.concept_text}"
        self.concept_embedding = self.embedding_model.encode(concept_text_with_prefix)

        # Verify that embeddings are normalized (L2 norm ≈ 1.0)
        # This ensures that dot product equals cosine similarity
        embedding_norm = np.linalg.norm(self.concept_embedding)
        if not np.isclose(embedding_norm, 1.0, atol=0.01):
            logger.warning(
                "Embedding from %s is not L2-normalized (norm=%.4f). "
                "Dot product may not equal cosine similarity. "
                "Consider normalizing embeddings explicitly.",
                embedding_model_name,
                embedding_norm,
            )

    def __repr__(self):
        """Return a string representation of the classifier."""
        return (
            f'{self.name}("{self.concept.preferred_label}", threshold={self.threshold})'
        )

    @property
    def id(self) -> ClassifierID:
        """Return a deterministic, human-readable identifier for the classifier."""
        return ClassifierID.generate(
            self.name,
            self.concept.id,
            self.embedding_model,
            self.threshold,
            self.document_prefix,
            self.query_prefix,
        )

    def __hash__(self) -> int:
        """Return a hash of the classifier."""
        return hash(self.id)

    def _generate_cache_key(self, text: str) -> str:
        """
        Generate a deterministic cache key for the given text.

        :param text: The input text to generate a key for
        :return: An 8-character identifier
        """
        return str(
            Identifier.generate(
                self.embedding_model_name,
                self.query_prefix,
                self.document_prefix,
                text,
            )
        )

    def _encode(self, texts: str | list[str], show_progress_bar: bool) -> np.ndarray:
        return self.embedding_model.encode(
            texts,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True,
        )

    def _predict(self, text: str, threshold: Optional[float] = None) -> list[Span]:
        """
        Predict whether the supplied text contains an instance of the concept.

        :param str text: The text to predict on
        :param float threshold: The similarity threshold for a positive prediction
        :return list[Span]: If the similarity is above the threshold, a single span
        will be returned, covering the full text. Otherwise, an empty list will be
        returned.
        """
        threshold = threshold or self.threshold

        query_embedding = None
        cache_key = None

        # try to get embedding from cache
        if self._cache is not None:
            cache_key = self._generate_cache_key(text)
            query_embedding = self._cache.get(cache_key)

        # if not in cache, compute and store it
        if query_embedding is None:
            text_with_prefix = f"{self.query_prefix}{text}"
            query_embedding = self._encode(text_with_prefix, show_progress_bar=False)
            if self._cache is not None and cache_key is not None:
                self._cache.set(cache_key, query_embedding)

        similarity = self.concept_embedding @ query_embedding.T
        spans = []
        if similarity > threshold:
            spans = [
                Span(
                    text=text,
                    concept_id=self.concept.wikibase_id,
                    start_index=0,
                    end_index=len(text),
                    labellers=[str(self)],
                    timestamps=[datetime.now()],
                )
            ]
        return spans

    def _predict_batch(
        self,
        texts: Sequence[str],
        threshold: Optional[float] = None,
        show_progress_bar: bool = False,
    ) -> list[list[Span]]:
        """
        Predict whether the supplied texts contain instances of the concept.

        :param list[str] texts: The texts to predict on
        :return list[list[Span]]: A list of spans in the texts for each text
        """
        threshold = threshold or self.threshold

        # get embeddings for all text
        if self._cache is None:
            # if there's no cache, calculate all embeddings
            texts_with_prefix = [f"{self.query_prefix}{text}" for text in texts]
            text_embeddings = self._encode(
                texts_with_prefix, show_progress_bar=show_progress_bar
            )
            # idxs are converted to string here so type checker always deals with
            # dict[str, Any]
            embeddings_dict = {
                str(idx): embedding for (idx, embedding) in enumerate(text_embeddings)
            }
            keys = [str(idx) for idx in range(len(texts))]
        else:
            # if there's a cache, go and find all the embedding that exist there and
            # only calculate the ones that don't (the 'misses')
            cache_keys = [self._generate_cache_key(text) for text in texts]
            embeddings_dict = self._cache.get_batch(cache_keys)

            if cache_misses := {
                key: text
                for key, text in zip(cache_keys, texts)
                if key not in embeddings_dict
            }:
                texts_with_prefix = [
                    f"{self.query_prefix}{text}" for text in cache_misses.values()
                ]
                new_embeddings = self._encode(
                    texts_with_prefix,
                    show_progress_bar=show_progress_bar,
                )
                new_items = {k: v for k, v in zip(cache_misses.keys(), new_embeddings)}
                self._cache.set_batch(new_items)
                embeddings_dict.update(new_items)

            keys = cache_keys

        spans_per_text = []
        for text, key in zip(texts, keys):
            similarity = self.concept_embedding @ embeddings_dict[key].T
            spans = []
            if similarity > threshold:
                spans = [
                    Span(
                        text=text,
                        concept_id=self.concept.wikibase_id,
                        start_index=0,
                        end_index=len(text),
                        labellers=[str(self)],
                        timestamps=[datetime.now()],
                    )
                ]
            spans_per_text.append(spans)

        return spans_per_text

    def get_cache_stats(self) -> dict:
        """
        Get embedding cache statistics.

        :return: Dictionary with cache hits, misses, total requests, and hit rate
        """
        if self._cache is None:
            return {"hits": 0, "misses": 0, "total_requests": 0, "hit_rate": 0.0}

        stats = self._cache.get_stats()
        if stats["total_requests"] > 0:
            logger.info(
                "Embedding cache stats: %d hits, %d misses, %.1f%% hit rate",
                stats["hits"],
                stats["misses"],
                stats["hit_rate"] * 100,
            )
        return stats
