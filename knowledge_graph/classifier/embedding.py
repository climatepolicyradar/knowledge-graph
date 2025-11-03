import logging
from datetime import datetime
from typing import Optional, Sequence

import numpy as np

from knowledge_graph.classifier.classifier import Classifier, ZeroShotClassifier
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import ClassifierID
from knowledge_graph.span import Span

logger = logging.getLogger(__name__)


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
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        threshold: float = 0.65,
        document_prefix: str = "",
        query_prefix: str = "",
        device: str | None = None,
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

        self.threshold = threshold
        self.document_prefix = document_prefix
        self.query_prefix = query_prefix

        self.concept_text = self.concept.to_markdown()
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

        text_with_prefix = f"{self.query_prefix}{text}"
        query_embedding = self.embedding_model.encode(text_with_prefix)
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

        texts_with_prefix = [f"{self.query_prefix}{text}" for text in texts]
        text_embeddings = self.embedding_model.encode(
            texts_with_prefix, show_progress_bar=show_progress_bar
        )
        spans_per_text = []

        for text, text_embedding in zip(texts, text_embeddings):
            similarity = self.concept_embedding @ text_embedding.T
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
