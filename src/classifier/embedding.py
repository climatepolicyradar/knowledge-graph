from datetime import datetime
from typing import Optional

from src.classifier.classifier import Classifier, ZeroShotClassifier
from src.concept import Concept
from src.identifiers import Identifier
from src.span import Span


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
    ):
        super().__init__(concept)

        try:
            from sentence_transformers import SentenceTransformer

            self.embedding_model = SentenceTransformer(embedding_model_name)
        except ImportError:
            raise ImportError(
                f"The `sentence-transformers` library is required to run {self.name}s. "
                "Install it with 'poetry install --with transformers'"
            )

        self.threshold = threshold

        self.concept_text = self.concept.to_markdown()
        self.concept_embedding = self.embedding_model.encode(self.concept_text)

    def __repr__(self):
        """Return a string representation of the classifier."""
        return (
            f'{self.name}("{self.concept.preferred_label}", threshold={self.threshold})'
        )

    @property
    def id(self) -> Identifier:
        """Return a hash of the classifier."""
        return Identifier.generate(
            self.name,
            self.concept,
            self.embedding_model,
            self.threshold,
        )

    def __hash__(self) -> int:
        """Return a hash of the classifier."""
        return hash(self.id)

    def predict(self, text: str, threshold: Optional[float] = None) -> list[Span]:
        """
        Predict whether the supplied text contains an instance of the concept.

        :param str text: The text to predict on
        :param float threshold: The similarity threshold for a positive prediction
        :return list[Span]: If the similarity is above the threshold, a single span
        will be returned, covering the full text. Otherwise, an empty list will be
        returned.
        """
        threshold = threshold or self.threshold
        query_embedding = self.embedding_model.encode(text)
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

    def predict_batch(
        self,
        texts: list[str],
        threshold: Optional[float] = None,
        show_progress_bar: bool = False,
    ) -> list[list[Span]]:
        """
        Predict whether the supplied texts contain instances of the concept.

        :param list[str] texts: The texts to predict on
        :return list[list[Span]]: A list of spans in the texts for each text
        """
        threshold = threshold or self.threshold
        text_embeddings = self.embedding_model.encode(
            texts, show_progress_bar=show_progress_bar
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
