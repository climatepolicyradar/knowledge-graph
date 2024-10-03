from sentence_transformers import SentenceTransformer

from src.classifier.classifier import Classifier
from src.concept import Concept
from src.span import Span


class EmbeddingClassifier(Classifier):
    """
    A classifier that uses an embedding model identify concepts in text.

    This classifier uses a SentenceTransformer model to create a vector representation
    of a concept. When .predict() is called, the classifier will encode the input text
    and compare it to the concept's embedding. If the similarity is above a given
    threshold, the classifier will return a Span object with the concept's Wikibase ID.
    """

    def __init__(
        self,
        concept: Concept,
        embedding_model: SentenceTransformer = SentenceTransformer(
            "BAAI/bge-small-en-v1.5"
        ),
    ):
        super().__init__(concept)
        self.concept = concept
        self.embedding_model = embedding_model

        # this is a VERY naive way of representing a concept as text. i'd like to
        # refine this and incorporate a bit more structure from the original concept at
        # some point
        self.concept_text = ", ".join(self.concept.all_labels)
        self.concept_embedding = self.embedding_model.encode(self.concept_text)

    def predict(self, text: str, threshold: float = 0.65) -> list[Span]:
        """
        Predict whether the supplied text contains an instance of the concept.

        :param str text: The text to predict on
        :param float threshold: The similarity threshold for a positive prediction
        :return list[Span]: If the similarity is above the threshold, a single span
        will be returned, covering the full text. Otherwise, an empty list will be
        returned.
        """
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
                )
            ]
        return spans
