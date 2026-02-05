"""
Geography classifier using GliNER NER and text search entity linking.

This module provides a two-stage pipeline for geography classification:
1. NER: GliNER extracts geographic entity spans (type-based: "country", "city", etc.)
2. Entity Linking: Text search maps extracted spans to Wikidata IDs via label index

The architecture uses a singleton backend to share the NER model across all
geography classifiers, enabling efficient batch processing.
"""

from __future__ import annotations

import logging
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, ClassVar, Sequence

import numpy as np

from knowledge_graph.classifier.classifier import (
    Classifier,
    GPUBoundClassifier,
    ProbabilityCapableClassifier,
    ZeroShotClassifier,
)
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import ClassifierID, WikibaseID
from knowledge_graph.span import Span

if TYPE_CHECKING:
    from gliner import GLiNER
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


@dataclass
class LabelIndex:
    """
    Maps text labels to Wikidata IDs for entity linking.

    This index is built from geography concepts and supports both unambiguous
    lookups (label maps to single QID) and ambiguous lookups (label maps to
    multiple QIDs requiring context-based disambiguation).

    Attributes:
        unambiguous: Mapping from normalized label to single WikibaseID.
        ambiguous: Mapping from normalized label to list of candidate WikibaseIDs.
        descriptions: Mapping from WikibaseID to concept description for disambiguation.
        qid_to_concept: Mapping from WikibaseID to the full Concept object.
    """

    unambiguous: dict[str, WikibaseID] = field(default_factory=dict)
    ambiguous: dict[str, list[WikibaseID]] = field(default_factory=dict)
    descriptions: dict[WikibaseID, str] = field(default_factory=dict)
    qid_to_concept: dict[WikibaseID, Concept] = field(default_factory=dict)

    @classmethod
    def from_concepts(cls, concepts: Sequence[Concept]) -> LabelIndex:
        """
        Build a label index from geography concepts with Wikidata labels.

        Args:
            concepts: Sequence of Concept objects with wikibase_id and labels.

        Returns:
            A LabelIndex instance with unambiguous and ambiguous label mappings.
        """
        label_to_qids: dict[str, list[WikibaseID]] = defaultdict(list)
        descriptions: dict[WikibaseID, str] = {}
        qid_to_concept: dict[WikibaseID, Concept] = {}

        for concept in concepts:
            if concept.wikibase_id is None:
                logger.warning(
                    f"Skipping concept '{concept.preferred_label}' without wikibase_id"
                )
                continue

            qid = concept.wikibase_id
            qid_to_concept[qid] = concept

            # Store description for disambiguation
            description = concept.description or concept.preferred_label
            if concept.definition:
                description = f"{description}. {concept.definition}"
            descriptions[qid] = description

            # Index all labels
            all_labels = [concept.preferred_label] + concept.alternative_labels
            for label in all_labels:
                normalized = label.lower().strip()
                if normalized and qid not in label_to_qids[normalized]:
                    label_to_qids[normalized].append(qid)

        # Split into unambiguous and ambiguous
        unambiguous = {k: v[0] for k, v in label_to_qids.items() if len(v) == 1}
        ambiguous = {k: v for k, v in label_to_qids.items() if len(v) > 1}

        if ambiguous:
            logger.info(
                f"Built label index with {len(unambiguous)} unambiguous and "
                f"{len(ambiguous)} ambiguous labels"
            )
            for label, qids in ambiguous.items():
                logger.debug(f"  Ambiguous label '{label}' -> {qids}")

        return cls(
            unambiguous=unambiguous,
            ambiguous=ambiguous,
            descriptions=descriptions,
            qid_to_concept=qid_to_concept,
        )

    def lookup(self, label: str) -> WikibaseID | list[WikibaseID] | None:
        """
        Look up a label in the index.

        Args:
            label: The text label to look up.

        Returns:
            - Single WikibaseID if unambiguous
            - List of WikibaseIDs if ambiguous
            - None if not found
        """
        normalized = label.lower().strip()
        if normalized in self.unambiguous:
            return self.unambiguous[normalized]
        if normalized in self.ambiguous:
            return self.ambiguous[normalized]
        return None

    def is_ambiguous(self, label: str) -> bool:
        """Check if a label maps to multiple QIDs."""
        return label.lower().strip() in self.ambiguous

    def __len__(self) -> int:
        """Return total number of unique labels in the index."""
        return len(self.unambiguous) + len(self.ambiguous)


class GeographyClassifierBackend:
    """
    Shared backend for all geography classifiers.

    This singleton class manages the GliNER NER model and embedding model
    used for entity linking. It provides efficient batch prediction across
    all geography concepts with a single forward pass.

    Attributes:
        ENTITY_TYPES: GliNER entity types for geographic entities.
        ner_model: The GliNER model for named entity recognition.
        embedder: SentenceTransformer model for context-based disambiguation.
        label_index: The label index built from geography concepts.
    """

    _instance: ClassVar[GeographyClassifierBackend | None] = None

    # GliNER entity types for geographic entities
    ENTITY_TYPES: ClassVar[list[str]] = [
        "country",
        "nation",
        "city",
        "region",
        "geographic location",
        "state",
        "province",
        "territory",
    ]

    def __init__(
        self,
        gliner_model: str = "urchade/gliner_multi-v2.1",
        embedding_model: str = "all-MiniLM-L6-v2",
        device: str | None = None,
    ):
        """
        Initialize the backend with NER and embedding models.

        Args:
            gliner_model: HuggingFace model ID for GliNER.
            embedding_model: SentenceTransformer model name for disambiguation.
            device: Device to run models on ('cuda', 'cpu', or None for auto).
        """
        self.gliner_model_name = gliner_model
        self.embedding_model_name = embedding_model
        self._device = device

        self._ner_model: GLiNER | None = None
        self._embedder: SentenceTransformer | None = None
        self.label_index: LabelIndex | None = None
        self._description_embeddings: dict[WikibaseID, np.ndarray] = {}
        self._initialized = False

    @classmethod
    def get_instance(
        cls,
        gliner_model: str = "urchade/gliner_multi-v2.1",
        embedding_model: str = "all-MiniLM-L6-v2",
        device: str | None = None,
    ) -> GeographyClassifierBackend:
        """
        Get the singleton instance of the backend.

        Args:
            gliner_model: HuggingFace model ID for GliNER.
            embedding_model: SentenceTransformer model name for disambiguation.
            device: Device to run models on.

        Returns:
            The singleton GeographyClassifierBackend instance.
        """
        if cls._instance is None:
            cls._instance = cls(
                gliner_model=gliner_model,
                embedding_model=embedding_model,
                device=device,
            )
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance. Useful for testing."""
        cls._instance = None

    @property
    def ner_model(self) -> GLiNER:
        """Lazily load and return the GliNER model."""
        if self._ner_model is None:
            from gliner import GLiNER

            logger.info(f"Loading GliNER model: {self.gliner_model_name}")
            self._ner_model = GLiNER.from_pretrained(self.gliner_model_name)

            # Move to appropriate device
            device = self._get_device()
            self._ner_model.to(device)
            logger.info(f"GliNER model loaded on device: {device}")

        return self._ner_model

    @property
    def embedder(self) -> SentenceTransformer:
        """Lazily load and return the SentenceTransformer model."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            device = self._get_device()
            self._embedder = SentenceTransformer(
                self.embedding_model_name, device=device
            )
            logger.info(f"Embedding model loaded on device: {device}")

        return self._embedder

    def _get_device(self) -> str:
        """Determine the device to use for model inference."""
        if self._device is not None:
            return self._device

        import torch

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def initialize_index(self, concepts: Sequence[Concept]) -> None:
        """
        Build the label index from geography concepts.

        This method must be called before predictions can be made.
        It builds the label index and pre-computes embeddings for
        ambiguous entity descriptions.

        Args:
            concepts: Sequence of geography Concept objects.
        """
        logger.info(f"Initializing label index with {len(concepts)} concepts")
        self.label_index = LabelIndex.from_concepts(concepts)

        # Pre-compute embeddings for ambiguous entity descriptions
        if self.label_index.ambiguous:
            ambiguous_qids = set(
                qid for qids in self.label_index.ambiguous.values() for qid in qids
            )
            logger.info(
                f"Pre-computing embeddings for {len(ambiguous_qids)} ambiguous entities"
            )

            for qid in ambiguous_qids:
                desc = self.label_index.descriptions[qid]
                self._description_embeddings[qid] = self.embedder.encode(
                    desc, convert_to_numpy=True
                )

        self._initialized = True
        logger.info("Backend initialization complete")

    def predict_all(
        self,
        texts: Sequence[str],
        threshold: float = 0.5,
        context_window: int = 100,
    ) -> dict[WikibaseID, list[list[Span]]]:
        """
        Extract and link geographic entities from texts.

        This method performs NER to extract geographic entity spans,
        then links them to Wikidata IDs using the label index.

        Args:
            texts: Sequence of texts to process.
            threshold: NER confidence threshold (0-1).
            context_window: Characters of context for disambiguation.

        Returns:
            Dict mapping WikibaseID to list of spans per text.
            For each WikibaseID, returns a list of length len(texts),
            where each element is a list of Span objects for that text.
        """
        if not self._initialized or self.label_index is None:
            raise RuntimeError(
                "Backend not initialized. Call initialize_index() first."
            )

        # Initialize results structure
        results: dict[WikibaseID, list[list[Span]]] = defaultdict(
            lambda: [[] for _ in texts]
        )
        unknown_entities: list[tuple[int, str, dict]] = []

        # Step 1: NER - extract geographic entity spans
        logger.debug(f"Running NER on {len(texts)} texts with threshold {threshold}")
        texts_list = list(texts)

        # GliNER batch prediction
        ner_results = self.ner_model.batch_predict_entities(
            texts_list,
            self.ENTITY_TYPES,
            threshold=threshold,
        )

        # Step 2: Entity linking via text search + disambiguation
        for text_idx, (text, entities) in enumerate(zip(texts_list, ner_results)):
            for entity in entities:
                label = entity["text"].lower().strip()

                # Fast path: unambiguous label
                if label in self.label_index.unambiguous:
                    qid = self.label_index.unambiguous[label]
                    span = self._make_span(text, entity, qid)
                    results[qid][text_idx].append(span)

                # Slow path: ambiguous label needs context check
                elif label in self.label_index.ambiguous:
                    qid = self._disambiguate(
                        text, entity, label, context_window=context_window
                    )
                    if qid:
                        span = self._make_span(text, entity, qid)
                        results[qid][text_idx].append(span)

                # Unknown entity - not in our index
                else:
                    unknown_entities.append((text_idx, label, entity))

        # Log unknown entities for review
        if unknown_entities:
            unique_unknown = set(label for _, label, _ in unknown_entities)
            logger.info(
                f"Found {len(unknown_entities)} mentions of {len(unique_unknown)} "
                f"unknown geography entities: {list(unique_unknown)[:10]}..."
            )

        return dict(results)

    def _disambiguate(
        self,
        text: str,
        entity: dict,
        label: str,
        context_window: int = 100,
    ) -> WikibaseID | None:
        """
        Disambiguate an ambiguous label using context.

        Uses embedding similarity between the context around the entity
        and the descriptions of candidate entities.

        Args:
            text: The full text containing the entity.
            entity: The NER entity dict with 'start' and 'end' keys.
            label: The normalized label string.
            context_window: Characters of context on each side.

        Returns:
            The most likely WikibaseID, or None if disambiguation fails.
        """
        if self.label_index is None:
            return None

        candidate_qids = self.label_index.ambiguous.get(label, [])
        if not candidate_qids:
            return None

        # Extract context window around the entity
        start = entity["start"]
        end = entity["end"]
        context_start = max(0, start - context_window)
        context_end = min(len(text), end + context_window)
        context = text[context_start:context_end]

        # Embed context and compare to candidate descriptions
        context_emb = self.embedder.encode(context, convert_to_numpy=True)

        best_qid = None
        best_score = -1.0
        for qid in candidate_qids:
            if qid not in self._description_embeddings:
                continue
            desc_emb = self._description_embeddings[qid]

            # Cosine similarity
            norm_context = np.linalg.norm(context_emb)
            norm_desc = np.linalg.norm(desc_emb)
            if norm_context == 0 or norm_desc == 0:
                continue

            score = float(np.dot(context_emb, desc_emb) / (norm_context * norm_desc))
            if score > best_score:
                best_score = score
                best_qid = qid

        if best_qid:
            logger.debug(
                f"Disambiguated '{label}' to {best_qid} with score {best_score:.3f}"
            )

        return best_qid

    def _make_span(self, text: str, entity: dict, qid: WikibaseID) -> Span:
        """
        Convert NER entity to Span object.

        Args:
            text: The full text containing the entity.
            entity: The NER entity dict with 'start', 'end', 'text', 'score'.
            qid: The linked WikibaseID.

        Returns:
            A Span object representing the entity mention.
        """
        return Span(
            text=text,
            start_index=entity["start"],
            end_index=entity["end"],
            prediction_probability=entity.get("score"),
            concept_id=qid,
            labellers=["GeographyClassifier"],
            timestamps=[datetime.now()],
        )


class GeographyClassifier(
    Classifier, ZeroShotClassifier, GPUBoundClassifier, ProbabilityCapableClassifier
):
    """
    Geography classifier using shared GliNER + text search backend.

    This classifier uses a shared backend to perform NER and entity linking.
    Each instance is associated with a specific geography Concept and filters
    backend results to return only spans matching its WikibaseID.

    This is a zero-shot classifier that requires no training data - it works
    from Wikidata labels defined in the Concept.

    Attributes:
        model_name: The GliNER model name used by the backend.
        prediction_threshold: NER confidence threshold for predictions.
    """

    model_name: ClassVar[str] = "urchade/gliner_multi-v2.1"

    def __init__(
        self,
        concept: Concept,
        prediction_threshold: float = 0.5,
        context_window: int = 100,
    ):
        """
        Initialize the geography classifier.

        Args:
            concept: The geography Concept to classify.
            prediction_threshold: NER confidence threshold (0-1).
            context_window: Characters of context for disambiguation.
        """
        super().__init__(concept)
        self._backend = GeographyClassifierBackend.get_instance()
        self.prediction_threshold = prediction_threshold
        self.context_window = context_window
        self.is_fitted = True  # Zero-shot, always ready

    @property
    def id(self) -> ClassifierID:
        """Return a deterministic identifier for this classifier."""
        return ClassifierID.generate(
            self.name,
            self.concept.id,
            self.model_name,
            self.prediction_threshold,
        )

    def _predict(self, text: str, threshold: float | None = None) -> list[Span]:
        """
        Predict geography mentions in a single text.

        Args:
            text: The text to analyze.
            threshold: Optional threshold override.

        Returns:
            List of Span objects for this classifier's concept.
        """
        return self._predict_batch([text], threshold=threshold)[0]

    def _predict_batch(
        self, texts: Sequence[str], threshold: float | None = None
    ) -> list[list[Span]]:
        """
        Predict geography mentions in multiple texts.

        Args:
            texts: Sequence of texts to analyze.
            threshold: Optional threshold override.

        Returns:
            List of span lists, one per input text.
        """
        effective_threshold = threshold or self.prediction_threshold

        # Get all results from the backend
        all_results = self._backend.predict_all(
            texts,
            threshold=effective_threshold,
            context_window=self.context_window,
        )

        # Return only spans for this classifier's concept
        if self.concept.wikibase_id is None:
            logger.warning(
                f"Concept '{self.concept.preferred_label}' has no wikibase_id"
            )
            return [[] for _ in texts]

        return all_results.get(self.concept.wikibase_id, [[] for _ in texts])

    def fit(self, **kwargs) -> GeographyClassifier:
        """
        No-op for zero-shot classifier.

        Returns:
            Self, unchanged.
        """
        warnings.warn(
            "GeographyClassifier is zero-shot based on Wikidata labels. "
            "No training is performed.",
            UserWarning,
        )
        return self


class GeographyBatchPredictor:
    """
    Predict all geographies in a single pass.

    This class provides efficient batch prediction across all geography
    concepts. Instead of running the NER model once per concept, it runs
    a single forward pass and returns all geography matches.

    Example:
        >>> predictor = GeographyBatchPredictor()
        >>> results = predictor.predict_all(texts)
        >>> brazil_spans = results.get(WikibaseID("Q155"), [[] for _ in texts])
    """

    def __init__(
        self,
        gliner_model: str = "urchade/gliner_multi-v2.1",
        embedding_model: str = "all-MiniLM-L6-v2",
        device: str | None = None,
    ):
        """
        Initialize the batch predictor.

        Args:
            gliner_model: HuggingFace model ID for GliNER.
            embedding_model: SentenceTransformer model name for disambiguation.
            device: Device to run models on.
        """
        self._backend = GeographyClassifierBackend.get_instance(
            gliner_model=gliner_model,
            embedding_model=embedding_model,
            device=device,
        )

    def initialize(self, concepts: Sequence[Concept]) -> None:
        """
        Initialize the backend with geography concepts.

        This must be called before predict_all().

        Args:
            concepts: Sequence of geography Concept objects.
        """
        self._backend.initialize_index(concepts)

    def predict_all(
        self,
        texts: Sequence[str],
        threshold: float = 0.5,
        context_window: int = 100,
    ) -> dict[WikibaseID, list[list[Span]]]:
        """
        Predict all geography mentions in texts.

        Args:
            texts: Sequence of texts to process.
            threshold: NER confidence threshold (0-1).
            context_window: Characters of context for disambiguation.

        Returns:
            Dict mapping WikibaseID to list of spans per text.
        """
        return self._backend.predict_all(
            texts, threshold=threshold, context_window=context_window
        )


def initialize_geography_classifiers(concepts: Sequence[Concept]) -> None:
    """
    Initialize the shared backend with all geography concepts.

    This function should be called at startup or when geography concepts
    are loaded. It builds the label index used by all GeographyClassifier
    instances.

    Args:
        concepts: Sequence of geography Concept objects.

    Example:
        >>> from knowledge_graph.classifier.geography import (
        ...     initialize_geography_classifiers,
        ...     GeographyClassifier,
        ... )
        >>> initialize_geography_classifiers(geography_concepts)
        >>> brazil_classifier = GeographyClassifier(brazil_concept)
        >>> uk_classifier = GeographyClassifier(uk_concept)
    """
    backend = GeographyClassifierBackend.get_instance()
    backend.initialize_index(concepts)


# =============================================================================
# Alternative Implementation: spaCy Entity Linker
# =============================================================================


class SpacyGeographyBackend:
    """
    Shared backend using spaCy with entity linker for Wikidata linking.

    This backend uses spaCy's NER pipeline combined with the spacy-entity-linker
    extension to directly link entities to Wikidata. Unlike the GliNER approach,
    this uses spaCy's built-in NER and relies on the entity linker's knowledge
    base for disambiguation.

    Attributes:
        nlp: The spaCy Language pipeline with entity linker.
        valid_qids: Set of WikibaseIDs we're interested in (our 200 geographies).
    """

    _instance: ClassVar[SpacyGeographyBackend | None] = None

    def __init__(
        self,
        spacy_model: str = "en_core_web_md",
    ):
        """
        Initialize the spaCy backend.

        Args:
            spacy_model: spaCy model to use (must support NER).
        """
        self.spacy_model_name = spacy_model
        self._nlp = None
        self.valid_qids: set[WikibaseID] = set()
        self.qid_to_concept: dict[WikibaseID, Concept] = {}
        self._initialized = False

    @classmethod
    def get_instance(
        cls,
        spacy_model: str = "en_core_web_md",
    ) -> SpacyGeographyBackend:
        """
        Get the singleton instance of the backend.

        Args:
            spacy_model: spaCy model to use.

        Returns:
            The singleton SpacyGeographyBackend instance.
        """
        if cls._instance is None:
            cls._instance = cls(spacy_model=spacy_model)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance. Useful for testing."""
        cls._instance = None

    @property
    def nlp(self):
        """Lazily load and return the spaCy pipeline with entity linker."""
        if self._nlp is None:
            import spacy

            logger.info(f"Loading spaCy model: {self.spacy_model_name}")
            self._nlp = spacy.load(self.spacy_model_name)

            # Add entity linker if not present
            if "entityLinker" not in self._nlp.pipe_names:
                logger.info("Adding entityLinker to spaCy pipeline")
                self._nlp.add_pipe("entityLinker", last=True)

            logger.info("spaCy pipeline loaded with entity linker")

        return self._nlp

    def initialize(self, concepts: Sequence[Concept]) -> None:
        """
        Initialize with the geography concepts we care about.

        Args:
            concepts: Sequence of geography Concept objects.
        """
        logger.info(f"Initializing spaCy backend with {len(concepts)} concepts")

        self.valid_qids = set()
        self.qid_to_concept = {}

        for concept in concepts:
            if concept.wikibase_id is not None:
                self.valid_qids.add(concept.wikibase_id)
                self.qid_to_concept[concept.wikibase_id] = concept

        self._initialized = True
        logger.info(f"Initialized with {len(self.valid_qids)} valid QIDs")

    def predict_all(
        self,
        texts: Sequence[str],
        threshold: float = 0.0,
    ) -> dict[WikibaseID, list[list[Span]]]:
        """
        Extract and link geographic entities using spaCy entity linker.

        Args:
            texts: Sequence of texts to process.
            threshold: Entity linking confidence threshold (0-1).

        Returns:
            Dict mapping WikibaseID to list of spans per text.
        """
        if not self._initialized:
            raise RuntimeError("Backend not initialized. Call initialize() first.")

        # Initialize results structure
        results: dict[WikibaseID, list[list[Span]]] = defaultdict(
            lambda: [[] for _ in texts]
        )
        unknown_entities: list[tuple[int, str]] = []

        # Process texts through spaCy pipeline
        texts_list = list(texts)
        docs = list(self.nlp.pipe(texts_list))

        for text_idx, doc in enumerate(docs):
            # Get linked entities from spacy-entity-linker
            if not hasattr(doc._, "linkedEntities"):
                continue

            for linked_entity in doc._.linkedEntities:
                # Get the Wikidata ID
                wikidata_id = linked_entity.get_id()
                if not wikidata_id:
                    continue

                # Normalize to WikibaseID format
                try:
                    qid = WikibaseID(wikidata_id)
                except ValueError:
                    continue

                # Check if this is one of our valid geographies
                if qid not in self.valid_qids:
                    unknown_entities.append((text_idx, wikidata_id))
                    continue

                # Check confidence threshold
                score = linked_entity.get_score()
                if score < threshold:
                    continue

                # Get span information
                span_obj = linked_entity.get_span()
                entity_span = self._make_span(
                    text=texts_list[text_idx],
                    start=span_obj.start_char,
                    end=span_obj.end_char,
                    score=score,
                    qid=qid,
                )
                results[qid][text_idx].append(entity_span)

        if unknown_entities:
            unique_unknown = set(qid for _, qid in unknown_entities)
            logger.debug(
                f"Found {len(unknown_entities)} entities not in our geography set"
            )

        return dict(results)

    def _make_span(
        self,
        text: str,
        start: int,
        end: int,
        score: float,
        qid: WikibaseID,
    ) -> Span:
        """
        Create a Span from spaCy entity information.

        Args:
            text: The full text.
            start: Start character index.
            end: End character index.
            score: Entity linking confidence score.
            qid: The linked WikibaseID.

        Returns:
            A Span object.
        """
        return Span(
            text=text,
            start_index=start,
            end_index=end,
            prediction_probability=score,
            concept_id=qid,
            labellers=["SpacyGeographyClassifier"],
            timestamps=[datetime.now()],
        )


class SpacyGeographyClassifier(
    Classifier, ZeroShotClassifier, ProbabilityCapableClassifier
):
    """
    Geography classifier using spaCy NER with Wikidata entity linking.

    This classifier uses spaCy's NER pipeline combined with spacy-entity-linker
    to directly link geographic entities to Wikidata IDs. Unlike the GliNER
    approach, this relies on spaCy's pre-trained NER and the entity linker's
    built-in knowledge base.

    Pros:
    - Direct Wikidata linking without custom label index
    - Leverages spaCy's mature NER pipeline
    - Entity linker handles disambiguation internally

    Cons:
    - May not recognize all 200 geographies (depends on spaCy NER)
    - Less control over entity types recognized
    - CPU-based (no GPU acceleration for entity linker)

    Attributes:
        model_name: The spaCy model name.
        prediction_threshold: Entity linking confidence threshold.
    """

    model_name: ClassVar[str] = "en_core_web_md"

    def __init__(
        self,
        concept: Concept,
        prediction_threshold: float = 0.0,
    ):
        """
        Initialize the spaCy geography classifier.

        Args:
            concept: The geography Concept to classify.
            prediction_threshold: Entity linking confidence threshold (0-1).
        """
        super().__init__(concept)
        self._backend = SpacyGeographyBackend.get_instance()
        self.prediction_threshold = prediction_threshold
        self.is_fitted = True  # Zero-shot, always ready

    @property
    def id(self) -> ClassifierID:
        """Return a deterministic identifier for this classifier."""
        return ClassifierID.generate(
            self.name,
            self.concept.id,
            self.model_name,
            self.prediction_threshold,
        )

    def _predict(self, text: str, threshold: float | None = None) -> list[Span]:
        """
        Predict geography mentions in a single text.

        Args:
            text: The text to analyze.
            threshold: Optional threshold override.

        Returns:
            List of Span objects for this classifier's concept.
        """
        return self._predict_batch([text], threshold=threshold)[0]

    def _predict_batch(
        self, texts: Sequence[str], threshold: float | None = None
    ) -> list[list[Span]]:
        """
        Predict geography mentions in multiple texts.

        Args:
            texts: Sequence of texts to analyze.
            threshold: Optional threshold override.

        Returns:
            List of span lists, one per input text.
        """
        effective_threshold = threshold or self.prediction_threshold

        # Get all results from the backend
        all_results = self._backend.predict_all(
            texts,
            threshold=effective_threshold,
        )

        # Return only spans for this classifier's concept
        if self.concept.wikibase_id is None:
            logger.warning(
                f"Concept '{self.concept.preferred_label}' has no wikibase_id"
            )
            return [[] for _ in texts]

        return all_results.get(self.concept.wikibase_id, [[] for _ in texts])

    def fit(self, **kwargs) -> SpacyGeographyClassifier:
        """
        No-op for zero-shot classifier.

        Returns:
            Self, unchanged.
        """
        warnings.warn(
            "SpacyGeographyClassifier is zero-shot using Wikidata entity linking. "
            "No training is performed.",
            UserWarning,
        )
        return self


class SpacyGeographyBatchPredictor:
    """
    Predict all geographies using spaCy entity linker in a single pass.

    This class provides batch prediction across all geography concepts
    using spaCy's entity linking to Wikidata.

    Example:
        >>> predictor = SpacyGeographyBatchPredictor()
        >>> predictor.initialize(geography_concepts)
        >>> results = predictor.predict_all(texts)
        >>> brazil_spans = results.get(WikibaseID("Q155"), [[] for _ in texts])
    """

    def __init__(
        self,
        spacy_model: str = "en_core_web_md",
    ):
        """
        Initialize the batch predictor.

        Args:
            spacy_model: spaCy model to use.
        """
        self._backend = SpacyGeographyBackend.get_instance(
            spacy_model=spacy_model,
        )

    def initialize(self, concepts: Sequence[Concept]) -> None:
        """
        Initialize the backend with geography concepts.

        This must be called before predict_all().

        Args:
            concepts: Sequence of geography Concept objects.
        """
        self._backend.initialize(concepts)

    def predict_all(
        self,
        texts: Sequence[str],
        threshold: float = 0.0,
    ) -> dict[WikibaseID, list[list[Span]]]:
        """
        Predict all geography mentions in texts.

        Args:
            texts: Sequence of texts to process.
            threshold: Entity linking confidence threshold (0-1).

        Returns:
            Dict mapping WikibaseID to list of spans per text.
        """
        return self._backend.predict_all(texts, threshold=threshold)


def initialize_spacy_geography_classifiers(concepts: Sequence[Concept]) -> None:
    """
    Initialize the spaCy backend with all geography concepts.

    This function should be called at startup when using SpacyGeographyClassifier.

    Args:
        concepts: Sequence of geography Concept objects.

    Example:
        >>> from knowledge_graph.classifier.geography import (
        ...     initialize_spacy_geography_classifiers,
        ...     SpacyGeographyClassifier,
        ... )
        >>> initialize_spacy_geography_classifiers(geography_concepts)
        >>> brazil_classifier = SpacyGeographyClassifier(brazil_concept)
    """
    backend = SpacyGeographyBackend.get_instance()
    backend.initialize(concepts)
