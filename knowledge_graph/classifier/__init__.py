from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from knowledge_graph.classifier.classifier import (
    Classifier,
    GPUBoundClassifier,
)
from knowledge_graph.classifier.keyword import KeywordClassifier
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import ClassifierID, WikibaseID


class ModelPath(BaseModel):
    """Represents the expected path to a model artifact locally or in S3."""

    wikibase_id: WikibaseID
    classifier_id: ClassifierID

    def __str__(self) -> str:
        """
        Return the path to the model artifact.

        e.g. 'Q123/v4prnc54'
        """
        return f"{self.wikibase_id}/{self.classifier_id}"

    def __fspath__(self) -> str:
        """Return the filesystem path representation for use with pathlib."""
        return str(self)


def get_local_classifier_path(target_path: ModelPath, version: str) -> Path:
    """Returns a path for a classifier file."""
    from knowledge_graph.config import classifier_dir, wandb_model_artifact_filename

    return classifier_dir / target_path / version / wandb_model_artifact_filename


def __getattr__(name):
    """Only import particular classifiers when they are actually requested"""
    if name == "EmbeddingClassifier":
        # This adds a special case for embeddings because they rely on very large external
        # libraries. Importing these libraries is very slow and having them installed
        # requires much more disc space, so we gave them a distinct group in the
        # pyproject.toml file (see: f53a404).
        module = importlib.import_module(".embedding", __package__)
        return getattr(module, name)
    elif name == "BertBasedClassifier":
        module = importlib.import_module(".bert_based", __package__)
        return getattr(module, name)
    elif name in (
        "EmissionsReductionTargetClassifier",
        "NetZeroTargetClassifier",
        "TargetClassifier",
    ):
        module = importlib.import_module(".targets", __package__)
        return getattr(module, name)
    elif name in (
        "BaseLLMClassifier",
        "LLMClassifier",
        "LocalLLMClassifier",
        "AutoLLMClassifier",
    ):
        module = importlib.import_module(".large_language_model", __package__)
        return getattr(module, name)
    else:
        return globals()[name]


__all__ = [
    "Classifier",
    "KeywordClassifier",
    "EmbeddingClassifier",  # type: ignore
    "EmissionsReductionTargetClassifier",  # type: ignore
    "NetZeroTargetClassifier",  # type: ignore
    "TargetClassifier",  # type: ignore
    "BertBasedClassifier",  # type: ignore
    "LLMClassifier",  # type: ignore
    "LocalLLMClassifier",  # type: ignore
    "AutoLLMClassifier",  # type: ignore
    "GPUBoundClassifier",
    "ModelPath",
    "get_local_classifier_path",
]


def create_classifier(
    concept: Concept, classifier_type: str, classifier_kwargs: dict[str, Any]
) -> Classifier:
    """
    Create a classifier from its type and any kwargs.

    :raises ValueError: if classifier_type is unknown
    """

    try:
        classifier_class = __getattr__(classifier_type)
        return classifier_class(concept=concept, **classifier_kwargs)

    except (ImportError, AttributeError) as e:
        raise ValueError(
            f"Unknown classifier type: '{classifier_type}'. "
            f"Available types: {', '.join(__all__)}"
        ) from e


class ClassifierFactory:
    # Map of Wikibase IDs to their bespoke classifier classes
    bespoke_classifier_map: dict[WikibaseID, tuple[str, str]] = {
        WikibaseID("Q1651"): ("TargetClassifier", ".targets"),
        WikibaseID("Q1652"): ("EmissionsReductionTargetClassifier", ".targets"),
        WikibaseID("Q1653"): ("NetZeroTargetClassifier", ".targets"),
        WikibaseID("Q1829"): ("BertBasedClassifier", ".bert_based"),
    }

    @staticmethod
    def create(
        concept: Concept,
        classifier_type: str | None = None,
        classifier_kwargs: dict[str, Any] = {},
    ) -> Classifier:
        """Create a classifier for a concept, depending on its attributes"""

        if classifier_type is not None:
            return create_classifier(
                concept=concept,
                classifier_type=classifier_type,
                classifier_kwargs=classifier_kwargs,
            )

        # Check whether we have a bespoke classifier for the concept
        if (
            concept.wikibase_id is not None
            and concept.wikibase_id in ClassifierFactory.bespoke_classifier_map
        ):
            name, module_path = ClassifierFactory.bespoke_classifier_map[
                concept.wikibase_id
            ]
            module = importlib.import_module(module_path, __package__)
            classifier_class = getattr(module, name)
            return classifier_class(concept)

        # Then handle more generic cases
        return KeywordClassifier(concept)
