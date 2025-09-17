import os
import random
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from knowledge_graph.identifiers import ConceptID, WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage

if TYPE_CHECKING:
    # only import this circular dependency if we're running in a type-checking
    # environment, eg for pyright
    from knowledge_graph.wikibase import WikibaseSession


class Concept(BaseModel):
    """Base class for a concept"""

    preferred_label: str = Field(
        ..., description="The preferred label for the concept", min_length=1
    )
    alternative_labels: list[str] = Field(
        default_factory=list, description="List of alternative labels for the concept"
    )
    negative_labels: list[str] = Field(
        default_factory=list,
        description=(
            "Labels which should not be matched instances of the concept. "
            "Negative labels should be unique, and cannot overlap with positive labels."
        ),
    )
    description: Optional[str] = Field(
        default=None,
        description=(
            "A short description of the concept which should be sufficient to "
            "disambiguate it from other concepts with similar labels"
        ),
    )
    definition: Optional[str] = Field(
        default=None,
        description=(
            "A more exhaustive definition of the concept, which should be enough for a "
            "human labeller to identify instances of the concept in a given text. "
        ),
    )
    wikibase_id: Optional[WikibaseID] = Field(
        default=None, description="The Wikibase ID for the concept"
    )
    wikibase_revision: Optional[int] = Field(
        default=None, description="The Wikibase revision for the concept"
    )
    subconcept_of: list[WikibaseID] = Field(
        default_factory=list,
        description="List of parent concept IDs",
    )
    has_subconcept: list[WikibaseID] = Field(
        default_factory=list, description="List of subconcept IDs"
    )
    related_concepts: list[WikibaseID] = Field(
        default_factory=list, description="List of related concept IDs"
    )
    negative_concepts: list[WikibaseID] = Field(
        default_factory=list,
        description=(
            "List of concept IDs for concepts this should not be confused with"
        ),
    )
    recursive_subconcept_of: Optional[list[WikibaseID]] = Field(
        default=None,
        description="List of all parent concept IDs, recursively up the hierarchy",
    )
    recursive_has_subconcept: Optional[list[WikibaseID]] = Field(
        default=None,
        description="List of all subconcept IDs, recursively down the hierarchy",
    )
    labelled_passages: list[LabelledPassage] = Field(
        default_factory=list,
        description="List of labelled passages which mention the concept",
    )

    @field_validator("alternative_labels", mode="before")
    @classmethod
    def _ensure_alternative_labels_are_unique(cls, values: list[str]) -> list[str]:
        """Ensure that the alternative labels are a unique set of strings"""
        return sorted(list(set(str(item) for item in values)))

    @model_validator(mode="before")
    @classmethod
    def _ensure_preferred_label_not_in_alternative_labels(cls, values: Dict) -> Dict:
        """Ensure that the preferred label is not in the alternative labels"""
        preferred_label = values.get("preferred_label")
        if preferred_label in values.get("alternative_labels", []):
            # remove the preferred label from the alternative labels
            values["alternative_labels"].remove(preferred_label)
        return values

    @model_validator(mode="before")
    @classmethod
    def _ensure_negative_labels_are_unique(cls, values: Dict) -> Dict:
        """Ensure that the negative labels are a unique set of strings"""
        negative_labels = values.get("negative_labels", [])
        if len(negative_labels) != len(set(negative_labels)):
            warnings.warn(
                "Duplicate negative labels found. Using unique values.", UserWarning
            )
        values["negative_labels"] = list(set(str(item) for item in negative_labels))
        return values

    @model_validator(mode="before")
    @classmethod
    def _ensure_negative_labels_are_not_in_positive_labels(cls, values: Dict) -> Dict:
        """Raise a ValueError if a negative label is also a positive label"""
        overlapping_labels = []
        for label in values["negative_labels"]:
            if label in values["alternative_labels"]:
                overlapping_labels.append(label)
        if overlapping_labels:
            wikibase_id = values.get("wikibase_id")
            preferred_label = values.get("preferred_label")
            raise ValueError(
                f"{wikibase_id} ({preferred_label}): A negative label should not be "
                f"the same as a positive label. Found in both: {overlapping_labels}"
            )
        return values

    @model_validator(mode="before")
    @classmethod
    def _strip_whitespace_from_labels(cls, values: Dict) -> Dict:
        """Strip leading and trailing whitespace from all labels"""
        for key in ["alternative_labels", "negative_labels"]:
            values[key] = [label.strip() for label in values.get(key, [])]
        values["preferred_label"] = values["preferred_label"].strip()
        return values

    def __repr__(self) -> str:
        """Return a short string representation of the concept"""
        return f"{self.preferred_label} ({self.wikibase_id})"

    def __str__(self) -> str:
        """Return a short string representation of the concept"""
        return self.__repr__()

    @property
    def id(self) -> ConceptID:
        """Return a unique ID for the concept"""
        return ConceptID.generate(
            self.wikibase_id,
            self.preferred_label,
            self.description,
            self.definition,
            *sorted(self.alternative_labels),  # Sort for deterministic ordering
            *sorted(self.negative_labels),  # Sort for deterministic ordering
        )

    def __hash__(self) -> int:
        """Return a hash for the concept"""
        return hash(self.id)

    @property
    def wikibase_url(self) -> str:
        """Return the URL for the concept's Wikibase item, if it exists"""
        if not os.getenv("WIKIBASE_URL"):
            raise ValueError("WIKIBASE_URL environment variable not set")
        if not self.wikibase_id:
            raise ValueError(
                f'No wikibase_id found for concept "{self.preferred_label}"'
            )
        return f"{os.getenv('WIKIBASE_URL')}/wiki/Item:{self.wikibase_id}"

    @property
    def all_labels(self) -> list[str]:
        """Return a list of all unique labels for the concept"""
        return list(set([self.preferred_label] + self.alternative_labels))

    def to_markdown(self, wikibase: Optional["WikibaseSession"] = None) -> str:
        """
        Return a complete representation of the concept in natural language

        Rather than outputting a machine-readable form of the concept (eg in JSON), the
        output should be human-readable, suitable for human or LLM consumption.

        :param WikibaseSession wikibase: A Wikibase session
        """
        sections = []

        # Title
        sections.append(f"# {self.preferred_label}")

        # Description and Definition
        if self.description:
            sections.append("## Description")
            sections.append(self.description)

        if self.definition:
            sections.append("## Definition")
            sections.append(self.definition)

        # Labels
        if self.alternative_labels:
            sections.append(
                "## Alternative labels, synonyms, acronyms, and related terms"
            )
            sections.append(
                "\n".join(f"- {label}" for label in self.alternative_labels)
            )

        if self.negative_labels:
            sections.append("## Not to be confused with")
            sections.append("\n".join(f"- {label}" for label in self.negative_labels))

        # Concept neighbourhood
        if wikibase:
            sections.append("## Concept neighbourhood")
            sections.append(
                "This concept exists within a knowledge graph of other concepts, "
                "with hierarchical and non-hierarchical relationships. Solid arrows "
                "denote hierarchical relationships, where subconcepts are wholly "
                "semantically/conceptually entailed by their parent concept. Dashed "
                "lines show non-hierarchical relationships, which imply semantic "
                "or real-world relatedness between the two concepts which may be "
                "less strict than a hierarchical relationship."
            )

            def sanitize_concept_label(label: str) -> str:
                return label.replace(" ", "_").replace("-", "_")

            mermaid_lines = ["graph TD"]
            self_label = sanitize_concept_label(self.preferred_label)

            # Add parent relationships
            for parent_id in self.subconcept_of:
                parent = wikibase.get_concept(parent_id)
                parent_label = sanitize_concept_label(parent.preferred_label)
                mermaid_lines.append(
                    f"    {parent_label}-->|has subconcept|{self_label}"
                )

                # Add sibling relationships via the parent
                for child_id in parent.has_subconcept:
                    if child_id != self.wikibase_id:
                        child = wikibase.get_concept(child_id)
                        child_label = sanitize_concept_label(child.preferred_label)
                        mermaid_lines.append(
                            f"    {parent_label}-->|has subconcept|{child_label}"
                        )

            # Add child relationships
            for child_id in self.has_subconcept:
                child = wikibase.get_concept(child_id)
                child_label = sanitize_concept_label(child.preferred_label)
                mermaid_lines.append(
                    f"    {self_label}-->|has subconcept|{child_label}"
                )

            # Add related concept relationships
            for related_id in self.related_concepts:
                related = wikibase.get_concept(related_id)
                related_label = sanitize_concept_label(related.preferred_label)
                mermaid_lines.append(f"    {self_label}-.->|related to|{related_label}")

            # Add styling to highlight the current concept
            mermaid_lines.append(
                f"    style {self_label} "
                "fill:#e8f4f8,stroke:#2c5282,stroke-width:2px,rx:5"
            )

            sections.append("```mermaid\n" + "\n".join(mermaid_lines) + "\n```")

        # Example passages

        if positive_passages := [
            passage
            for passage in self.labelled_passages
            if any(span.concept_id == self.wikibase_id for span in passage.spans)
        ]:
            sections.append("## Example passages")
            sections.append(
                "These are examples of passages from real documents which mention the "
                "concept. They are not exhaustive, but should give a sense of the "
                "concept's meaning and usage."
            )
            sample_size = min(5, len(positive_passages))
            for passage in random.sample(positive_passages, sample_size):
                sections.append("> " + passage.text.replace("\n", "\n> "))

        return "\n\n".join(sections)

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the concept to a JSON file at the specified path

        :param Union[str, Path] path: The path to save the concept to
        """
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Concept":
        """
        Load a concept from a JSON file at the specified path

        :param Union[str, Path] path: The path to load the concept from
        :return Concept: The loaded concept
        """
        with open(path, "r", encoding="utf-8") as f:
            return cls.model_validate_json(f.read())
