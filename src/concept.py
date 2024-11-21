import os
import random
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator

from src.identifiers import WikibaseID
from src.labelled_passage import LabelledPassage

if TYPE_CHECKING:
    # only import this circular dependency if we're running in a type-checking
    # environment, eg for pyright
    from src.wikibase import WikibaseSession


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
    wikibase_id: Optional[WikibaseID] = Field(
        default=None, description="The Wikibase ID for the concept"
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
    definition: Optional[str] = Field(
        default=None, description="The definition of the concept"
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
        if any(
            label in values["alternative_labels"] for label in values["negative_labels"]
        ):
            raise ValueError(
                "A negative label should not be the same as a positive label"
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

    def __hash__(self) -> int:
        """Return a unique hash for the concept"""
        return hash((self.wikibase_id, self.preferred_label, *self.alternative_labels))

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
        output should be human-readable, suitable for human or LLM consumption. For
        example, these formatted versions of the concept might be used to ground a RAG
        response if the concept is mentioned in the query.

        :param WikibaseSession wikibase: A Wikibase session
        """
        formatted_concept = [f"# {self.preferred_label}"]

        if self.description:
            formatted_concept.extend(["\n## Description", self.description, ""])

        if self.definition:
            formatted_concept.extend(["## Definition", self.definition, ""])

        if self.alternative_labels:
            formatted_concept.extend(
                [
                    "## Alternative Names",
                    "\n".join(f"- {label}" for label in self.alternative_labels),
                    "",
                ]
            )

        if self.negative_labels:
            formatted_concept.extend(
                [
                    "## Not to be Confused With",
                    "\n".join(f"- {label}" for label in self.negative_labels),
                    "",
                ]
            )

        if wikibase:
            parent_names = [
                wikibase.get_concept(parent_id).preferred_label
                for parent_id in self.subconcept_of
            ]
            if parent_names:
                formatted_concept.extend(
                    [
                        "## Parent Concepts",
                        "\n".join(f"- {name}" for name in parent_names),
                        "",
                    ]
                )

            subconcept_names = [
                wikibase.get_concept(subconcept_id).preferred_label
                for subconcept_id in self.has_subconcept
            ]
            if subconcept_names:
                formatted_concept.extend(
                    [
                        "## Subconcepts",
                        "\n".join(f"- {name}" for name in subconcept_names),
                        "",
                    ]
                )

            related_names = [
                wikibase.get_concept(related_id).preferred_label
                for related_id in self.related_concepts
            ]
            if related_names:
                formatted_concept.extend(
                    [
                        "## Related Concepts",
                        "\n".join(f"- {name}" for name in related_names),
                        "",
                    ]
                )

        positive_labelled_passages = [
            passage
            for passage in self.labelled_passages
            if any(span.concept_id == self.wikibase_id for span in passage.spans)
        ]

        if positive_labelled_passages:
            formatted_concept.extend(
                [
                    "## Example Passages",
                    *[
                        "> " + passage.text.replace("\n", "\n> ") + "\n"
                        for passage in random.sample(
                            positive_labelled_passages,
                            min(5, len(positive_labelled_passages)),
                        )
                    ],
                ]
            )

        return "\n".join(formatted_concept)

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
