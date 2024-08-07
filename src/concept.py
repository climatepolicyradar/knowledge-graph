import os
import re
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class WikibaseID(str):
    """A Wikibase ID, which is a string that starts with a 'Q' followed by a number."""

    @classmethod
    def _validate(cls, value: str, field=None) -> str:
        """Validate that the Wikibase ID is in the correct format"""
        if not re.match(r"^Q[1-9]\d*$", value):
            raise ValueError(f"{value} is not a valid Wikibase ID")
        return value

    @classmethod
    def __get_validators__(cls):
        """Return a generator of validators for the WikibaseID class"""
        yield cls._validate

    def __new__(cls, value: str) -> "WikibaseID":
        """Create a new instance of WikibaseID after validation"""
        validated_value = cls._validate(value)
        return str.__new__(cls, validated_value)


class Concept(BaseModel):
    """Base class for a concept"""

    preferred_label: str = Field(..., description="The preferred label for the concept")
    alternative_labels: List[str] = Field(
        default_factory=list, description="List of alternative labels for the concept"
    )
    description: Optional[str] = Field(
        default=None,
        description="A short description of the concept which should be sufficient to disambiguate it from other concepts with similar labels",
    )
    wikibase_id: Optional[WikibaseID] = Field(
        default=None, description="The Wikibase ID for the concept"
    )
    subconcept_of: List[WikibaseID] = Field(
        default_factory=list,
        description="List of parent concept IDs",
    )
    has_subconcept: List[WikibaseID] = Field(
        default_factory=list, description="List of subconcept IDs"
    )
    related_concepts: List[WikibaseID] = Field(
        default_factory=list, description="List of related concept IDs"
    )
    definition: Optional[str] = Field(
        default=None, description="The definition of the concept"
    )

    @field_validator("alternative_labels", mode="before")
    @classmethod
    def _ensure_alternative_labels_are_unique(cls, values: List[str]) -> List[str]:
        """Ensure that the alternative labels are a unique set of strings"""
        return list(set(str(item) for item in values))

    @model_validator(mode="before")
    @classmethod
    def _ensure_preferred_label_not_in_alternative_labels(cls, values: Dict) -> Dict:
        """Ensure that the preferred label is not in the alternative labels"""
        preferred_label = values.get("preferred_label")
        if preferred_label in values.get("alternative_labels", []):
            # remove the preferred label from the alternative labels
            values["alternative_labels"].remove(preferred_label)
        return values

    def __repr__(self) -> str:
        """Return a string representation of the concept"""
        return f'Concept({self.wikibase_id}, "{self.preferred_label}")'

    def __str__(self) -> str:
        """Return a string representation of the concept"""
        return super().__str__()

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
    def all_labels(self) -> List[str]:
        """Return a list of all unique labels for the concept"""
        return list(set([self.preferred_label] + self.alternative_labels))

    def __hash__(self) -> int:
        """Return a unique hash for the concept"""
        return hash((self.wikibase_id, self.preferred_label, *self.alternative_labels))
