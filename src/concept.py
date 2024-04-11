import os
from typing import Dict, List, Optional

from pydantic import BaseModel, root_validator, validator


class Concept(BaseModel):
    """Base class for a concept"""

    preferred_label: str
    alternative_labels: List[str] = []
    wikibase_id: Optional[str] = None
    subconcepts: List["Concept"] = []
    related_concepts: List["Concept"] = []

    def __getitem__(self, key: str) -> "Concept":
        """
        Index into the concept's subconcepts according to their preferred_label

        :param str key: The preferred_label of the subconcept to retrieve
        :raises KeyError: Raised if the key is not found in the subconcepts
        :return Concept: The subconcept with the given preferred_label
        """
        for subconcept in self.subconcepts:
            if subconcept.preferred_label == key:
                return subconcept
        raise KeyError(f'"{key}" not found in subconcepts of "{self.preferred_label}"')

    @validator("alternative_labels")
    @classmethod
    def _ensure_alternative_labels_are_unique(cls, values: List[str]) -> List[str]:
        """Ensure that the alternative labels are a unique set of strings"""
        return list(set(str(item) for item in values))

    @root_validator
    @classmethod
    def _ensure_preferred_label_not_in_alternative_labels(cls, values: Dict) -> Dict:
        """Ensure that the preferred label is not in the alternative labels"""
        if values.get("preferred_label") in values.get("alternative_labels", []):
            # remove the preferred label from the alternative labels
            values["alternative_labels"].remove(values.get("preferred_label"))
        return values

    def dict(self) -> dict:
        """Return a dictionary representation of the concept"""
        return {
            "preferred_label": self.preferred_label,
            "alternative_labels": list(self.alternative_labels),
            "subconcepts": [c.dict() for c in self.subconcepts],
        }

    @classmethod
    def from_dict(cls, concept_dict: Dict) -> "Concept":
        """Create a Concept instance from a dictionary"""
        return cls(**concept_dict)

    def __repr__(self) -> str:
        """Return a string representation of the concept"""
        n_subconcepts = f"{len(self.subconcepts)} subconcept{'' if len(self.subconcepts) == 1 else 's'}"
        return f'Concept("{self.preferred_label}", {n_subconcepts})'

    @property
    def all_subconcepts(self) -> List["Concept"]:
        """Return a list of all subconcepts, including subconcepts of subconcepts"""
        all_subconcepts = []
        for subconcept in self.subconcepts:
            all_subconcepts.append(subconcept)
            all_subconcepts.extend(subconcept.all_subconcepts)
        return all_subconcepts

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
