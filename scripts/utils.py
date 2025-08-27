from enum import Enum
from pathlib import Path

from pydantic import BaseModel

from scripts.config import classifier_dir
from src.identifiers import ClassifierID, WikibaseID


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
    return classifier_dir / target_path / version / "model.pickle"


class DontRunOnEnum(Enum):
    """A `source` that will be filtered out in inference."""

    sabin = "sabin"
    cclw = "cclw"
    cpr = "cpr"
    af = "af"
    cif = "cif"
    gcf = "gcf"
    gef = "gef"
    oep = "oep"
    unfccc = "unfccc"

    def __str__(self) -> str:
        """Return a string representation"""
        return self.value
