import re
from enum import Enum

from pydantic import BaseModel, ConfigDict, NonNegativeInt

from src.identifiers import deterministic_hash


class Semantic(Enum):
    """A string that represents a semantic kind of version"""

    Latest = "latest"
    Primary = "primary"


class Version(BaseModel):
    """A version, AKA alias, as mandated by W&B."""

    model_config = ConfigDict(frozen=True)

    value: NonNegativeInt | Semantic

    @classmethod
    def from_str(cls, version_str: str):
        """Create a Version from a string representation."""
        if version_str == "latest":
            return cls(value=Semantic.Latest)

        if not re.match(r"^v\d+$", version_str):
            raise ValueError(
                'version must be in the format "v" followed by a number (e.g., v3)'
            )

        return cls(value=int(version_str[1:]))

    def __str__(self) -> str:
        """Return a string representation of the Version."""
        if isinstance(self.value, Semantic):
            return self.value.value
        else:
            return f"v{self.value}"

    def __eq__(self, other) -> bool:
        """Check if this Version is equal to another Version."""
        return self.value == other.value

    def __gt__(self, other) -> bool:
        """Check if this Version is greater than another Version."""
        if isinstance(self.value, Semantic) or isinstance(other.value, Semantic):
            raise ValueError(f"cannot compare semantic version `{self}`, `{other}`")
        return self.value > other.value

    def __le__(self, other) -> bool:
        """Check if this Version is less than or equal to another Version."""
        if isinstance(self.value, Semantic) or isinstance(other.value, Semantic):
            raise ValueError(f"cannot compare semantic version `{self}`, `{other}`")
        return self.value <= other.value

    def __ge__(self, other) -> bool:
        """Check if this Version is greater than or equal to another Version."""
        if isinstance(self.value, Semantic) or isinstance(other.value, Semantic):
            raise ValueError(f"cannot compare semantic version `{self}`, `{other}`")
        return self.value >= other.value

    def __hash__(self):
        """Return a hash value for the Version."""
        return deterministic_hash(self.value)

    def increment(self):
        """Increment the version number by 1."""
        if isinstance(self.value, Semantic):
            raise ValueError(f"cannot increment semantic version `{self}`")
        return Version(value=self.value + 1)
