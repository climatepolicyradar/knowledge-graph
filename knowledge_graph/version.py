import re

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema


class Version:
    """A version as mandated by W&B."""

    regex = r"^v\d+$"

    value: int

    def __init__(self, value: str):
        """Create a new instance of Version after validation."""
        self.value = self._validate(value)

    def __hash__(self):
        """Return a hash of the Version."""
        return hash(self.value)

    @classmethod
    def _validate(cls, value: str, field=None) -> int:
        if value == "latest":
            raise ValueError("`latest` isn't yet supported")

        if not re.match(cls.regex, value):
            raise ValueError(
                'version must be in the format "v" followed by a number (e.g., v3)'
            )

        version_number = int(value[1:])

        min_version = 0

        if version_number < min_version:
            raise ValueError(
                f"version number must be greater than or equal to {min_version}"
            )

        return version_number

    def __str__(self):
        """Return a string representation of the Version."""
        return f"v{self.value}"

    def __repr__(self):
        """Return a string representation of the Version object."""
        return f"Version('v{self.value}')"

    def __eq__(self, other):
        """Check if this Version is equal to another Version or string."""
        if isinstance(other, Version):
            return self.value == other.value
        return str(self) == other

    def __lt__(self, other):
        """Check if this Version is less than another Version or string."""
        if isinstance(other, Version):
            return self.value < other.value
        return str(self) < other

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type, handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        """Return a generator of validators for Pydantic v2 compatibility."""
        return core_schema.no_info_after_validator_function(
            cls,
            core_schema.union_schema(
                [core_schema.int_schema(), core_schema.str_schema(pattern=cls.regex)]
            ),
        )

    def increment(self) -> "Version":
        """Increment the version number by 1."""
        return Version(f"v{self.value + 1}")
