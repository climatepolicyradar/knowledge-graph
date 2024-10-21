import re


class Version:
    """A version as mandated by W&B."""

    regex = r"^v\d+$"

    value: int

    def __init__(self, value: str):
        """Create a new instance of Version after validation."""
        self.value = self._validate(value)

    @classmethod
    def _validate(cls, value: str) -> int:
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

    def __hash__(self):
        """Return a hash value for the Version."""
        return hash(self.value)

    def increment(self) -> "Version":
        """Increment the version number by 1."""
        return Version(f"v{self.value + 1}")
