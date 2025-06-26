import hashlib
import re
from enum import Enum
from functools import total_ordering

from pydantic import BaseModel, Field


@total_ordering
class WikibaseID(str):
    """A Wikibase ID, which is a string that starts with a 'Q' followed by a number."""

    regex = r"^Q[1-9]\d*$"

    @property
    def numeric(self) -> int:
        """The numeric value of the Wikibase ID"""
        return int(self[1:])

    def __lt__(self, other) -> bool:
        """Compare two Wikibase IDs numerically"""
        if isinstance(other, str):
            other = WikibaseID(other)
        if not isinstance(other, WikibaseID):
            return NotImplemented
        return self.numeric < other.numeric

    def __eq__(self, other) -> bool:
        """Check if two Wikibase IDs are equal"""
        if isinstance(other, str):
            other = WikibaseID(other)
        if not isinstance(other, WikibaseID):
            return NotImplemented
        return self.numeric == other.numeric

    def __hash__(self) -> int:
        """Hash a Wikibase ID consistently with string representation"""
        return hash(str(self))

    @classmethod
    def _validate(cls, value: str, field=None) -> str:
        """Validate that the Wikibase ID is in the correct format"""
        if not re.match(cls.regex, value):
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


class VespaSchema(Enum):
    """Schema definitions' names for Vespa"""

    FamilyDocument = "family_document"
    DocumentPassage = "document_passage"


class VespaID(BaseModel):
    """Base class for a Vespa schema ID"""

    prefix: str = Field(description="TODO", frozen=True, default="id:doc_search")
    kind: VespaSchema = Field(description="TODO")
    id: str = Field(description="TODO")

    def __str__(self) -> str:
        """String representation of a Vespa ID, ready to be used with Vespa queries"""
        return f"{self.prefix}:{self.kind.value}::{self.id}"


class FamilyDocumentID(VespaID):
    """An ID for a family document in Vespa"""

    kind: VespaSchema = Field(default=VespaSchema.FamilyDocument, exclude=True)


class DocumentPassageID(VespaID):
    """An ID for a document passage in Vespa"""

    kind: VespaSchema = Field(default=VespaSchema.DocumentPassage, exclude=True)


class Identifier(str):
    """
    A unique ID for a resource, comprised of 8 unambiguous lowercase and numeric characters.

    IDs are generated deterministically from input data, and look something like:
    ["2sgknw32", "gg7h2j2s", ...]

    With a set of 31 possible characters and 8 spaces in the ID, there's a total of
    31^8 = 852,891,037,441 available values in the space. This should be more than
    enough for most use cases!

    Usage:
      To generate an ID: `my_id = Identifier.generate("some", "data")`
      To cast/validate a string: `my_id = Identifier("abcdef12")` (raises ValueError if invalid)
    """

    # the following list of characters excludes "i", "l", "1", "o", "0" to minimise
    # ambiguity when people read the identifiers at a glance
    valid_characters = "abcdefghjkmnpqrstuvwxyz23456789"

    # Pattern needs to be defined using the literal string for valid_characters
    # as class attributes are resolved at class creation time.
    pattern = re.compile(rf"^[{valid_characters}]{{8}}$")

    @classmethod
    def generate(cls, *args) -> "Identifier":
        """Generates a new Identifier from the supplied data."""
        if not args:
            raise TypeError(
                f"{cls.__name__}.generate() requires at least one argument."
            )
        stringified_args = ""
        for arg in args:
            if isinstance(arg, BaseModel):
                stringified_args += arg.model_dump_json()
            else:
                stringified_args += str(arg)
        hashed_data = hashlib.sha256(stringified_args.encode()).digest()
        identifier = "".join(
            cls.valid_characters[b % len(cls.valid_characters)]
            for b in hashed_data[:8]  # Use first 8 bytes of hash
        )
        return cls(identifier)

    @classmethod
    def _validate(cls, value: str) -> str:
        """Validate that the Identifier string is in the correct format"""
        if not isinstance(value, str):
            raise TypeError(
                f"{cls.__name__} value must be a string, received {type(value).__name__}"
            )
        if not cls.pattern.match(value):
            raise ValueError(
                f"'{value}' is not a valid {cls.__name__}. Must be 8 characters from "
                f"the set '{cls.valid_characters}' (pattern: r'{cls.pattern.pattern}')."
            )
        return value

    def __new__(cls, value_to_become_identifier: str) -> "Identifier":
        """
        Create a new Identifier from a string, after validation.

        To generate an Identifier from arbitrary data, use `Identifier.generate(*args)`.
        """
        validated_value = cls._validate(value_to_become_identifier)
        return str.__new__(cls, validated_value)

    @classmethod
    def __get_validators__(cls):
        """Return a generator of validators for Pydantic compatibility."""
        yield cls._validate
