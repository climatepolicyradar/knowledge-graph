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


def deterministic_hash(*args) -> int:
    """
    Generate a deterministic hash of the input data using SHA-256.

    Hashes should be consistent across different runs of a program.
    """
    input_string = "".join([str(arg) for arg in args])
    return int.from_bytes(
        hashlib.sha256(input_string.encode()).digest()[:8], byteorder="big"
    )


def generate_identifier(*args) -> str:
    """
    Generates a neat identifier using eight unambiguous lowercase and numeric characters

    The resulting identifiers look something like this: ["2sgknw32", "gg7h2j2s", ...]

    With a set of 31 possible characters and 8 positions, this function is able to
    generate 31^8 = 852,891,037,441 unique identifiers. This should be more than enough
    for most use cases!

    :param args: an arbitrary set of data to be hashed
    :return str: a unique identifier based on the input string
    """
    # the following list of characters excludes "i", "l", "1", "o", "0" to minimise
    # ambiguity when reading the identifiers
    characters = "abcdefghjkmnpqrstuvwxyz23456789"
    hashed_data = hashlib.sha256("".join([str(arg) for arg in args]).encode()).digest()
    return "".join(characters[b % len(characters)] for b in hashed_data[:8])
