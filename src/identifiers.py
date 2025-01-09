import hashlib
import re


class WikibaseID(str):
    """A Wikibase ID, which is a string that starts with a 'Q' followed by a number."""

    regex = r"^Q[1-9]\d*$"

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
