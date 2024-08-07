import hashlib


def generate_identifier(input_string: str) -> str:
    """
    Generates a neat identifier using eight unambiguous lowercase and numeric characters

    The resulting identifiers look something like this: ["2sgknw32", "gg7h2j2s", ...]

    With a set of 31 possible characters and 8 positions, this function is able to
    generate 31^8 = 852,891,037,441 unique identifiers. This should be more than enough
    for most use cases!

    :param str input_string: the string to generate the identifier from
    :return str: a unique identifier based on the input string
    """
    # the following list of characters excludes "i", "l", "1", "o", "0" to minimise
    # ambiguity when reading the identifiers
    characters = "abcdefghjkmnpqrstuvwxyz23456789"
    hashed_data = hashlib.sha256(input_string.encode()).digest()

    output = []
    for i in range(8):
        hash_byte = hashed_data[i]
        character_index = hash_byte % len(characters)
        output.append(characters[character_index])

    return "".join(output)
