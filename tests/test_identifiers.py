import subprocess

import pytest
from hypothesis import given
from hypothesis import strategies as st

from knowledge_graph.identifiers import Identifier, WikibaseID
from tests.common_strategies import wikibase_id_strategy

invalid_wikibase_id_strategy = st.text().filter(
    lambda x: not x.startswith("Q") or not x[1:].isdigit() or x == "Q0"
)

valid_identifier_string_strategy = st.text(
    alphabet=Identifier.valid_characters, min_size=8, max_size=8
)

invalid_identifier_string_strategy = st.text().filter(
    lambda x: not Identifier.pattern.fullmatch(x)
)

identifiable_data_strategy = st.lists(
    st.one_of(st.text(), st.integers(), st.floats()), min_size=1, max_size=5
)


@given(wikibase_id_strategy)
def test_whether_valid_wikibase_ids_are_accepted(value):
    wikibase_id = WikibaseID(value)
    assert isinstance(wikibase_id, WikibaseID), (
        f"Expected WikibaseID, got {type(wikibase_id)}"
    )
    assert str(wikibase_id) == value, f"Expected {value}, got {wikibase_id}"


@given(invalid_wikibase_id_strategy)
def test_whether_invalid_wikibase_ids_are_rejected(value):
    with pytest.raises(ValueError):
        WikibaseID(value)


@given(wikibase_id_strategy)
def test_whether_wikibase_ids_are_strings(value):
    wikibase_id = WikibaseID(value)
    assert isinstance(wikibase_id, str), "WikibaseID is not a subclass of str"


def test_whether_wikibase_ids_can_be_compared():
    id1 = WikibaseID("Q123")
    id2 = WikibaseID("Q123")
    id3 = WikibaseID("Q456")

    assert id1 == id2, "Expected id1 == id2"
    assert id1 != id3, "Expected id1 != id3"
    assert id1 == "Q123", "Expected id1 == 'Q123'"
    assert id1 != "Q456", "Expected id1 != 'Q456'"


def test_whether_wikibase_ids_are_sorted_numerically():
    ids = [WikibaseID("Q10"), WikibaseID("Q2"), WikibaseID("Q1")]
    assert sorted(ids) == [WikibaseID("Q1"), WikibaseID("Q2"), WikibaseID("Q10")]


@given(
    st.lists(st.text(min_size=10, max_size=100)),
)
def test_whether_identifier_generation_is_deterministic(input_strings):
    identifiers = [Identifier.generate(input_string) for input_string in input_strings]
    assert identifiers == [
        Identifier.generate(input_string) for input_string in input_strings
    ]


@given(
    st.sets(st.text(min_size=10, max_size=100)),
)
def test_whether_identifiers_are_unique_for_unique_input_strings(input_strings):
    identifiers = [Identifier.generate(input_string) for input_string in input_strings]
    assert len(identifiers) == len(set(identifiers))


def test_whether_default_python_hash_is_consistent_across_distinct_python_processes():
    """Default hashes should not be the same from python session to python session"""
    # Get hash from current process
    hash_a = hash("test")

    # Get hash from a separate Python process
    cmd = "python3 -c \"print(hash('test'))\""
    hash_b = int(subprocess.check_output(cmd, shell=True).decode().strip())

    assert hash_a != hash_b, "Hashes should be different across Python processes"


def test_whether_identifier_generation_is_consistent_across_distinct_python_processes():
    """Deterministic hashes should be the same from python session to python session"""
    # Get id from current process
    id_a = Identifier.generate("test")

    # Get id from a separate Python process
    cmd = "python3 -c \"from knowledge_graph.identifiers import Identifier; print(Identifier.generate('test'))\""
    id_b = subprocess.check_output(cmd, shell=True).decode().strip()

    assert id_a == id_b, (
        "Deterministic hashes should be identical across Python processes"
    )


@given(valid_identifier_string_strategy)
def test_whether_creating_an_identifier_with_valid_string_data_succeeds(value):
    """Test that Identifier can be created from a valid pre-existing ID string."""
    identifier = Identifier(value)
    assert isinstance(identifier, Identifier)
    assert str(identifier) == value


@given(invalid_identifier_string_strategy)
def test_whether_creating_an_identifier_with_invalid_string_data_raises_value_error(
    value,
):
    """Test that Identifier raises ValueError for invalid pre-existing ID strings."""
    with pytest.raises(ValueError):
        Identifier(value)


@given(
    st.one_of(
        st.integers(),
        st.floats(),
        st.lists(st.text()),
        st.dictionaries(st.text(), st.text()),
    )
)
def test_whether_creating_an_identifier_with_non_string_data_raises_type_error(value):
    """Test that Identifier raises TypeError if a non-string is passed to the constructor."""
    with pytest.raises(TypeError):
        Identifier(value)


@given(identifiable_data_strategy)
def test_whether_generated_identifiers_have_correct_properties(args):
    """Test that generated identifiers have correct length and character set."""
    identifier = Identifier.generate(*args)
    assert isinstance(identifier, Identifier)
    assert len(identifier) == 8
    assert all(char in Identifier.valid_characters for char in identifier)
    assert Identifier.pattern.fullmatch(identifier), (
        f"Generated ID '{identifier}' does not match pattern."
    )


def test_whether_identifier_generate_requires_args():
    """Test that Identifier.generate() raises TypeError if no arguments are provided."""
    with pytest.raises(TypeError):
        Identifier.generate()
