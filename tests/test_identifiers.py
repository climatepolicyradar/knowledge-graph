import pytest
from hypothesis import given
from hypothesis import strategies as st

from src.identifiers import WikibaseID, generate_identifier

valid_wikibase_id = st.from_regex(r"Q[1-9]\d*", fullmatch=True)
invalid_wikibase_id = st.text().filter(
    lambda x: not x.startswith("Q") or not x[1:].isdigit() or x == "Q0"
)


@given(valid_wikibase_id)
def test_whether_valid_wikibase_ids_are_accepted(value):
    wikibase_id = WikibaseID(value)
    assert isinstance(
        wikibase_id, WikibaseID
    ), f"Expected WikibaseID, got {type(wikibase_id)}"
    assert str(wikibase_id) == value, f"Expected {value}, got {wikibase_id}"


@given(invalid_wikibase_id)
def test_whether_invalid_wikibase_ids_are_rejected(value):
    with pytest.raises(ValueError):
        WikibaseID(value)


@given(valid_wikibase_id)
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


@given(
    st.lists(st.text(min_size=10, max_size=100)),
)
def test_whether_identifier_generation_is_deterministic(input_strings):
    identifiers = [generate_identifier(input_string) for input_string in input_strings]
    assert identifiers == [
        generate_identifier(input_string) for input_string in input_strings
    ]


@given(
    st.sets(st.text(min_size=10, max_size=100)),
)
def test_whether_identifiers_are_unique_for_unique_input_strings(input_strings):
    identifiers = [generate_identifier(input_string) for input_string in input_strings]
    assert len(identifiers) == len(set(identifiers))
