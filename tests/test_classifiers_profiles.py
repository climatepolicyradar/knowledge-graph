import pytest

from flows.result import Err
from knowledge_graph.classifiers_profiles import (
    ClassifiersProfileMapping,
    Profile,
    validate_classifiers_profiles_mappings,
)
from knowledge_graph.identifiers import ClassifierID, WikibaseID


@pytest.fixture
def mock_profiles():
    """Classifiers profiles that pass validation"""
    return [
        ClassifiersProfileMapping(
            wikibase_id=WikibaseID("Q100"),
            classifier_id=ClassifierID("aaaa2222"),
            classifiers_profile=Profile.PRIMARY,
        ),
        ClassifiersProfileMapping(
            wikibase_id=WikibaseID("Q111"),
            classifier_id=ClassifierID("bbbb3333"),
            classifiers_profile=Profile.EXPERIMENTAL,
        ),
        ClassifiersProfileMapping(
            wikibase_id=WikibaseID("Q123"),
            classifier_id=ClassifierID("cccc3333"),
            classifiers_profile=Profile.RETIRED,
        ),
    ]


def test_validate_no_errors(mock_profiles):
    """Test that validation passes when there are no errors."""
    # No exception should be raised
    valid_profiles, errors = validate_classifiers_profiles_mappings(mock_profiles)
    assert len(errors) == 0


def test_validate_too_many_retired_profiles():
    """Test that validation fails when there are too many retired profiles."""
    profiles = [
        ClassifiersProfileMapping(
            wikibase_id=WikibaseID("Q1"),
            classifier_id=ClassifierID("abcd2345"),
            classifiers_profile=Profile.RETIRED,
        ),
        ClassifiersProfileMapping(
            wikibase_id=WikibaseID("Q1"),
            classifier_id=ClassifierID("zzzz3333"),
            classifiers_profile=Profile.RETIRED,
        ),
        ClassifiersProfileMapping(
            wikibase_id=WikibaseID("Q1"),
            classifier_id=ClassifierID("tttt7777"),
            classifiers_profile=Profile.RETIRED,
        ),
        ClassifiersProfileMapping(
            wikibase_id=WikibaseID("Q1"),
            classifier_id=ClassifierID("eeee3333"),
            classifiers_profile=Profile.RETIRED,
        ),
    ]

    valid_profiles, errors = validate_classifiers_profiles_mappings(profiles)
    assert len(errors) == 1
    assert len(valid_profiles) == 0

    failures = [r._error for r in errors if isinstance(r, Err)]
    assert failures[0].metadata == {
        "wikibase_id": {"Q1"},
        "profile": {"retired"},
        "count": {4},
    }
    assert failures[0].msg == "Validation Error: classifier multiplicity"


def test_validate_too_many_primary_profiles():
    """Test that validation fails when there are too many primary profiles."""
    profiles = [
        ClassifiersProfileMapping(
            wikibase_id=WikibaseID("Q1"),
            classifier_id=ClassifierID("aaaa5555"),
            classifiers_profile=Profile.PRIMARY,
        ),
        ClassifiersProfileMapping(
            wikibase_id=WikibaseID("Q1"),
            classifier_id=ClassifierID("jjjj5555"),
            classifiers_profile=Profile.PRIMARY,
        ),
    ]

    valid_profiles, errors = validate_classifiers_profiles_mappings(profiles)
    assert len(errors) == 1
    assert len(valid_profiles) == 0

    failures = [r._error for r in errors if isinstance(r, Err)]
    assert failures[0].metadata == {
        "wikibase_id": {"Q1"},
        "profile": {"primary"},
        "count": {2},
    }
    assert failures[0].msg == "Validation Error: classifier multiplicity"


def test_validate_duplicate_classifier_ids():
    """Test that validation fails when a classifier_id is associated with multiple wikibase_ids."""
    profiles = [
        ClassifiersProfileMapping(
            wikibase_id=WikibaseID("Q1"),
            classifier_id=ClassifierID("aaaa4444"),
            classifiers_profile=Profile.PRIMARY,
        ),
        ClassifiersProfileMapping(
            wikibase_id=WikibaseID("Q2"),
            classifier_id=ClassifierID("aaaa4444"),
            classifiers_profile=Profile.EXPERIMENTAL,
        ),
    ]

    valid_profiles, errors = validate_classifiers_profiles_mappings(profiles)
    assert len(errors) == 1
    assert len(valid_profiles) == 0

    failures = [r._error for r in errors if isinstance(r, Err)]
    assert failures[0].metadata == {
        "wikibase_id": {"Q1", "Q2"},
        "classifier_id": {"aaaa4444"},
    }
    assert (
        failures[0].msg == "Validation Error: classifiers ID has multiple wikibase IDs"
    )


def test_append_valid_profile():
    """Test that appending a valid profile works."""
    profiles = []
    profiles.append(
        ClassifiersProfileMapping(
            wikibase_id=WikibaseID("Q123"),
            classifier_id=ClassifierID("aabb3344"),
            classifiers_profile=Profile.PRIMARY,
        )
    )
    valid_profiles, errors = validate_classifiers_profiles_mappings(profiles)

    assert len(valid_profiles) == 1
    assert valid_profiles[0].wikibase_id == WikibaseID("Q123")
    assert len(errors) == 0


def test_append_invalid_profile():
    """Test that appending an invalid profile raises an error and removes all invalid wikibase IDs from profiles."""
    profiles = [
        ClassifiersProfileMapping(
            wikibase_id=WikibaseID("Q100"),
            classifier_id=ClassifierID("cxcx2929"),
            classifiers_profile=Profile.PRIMARY,
        ),
    ]

    profiles.append(
        ClassifiersProfileMapping(
            wikibase_id=WikibaseID("Q100"),
            classifier_id=ClassifierID("xyxy9292"),
            classifiers_profile=Profile.PRIMARY,
        )
    )

    valid_profiles, errors = validate_classifiers_profiles_mappings(profiles)

    assert len(valid_profiles) == 0
    assert len(errors) == 1

    failures = [r._error for r in errors if isinstance(r, Err)]
    assert failures[0].msg == "Validation Error: classifier multiplicity"


def test_extend_valid_profiles():
    """Test that extending with valid profiles works."""
    profiles = []
    profiles.extend(
        [
            ClassifiersProfileMapping(
                wikibase_id=WikibaseID("Q1111"),
                classifier_id=ClassifierID("abdd2323"),
                classifiers_profile=Profile.PRIMARY,
            ),
            ClassifiersProfileMapping(
                wikibase_id=WikibaseID("Q1111"),
                classifier_id=ClassifierID("wwww8484"),
                classifiers_profile=Profile.EXPERIMENTAL,
            ),
        ],
    )
    valid_profiles, errors = validate_classifiers_profiles_mappings(profiles)

    assert len(valid_profiles) == 2
    assert profiles[0].classifier_id == ClassifierID("abdd2323")
    assert profiles[1].classifier_id == ClassifierID("wwww8484")
    assert len(errors) == 0


def test_extend_invalid_profiles(mock_profiles):
    """Test that extending with invalid profiles raises an error and removes invalid wikibase IDs from classifiers profiles."""
    profiles = []
    profiles.extend(mock_profiles)
    new_profiles = [
        ClassifiersProfileMapping(
            wikibase_id=WikibaseID("Q999"),
            classifier_id=ClassifierID("abcd2345"),
            classifiers_profile=Profile.EXPERIMENTAL,
        ),
        ClassifiersProfileMapping(
            wikibase_id=WikibaseID("Q999"),
            classifier_id=ClassifierID("dacb2345"),
            classifiers_profile=Profile.EXPERIMENTAL,
        ),
    ]
    profiles.extend(new_profiles)

    valid_profiles, errors = validate_classifiers_profiles_mappings(profiles)

    assert len(valid_profiles) == len(mock_profiles)
    assert len(errors) == 1

    failures = [r._error for r in errors if isinstance(r, Err)]
    assert failures[0].msg == "Validation Error: classifier multiplicity"
