import pytest

from knowledge_graph.classifiers_profiles import (
    ClassifiersProfileMapping,
    ClassifiersProfiles,
    Profile,
)
from knowledge_graph.identifiers import ClassifierID, WikibaseID


@pytest.fixture
def mock_profiles():
    """Classifiers profiles that pass validation"""
    return ClassifiersProfiles(
        [
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
    )


def test_validate_no_errors(mock_profiles):
    """Test that validation passes when there are no errors."""
    # No exception should be raised
    mock_profiles.validate()


def test_validate_too_many_retired_profiles():
    """Test that validation fails when there are too many retired profiles."""
    profiles = ClassifiersProfiles(
        [
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
    )

    with pytest.raises(ValueError, match="Wikibase ID 'Q1' has 4 retired profiles"):
        profiles.validate()


def test_validate_too_many_primary_profiles():
    """Test that validation fails when there are too many primary profiles."""
    profiles = ClassifiersProfiles(
        [
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
    )

    with pytest.raises(ValueError, match="Wikibase ID 'Q1' has 2 primary profiles"):
        profiles.validate()


def test_validate_duplicate_classifier_ids():
    """Test that validation fails when a classifier_id is associated with multiple wikibase_ids."""
    profiles = ClassifiersProfiles(
        [
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
    )

    with pytest.raises(
        ValueError,
        match="Classifier ID 'aaaa4444' is associated with multiple wikibase IDs",
    ):
        profiles.validate()


def test_append_valid_profile():
    """Test that appending a valid profile works."""
    profiles = ClassifiersProfiles()
    profiles.append(
        ClassifiersProfileMapping(
            wikibase_id=WikibaseID("Q123"),
            classifier_id=ClassifierID("aabb3344"),
            classifiers_profile=Profile.PRIMARY,
        )
    )

    assert len(profiles) == 1
    assert profiles[0].wikibase_id == WikibaseID("Q123")


def test_append_invalid_profile():
    """Test that appending an invalid profile raises an error and removes all invalid wikibase IDs from profiles."""
    profiles = ClassifiersProfiles(
        [
            ClassifiersProfileMapping(
                wikibase_id=WikibaseID("Q100"),
                classifier_id=ClassifierID("cxcx2929"),
                classifiers_profile=Profile.PRIMARY,
            ),
        ]
    )

    with pytest.raises(ValueError, match="Wikibase ID 'Q100' has 2 primary profiles"):
        profiles.append(
            ClassifiersProfileMapping(
                wikibase_id=WikibaseID("Q100"),
                classifier_id=ClassifierID("xyxy9292"),
                classifiers_profile=Profile.PRIMARY,
            )
        )
    assert len(profiles) == 0


def test_extend_valid_profiles():
    """Test that extending with valid profiles works."""
    profiles = ClassifiersProfiles()
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
        ]
    )

    assert len(profiles) == 2
    assert profiles[0].classifier_id == ClassifierID("abdd2323")
    assert profiles[1].classifier_id == ClassifierID("wwww8484")


def test_extend_invalid_profiles(mock_profiles):
    """Test that extending with invalid profiles raises an error and removes invalid wikibase IDs from classifiers profiles."""
    profiles = mock_profiles
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

    with pytest.raises(
        ValueError, match="Wikibase ID 'Q999' has 2 experimental profiles"
    ):
        profiles.extend(new_profiles)

    assert len(profiles) == len(mock_profiles)
