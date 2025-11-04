from unittest.mock import AsyncMock

import pytest

from flows.classifiers_profiles import get_classifier_profiles
from flows.result import Err
from knowledge_graph.classifiers_profiles import ClassifiersProfile, Profile
from knowledge_graph.concept import Concept
from knowledge_graph.identifiers import ClassifierID, WikibaseID
from knowledge_graph.wikibase import StatementRank, WikibaseSession


def test_add_classifier_profile():
    pass


def test_remove_classifier_profile():
    pass


def test_update_classifier_profile():
    pass


@pytest.mark.asyncio
async def test_get_classifier_profiles():
    # mock concepts
    list_concepts = [
        Concept(wikibase_id="Q123", preferred_label="Concept 123"),
        Concept(wikibase_id="Q100", preferred_label="Concept 100"),
        Concept(wikibase_id="Q999", preferred_label="Concept 999"),
        Concept(wikibase_id="Q200", preferred_label="Concept 200"),
    ]

    # mock response from wikibase.get_classifier_ids_async
    mock_wikibase = AsyncMock(spec=WikibaseSession)

    # 1 success, 3 failures
    mock_wikibase.get_classifier_ids_async.side_effect = [
        # Q123: success
        [
            (StatementRank.PREFERRED, ClassifierID("aaaa2222")),
            (StatementRank.DEPRECATED, ClassifierID("yyyy9999")),
        ],
        # Q100: failure
        Exception("Failed to fetch classifier profiles for Q100"),
        # Q999: fail validation - 2 primary profiles
        [
            (StatementRank.PREFERRED, ClassifierID("bbbb3333")),
            (StatementRank.PREFERRED, ClassifierID("cccc4444")),
        ],
        # Q200: fail validation - same classifier ID in 2 profiles
        [
            (StatementRank.PREFERRED, ClassifierID("xyzz2345")),
            (StatementRank.DEPRECATED, ClassifierID("xyzz2345")),
        ],
    ]

    # Call the function under test
    classifier_profiles, results = await get_classifier_profiles(
        wikibase=mock_wikibase, concepts=list_concepts
    )

    # assert successful profiles
    assert len(classifier_profiles) == 2
    assert classifier_profiles[0] == ClassifiersProfile(
        wikibase_id=WikibaseID("Q123"),
        classifier_id=ClassifierID("aaaa2222"),
        classifier_profile=Profile.PRIMARY,
    )
    assert classifier_profiles[1] == ClassifiersProfile(
        wikibase_id=WikibaseID("Q123"),
        classifier_id=ClassifierID("yyyy9999"),
        classifier_profile=Profile.RETIRED,
    )

    # assert validation errors
    failures = [r._error for r in results if isinstance(r, Err)]

    assert len(failures) == 3
    assert failures[0].metadata.get("wikibase_id") == "Q100"
    assert failures[0].msg == "Failed to fetch classifier profiles for Q100"

    # check mocked method called
    assert mock_wikibase.get_classifier_ids_async.call_count == 4
