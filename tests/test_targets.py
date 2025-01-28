from unittest.mock import MagicMock

import pytest
from dotenv import find_dotenv, load_dotenv

from src.classifier.targets import (
    EmissionsReductionTargetClassifier,
    NetZeroTargetClassifier,
    TargetClassifier,
)
from src.concept import Concept
from src.identifiers import WikibaseID

load_dotenv(find_dotenv())


@pytest.mark.transformers
def test_target_classifier():
    concept = MagicMock(
        spec=Concept, wikibase_id=WikibaseID("Q1651"), preferred_label="Target"
    )
    classifier = TargetClassifier(concept=concept)
    classifier.pipeline = MagicMock(
        return_value=[
            [
                {"label": "NZT", "score": 0.9277845025062561},
                {"label": "Reduction", "score": 0.8931348323822021},
                {"label": "Other", "score": 0.055339086800813675},
            ],
            [
                {"label": "NZT", "score": 0.15993043780326843},
                {"label": "Reduction", "score": 0.9584472179412842},
                {"label": "Other", "score": 0.027926167473196983},
            ],
        ]
    )

    results = classifier.predict_batch(["text1", "text2"], threshold=0.5)
    assert len([r for r in results if r]) == 2


@pytest.mark.transformers
def test_net_zero_target_classifier():
    concept = MagicMock(
        spec=Concept,
        wikibase_id=WikibaseID("Q1653"),
        preferred_label="Emissions reduction target",
    )
    classifier = NetZeroTargetClassifier(concept=concept)
    classifier.pipeline = MagicMock(
        return_value=[
            [
                {"label": "NZT", "score": 0.9142044186592102},
                {"label": "Reduction", "score": 0.04552844911813736},
                {"label": "Other", "score": 0.07590094953775406},
            ],
            [
                {"label": "NZT", "score": 0.15993043780326843},
                {"label": "Reduction", "score": 0.9584472179412842},
                {"label": "Other", "score": 0.027926167473196983},
            ],
            [
                {"label": "NZT", "score": 0.9277845025062561},
                {"label": "Reduction", "score": 0.8931348323822021},
                {"label": "Other", "score": 0.055339086800813675},
            ],
        ]
    )

    results = classifier.predict_batch(["text1", "text2", "text3"], threshold=0.5)
    assert len([r for r in results if r]) == 2


@pytest.mark.transformers
def test_emissions_reduction_target_classifier():
    concept = MagicMock(
        spec=Concept, wikibase_id=WikibaseID("Q1652"), preferred_label="Target"
    )
    classifier = EmissionsReductionTargetClassifier(concept=concept)
    classifier.pipeline = MagicMock(
        return_value=[
            [
                # only NZT
                {"label": "NZT", "score": 0.9142044186592102},
                {"label": "Reduction", "score": 0.04552844911813736},
                {"label": "Other", "score": 0.07590094953775406},
            ],
            [
                # only Reduction
                {"label": "NZT", "score": 0.15993043780326843},
                {"label": "Reduction", "score": 0.9584472179412842},
                {"label": "Other", "score": 0.027926167473196983},
            ],
            [
                # both NZT and Reduction
                {"label": "NZT", "score": 0.9277845025062561},
                {"label": "Reduction", "score": 0.8931348323822021},
                {"label": "Other", "score": 0.055339086800813675},
            ],
            [
                # neither NZT nor Reduction
                {"label": "NZT", "score": 0.0277845025062561},
                {"label": "Reduction", "score": 0.2931348323822021},
                {"label": "Other", "score": 0.055339086800813675},
            ],
            [
                # NZT and Other
                {"label": "NZT", "score": 0.9277845025062561},
                {"label": "Reduction", "score": 0.055339086800813675},
                {"label": "Other", "score": 0.8931348323822021},
            ],
        ]
    )

    results = classifier.predict_batch(
        ["text1", "text2", "text3", "text4", "text5"], threshold=0.5
    )
    assert len([r for r in results if r]) == 4
