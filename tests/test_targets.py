from src.wikibase import WikibaseSession
from src.classifier.targets import (
    EmissionsReductionTargetClassifier,
    NetZeroTargetClassifier,
    TargetClassifier,
)
from src.concept import Concept

from unittest.mock import MagicMock
from src.identifiers import WikibaseID

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


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
    assert len(results) == 3
