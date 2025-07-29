from pathlib import Path

from scripts.config import classifier_dir
from src.classifier import Classifier


def get_local_classifier_path(classifier: Classifier) -> Path:
    return (
        classifier_dir / str(classifier.concept.wikibase_id) / f"{classifier.id}.pickle"
    )
