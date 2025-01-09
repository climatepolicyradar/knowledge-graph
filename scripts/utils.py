from pathlib import Path

from scripts.config import processed_data_dir
from src.classifier import Classifier
from src.concept import Concept


def get_local_classifier_path(concept: Concept, classifier: Classifier) -> Path:
    return (
        processed_data_dir
        / Path("classifiers")
        / str(concept.wikibase_id)
        / f"{classifier.id}.pickle"
    )
