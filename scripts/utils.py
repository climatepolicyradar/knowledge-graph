from pathlib import Path

from scripts.config import classifier_dir


def get_local_classifier_path(target_path: str, next_version: str) -> Path:
    """Returns a path for a classifier file."""
    return classifier_dir / target_path / f"{next_version}.pickle"
