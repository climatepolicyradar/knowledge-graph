import json
import os
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from scripts.config import get_git_root
from src.classifier import Classifier
from src.concept import Concept


@pytest.fixture(scope="function")
def aws_credentials():
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"


@pytest.fixture
def metrics_df() -> pd.DataFrame:
    fixture_path = Path(
        "tests/fixtures/data/processed/classifiers_performance/Q787.json"
    )
    with open(fixture_path) as f:
        data = json.load(f)

    return pd.DataFrame(data)


@pytest.fixture
def concept_wikibase_id() -> str:
    return "Q787"


@pytest.fixture
def concept() -> Concept:
    fixture_path = Path("tests/fixtures/data/processed/concepts/Q787.json")
    return Concept.load(fixture_path)


@pytest.fixture
def classifier() -> Classifier:
    fixture_path = Path("tests/fixtures/data/processed/classifiers/Q787")
    return Classifier.load(fixture_path)


@pytest.fixture
def run_just_command():
    repo_root = get_git_root()
    if repo_root is None:
        raise ValueError("Could not find the root of the git repository")

    def _run_command(command_name):
        # Split the command into parts to handle arguments correctly
        command_parts = command_name.split()
        result = subprocess.run(
            ["just"] + command_parts, capture_output=True, text=True, cwd=repo_root
        )
        return result

    return _run_command
