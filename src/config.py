import os
import subprocess
from pathlib import Path
from typing import List


def get_git_root() -> Path:
    """Get the root directory of the git repository."""
    try:
        git_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], universal_newlines=True
        ).strip()
        return Path(git_root)
    except (subprocess.CalledProcessError, FileNotFoundError):
        # If we're not in a git repo or git isn't installed,
        # make a reasonable guess at the root directory
        return Path(__file__).parent.parent


DATA_DIR_NAME = os.getenv("DATA_DIR_NAME", "data")
root_dir = get_git_root()
data_dir = root_dir / DATA_DIR_NAME if root_dir else Path(DATA_DIR_NAME)
raw_data_dir = data_dir / "raw"
interim_data_dir = data_dir / "interim"
processed_data_dir = data_dir / "processed"
classifier_dir = processed_data_dir / "classifiers"
metrics_dir = processed_data_dir / "classifiers_performance"
concept_dir = processed_data_dir / "concepts"

model_artifact_name = os.getenv("MODEL_ARTIFACT_NAME", "model.pickle")

aws_region = os.getenv("AWS_REGION", "eu-west-1")

_equity_columns_str = os.getenv(
    "EQUITY_COLUMNS",
    "translated,world_bank_region,document_metadata.corpus_type_name",
)
equity_columns: List[str] = [col.strip() for col in _equity_columns_str.split(",")]
