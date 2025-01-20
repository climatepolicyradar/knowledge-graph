import subprocess
from pathlib import Path


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


# directories
DEFAULT_DATA_DIR_NAME = "data"
root_dir = get_git_root()
data_dir = root_dir / DEFAULT_DATA_DIR_NAME if root_dir else Path(DEFAULT_DATA_DIR_NAME)
raw_data_dir = data_dir / "raw"
interim_data_dir = data_dir / "interim"
processed_data_dir = data_dir / "processed"
classifier_dir = processed_data_dir / "classifiers"
metrics_dir = processed_data_dir / "classifiers_performance"
concept_dir = processed_data_dir / "concepts"
# files
model_artifact_name = "model.pickle"

# aws
aws_region = "eu-west-1"

# sampling
equity_columns = [
    "translated",
    "world_bank_region",
    "document_metadata.corpus_type_name",
]
