import subprocess
from pathlib import Path
from typing import Optional


def get_git_root() -> Optional[Path]:
    try:
        git_root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"], universal_newlines=True
        ).strip()
        return Path(git_root)
    except subprocess.CalledProcessError:
        # This exception is raised if the command returns a non-zero exit status
        # (i.e., we're not in a git repository)
        return None
    except FileNotFoundError:
        # This exception is raised if the 'git' command is not found
        print("Git command not found. Make sure Git is installed and in your PATH.")
        return None


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
