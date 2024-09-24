from pathlib import Path

# directories
data_dir = Path("data")
raw_data_dir = data_dir / "raw"
interim_data_dir = data_dir / "interim"
processed_data_dir = data_dir / "processed"
classifier_dir = processed_data_dir / "classifiers"
concept_dir = processed_data_dir / "concepts"

# aws
aws_region = "eu-west-1"

# sampling config
SAMPLE_SIZE = 130
NEGATIVE_PROPORTION = 0.2
STRATIFIED_COLUMNS = ["world_bank_region", "dataset_name"]
EQUAL_COLUMNS = ["translated"]
