from pathlib import Path

# directories
data_dir = Path("data")
raw_data_dir = data_dir / "raw"
interim_data_dir = data_dir / "interim"
processed_data_dir = data_dir / "processed"
classifier_dir = processed_data_dir / "classifiers"
concept_dir = processed_data_dir / "concepts"
config_dir = data_dir / "config"

# aws
aws_region = "eu-west-1"

# classifier config
stratified_columns = ["world_bank_region", "dataset_name"]
equal_columns = ["translated"]
sample_size = 130
negative_proportion = 0.2
