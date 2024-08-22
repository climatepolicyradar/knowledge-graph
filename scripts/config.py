from pathlib import Path

data_dir = Path("data")
raw_data_dir = data_dir / "raw"
interim_data_dir = data_dir / "interim"
processed_data_dir = data_dir / "processed"
classifier_dir = processed_data_dir / "classifiers"

config_dir = data_dir / "config"

aws_region = "eu-west-1"
