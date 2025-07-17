from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

from scripts.cloud import AwsEnv
from src.identifiers import Identifier, WikibaseID

SPEC_DIR = Path("flows") / "classifier_specs" / "v2"


class ClassifierSpec(BaseModel):
    """Details for a classifier to run."""

    class ComputeEnvironment(BaseModel):
        """Details about the compute environment the classifier should on."""

        gpu: bool = Field(
            description=("Whether the classifier should be run on a GPU."),
            default=False,
        )

    wikibase_id: WikibaseID = Field(
        description=(
            "The wikibase id for the underlying concept being classified. e.g. 'Q992'"
        ),
    )
    classifier_id: Identifier = Field(
        description=(
            "The unique identifier for the classifier, built from its internals."
        ),
    )
    wandb_registry_version: str = Field(
        description=("The version of the classifier in wandb registry. e.g. v1"),
        pattern=r"^v\d+$",
    )
    compute_environment: Optional[ComputeEnvironment] = Field(
        description=ComputeEnvironment.__doc__,
        default=None,
    )

    def __hash__(self):
        """Make ClassifierSpec hashable for use in sets and as dict keys."""
        return hash(self.classifier_id)

    def __str__(self):
        """Return a string representation of the classifier spec."""
        return f"{self.wikibase_id}:{self.classifier_id}"


def determine_spec_file_path(aws_env: AwsEnv, spec_dir: Path = SPEC_DIR) -> Path:
    """Determine the path to the spec file for a given AWS environment."""
    return spec_dir / f"{aws_env}.yaml"


def load_classifier_specs(
    aws_env: AwsEnv, spec_dir: Path = SPEC_DIR
) -> list[ClassifierSpec]:
    """Load classifier specs into python for a given environment."""
    file_path = determine_spec_file_path(aws_env, spec_dir)

    with open(file_path, "r") as file:
        contents = yaml.load(file, Loader=yaml.FullLoader)

    classifier_specs = []
    for spec in contents:
        classifier_specs.append(ClassifierSpec.model_validate(spec))

    return classifier_specs


if __name__ == "__main__":
    classifier_specs = load_classifier_specs(AwsEnv("sandbox"))
    print(classifier_specs)
