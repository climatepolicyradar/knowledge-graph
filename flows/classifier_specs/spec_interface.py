from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field

from src.cloud import AwsEnv
from src.identifiers import ClassifierID, WikibaseID
from src.version import Version

SPEC_DIR = Path("flows") / "classifier_specs" / "v2"


class DontRunOnEnum(Enum):
    """A `source` that will be filtered out in inference."""

    sabin = "sabin"
    cclw = "cclw"
    cpr = "cpr"
    af = "af"
    cif = "cif"
    gcf = "gcf"
    gef = "gef"
    oep = "oep"
    unfccc = "unfccc"

    def __str__(self) -> str:
        """Return a string representation"""
        return self.value


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
    classifier_id: ClassifierID = Field(
        description=(
            "The unique identifier for the classifier, built from its internals."
        ),
    )
    wandb_registry_version: Version = Field(
        description=("The version of the classifier in wandb registry. e.g. v1"),
    )
    compute_environment: Optional[ComputeEnvironment] = Field(
        description=ComputeEnvironment.__doc__,
        default=None,
    )
    dont_run_on: list[DontRunOnEnum] | None = Field(
        description="A list of `source`'s that will be filtered out in inference.",
        default=None,
    )

    def __hash__(self):
        """Make ClassifierSpec hashable for use in sets and as dict keys."""
        return hash(self.classifier_id)

    def __str__(self):
        """Return a string representation of the classifier spec."""
        return f"{self.wikibase_id}:{self.classifier_id}"


def determine_spec_file_path(aws_env: AwsEnv) -> Path:
    """Determine the path to the spec file for a given AWS environment."""
    return SPEC_DIR / f"{aws_env}.yaml"


def load_classifier_specs(
    aws_env: AwsEnv, spec_dir: Path = SPEC_DIR
) -> list[ClassifierSpec]:
    """Load classifier specs into python for a given environment."""
    file_path = determine_spec_file_path(aws_env)

    with open(file_path, "r") as file:
        contents = yaml.load(file, Loader=yaml.FullLoader)

    classifier_specs = []
    for spec in contents:
        classifier_specs.append(ClassifierSpec.model_validate(spec))

    return classifier_specs
