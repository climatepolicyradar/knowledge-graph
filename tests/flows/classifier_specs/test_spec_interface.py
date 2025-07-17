import tempfile
import textwrap
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from flows.classifier_specs.spec_interface import ClassifierSpec, load_classifier_specs
from scripts.cloud import AwsEnv
from src.identifiers import WikibaseID


@pytest.mark.parametrize(
    ("spec_dict", "expectation"),
    [
        (  # missing required fields - bad
            {},
            pytest.raises(ValidationError),
        ),
        (
            {  # missing optional fields - fine
                "wikibase_id": WikibaseID("Q123"),
                "classifier_id": "test_classifier",
                "wandb_registry_version": 1,
            },
            does_not_raise(),
        ),
        (
            {  # extra fields - fine
                "wikibase_id": WikibaseID("Q123"),
                "classifier_id": "test_classifier",
                "wandb_registry_version": 1,
                "extra_info": "will be ignored",
            },
            does_not_raise(),
        ),
        (
            {  # all fields - fine
                "wikibase_id": WikibaseID("Q123"),
                "classifier_id": "test_classifier",
                "wandb_registry_version": 1,
                "gpu": True,
            },
            does_not_raise(),
        ),
    ],
)
def test_classifier_spec_creation(spec_dict, expectation):
    """Test creating a ClassifierSpec with the new fields."""
    # missing required fields
    with expectation:
        ClassifierSpec(**spec_dict)


def test_load_classifier_specs():
    """Test loading classifier specs from a YAML file."""
    sample_data = textwrap.dedent("""
        ---
        - wikibase_id: Q368
          classifier_id: ju9239oi
          wandb_registry_version: 3
          gpu: True
        - wikibase_id: Q123
          classifier_id: x438f30k
          wandb_registry_version: 1
          extra_info: will be ignored
        - wikibase_id: Q999
          classifier_id: ju3f93jf
          wandb_registry_version: 2
    """).lstrip()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_spec_dir = Path(temp_dir)
        spec_file = temp_spec_dir / "sandbox.yaml"
        with open(spec_file, "w") as f:
            f.write(sample_data)

        with patch("flows.classifier_specs.spec_interface.SPEC_DIR", temp_spec_dir):
            specs = load_classifier_specs(AwsEnv("sandbox"))

        assert len(specs) == 3
