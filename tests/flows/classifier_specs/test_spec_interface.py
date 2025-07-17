import tempfile
import textwrap
from contextlib import nullcontext as does_not_raise
from pathlib import Path

import pytest
from pydantic import ValidationError

from flows.classifier_specs.spec_interface import ClassifierSpec, load_classifier_specs
from scripts.cloud import AwsEnv
from src.identifiers import Identifier, WikibaseID


@pytest.mark.parametrize(
    ("spec_dict", "expectation"),
    [
        (  # missing required fields - bad
            {},
            pytest.raises(ValidationError),
        ),
        (
            {  # Bad registry version - bad
                "wikibase_id": WikibaseID("Q123"),
                "classifier_id": Identifier("abcd2345"),
                "wandb_registry_version": "latest",
            },
            pytest.raises(ValidationError),
        ),
        (
            {  # missing optional fields - fine
                "wikibase_id": WikibaseID("Q123"),
                "classifier_id": Identifier("abcd2345"),
                "wandb_registry_version": "v1",
            },
            does_not_raise(),
        ),
        (
            {  # extra fields - fine
                "wikibase_id": WikibaseID("Q123"),
                "classifier_id": Identifier("abcd2345"),
                "wandb_registry_version": "v1",
                "extra_info": "will be ignored",
            },
            does_not_raise(),
        ),
        (
            {  # all fields - fine
                "wikibase_id": WikibaseID("Q123"),
                "classifier_id": Identifier("abcd2345"),
                "wandb_registry_version": "v1",
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
          classifier_id: abcd2345
          wandb_registry_version: v3
          gpu: True
        - wikibase_id: Q123
          classifier_id: abcd2345
          wandb_registry_version: v1
          extra_info: will be ignored
        - wikibase_id: Q999
          classifier_id: abcd2345
          wandb_registry_version: v2
    """).lstrip()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_spec_dir = Path(temp_dir)
        spec_file = temp_spec_dir / "sandbox.yaml"
        with open(spec_file, "w") as f:
            f.write(sample_data)

        specs = load_classifier_specs(AwsEnv("sandbox"), spec_dir=temp_spec_dir)

        assert len(specs) == 3
