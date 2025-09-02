import tempfile
import textwrap
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from flows.classifier_specs.spec_interface import (
    ClassifierSpec,
    DontRunOnEnum,
    load_classifier_specs,
    should_skip_doc,
)
from src.cloud import AwsEnv
from src.identifiers import ClassifierID, WikibaseID


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
                "classifier_id": ClassifierID("abcd2345"),
                "wandb_registry_version": "latest",
            },
            pytest.raises(ValidationError),
        ),
        (
            {  # missing optional fields - fine
                "wikibase_id": WikibaseID("Q123"),
                "classifier_id": ClassifierID("abcd2345"),
                "wandb_registry_version": "v1",
            },
            does_not_raise(),
        ),
        (
            {  # extra fields - fine
                "wikibase_id": WikibaseID("Q123"),
                "classifier_id": ClassifierID("abcd2345"),
                "wandb_registry_version": "v1",
                "extra_info": "will be ignored",
            },
            does_not_raise(),
        ),
        (
            {  # all fields - fine
                "wikibase_id": WikibaseID("Q123"),
                "classifier_id": ClassifierID("abcd2345"),
                "wandb_registry_version": "v1",
                "compute_environment": {
                    "gpu": True,
                },
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
          compute_environment:
            gpu: true
        - wikibase_id: Q123
          classifier_id: abcd2345
          wandb_registry_version: v1
          extra_info: will be ignored
        - wikibase_id: Q999
          classifier_id: abcd2345
          wandb_registry_version: v2
          dont_run_on:
            - sabin
            - cpr
    """).lstrip()

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_spec_dir = Path(temp_dir)
        spec_file = temp_spec_dir / "sandbox.yaml"
        with open(spec_file, "w") as f:
            f.write(sample_data)

        with patch("flows.classifier_specs.spec_interface.SPEC_DIR", temp_spec_dir):
            specs = load_classifier_specs(AwsEnv("sandbox"))

        assert len(specs) == 3
        assert specs[0].compute_environment.gpu
        assert not specs[1].compute_environment
        assert specs[2].dont_run_on == [DontRunOnEnum("sabin"), DontRunOnEnum("cpr")]


@pytest.mark.parametrize(
    "stem, expected",
    [
        ("GCF.document.FP154_22820.12118", False),
        ("CPR.document.i00003209.n0000", False),
        ("AF.document.061MCLAR.n0000_translated_en", True),
        ("Sabin.document.89169.89170", True),
    ],
)
def test_should_skip_doc(stem, expected):
    spec = ClassifierSpec(
        wikibase_id="Q1",
        classifier_id="9999zzzz",
        wandb_registry_version="v1",
        dont_run_on=["af", "sabin"],
    )
    assert expected == should_skip_doc(stem, spec)
