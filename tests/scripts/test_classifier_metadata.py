import textwrap
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import pytest

from flows.classifier_specs.spec_interface import DontRunOnEnum
from scripts.classifier_metadata import update, update_entire_env
from src.cloud import AwsEnv
from src.identifiers import ClassifierID, WikibaseID


@dataclass
class MetadataTestCase:
    """Test case for metadata operations."""

    description: str
    clear_dont_run_on: bool
    add_dont_run_on: list[DontRunOnEnum]
    clear_require_gpu: bool
    add_require_gpu: bool
    initial_metadata: dict
    expected_metadata: dict


@pytest.fixture
def mock_wandb_context():
    """Mock wandb.init context manager and artifact."""
    with patch("scripts.classifier_metadata.wandb.init") as mock_init:
        # Mock the context manager
        mock_run = Mock()
        mock_init.return_value = nullcontext(mock_run)

        # Mock artifact metadata
        mock_artifact = Mock()
        mock_run.use_artifact.return_value = mock_artifact

        yield mock_run, mock_artifact


@pytest.mark.parametrize(
    "test_case",
    [
        MetadataTestCase(
            description="Clear dont_run_on completely",
            clear_dont_run_on=True,
            add_dont_run_on=None,
            clear_require_gpu=False,
            add_require_gpu=False,
            initial_metadata={"dont_run_on": ["sabin", "cclw"]},
            expected_metadata={},
        ),
        MetadataTestCase(
            description="Add to existing dont_run_on list",
            clear_dont_run_on=False,
            add_dont_run_on=[DontRunOnEnum.gef, DontRunOnEnum.unfccc],
            clear_require_gpu=False,
            add_require_gpu=False,
            initial_metadata={"dont_run_on": ["sabin"]},
            expected_metadata={"dont_run_on": ["sabin", "gef", "unfccc"]},
        ),
        MetadataTestCase(
            description="Clear and replace dont_run_on list",
            clear_dont_run_on=True,
            add_dont_run_on=[DontRunOnEnum.cpr, DontRunOnEnum.af],
            clear_require_gpu=False,
            add_require_gpu=False,
            initial_metadata={"dont_run_on": ["sabin", "cclw"]},
            expected_metadata={"dont_run_on": ["cpr", "af"]},
        ),
        MetadataTestCase(
            description="Clear GPU requirement and add dont_run_on",
            clear_dont_run_on=False,
            add_dont_run_on=[DontRunOnEnum.cpr],
            clear_require_gpu=True,
            add_require_gpu=False,
            initial_metadata={"compute_environment": {"gpu": True}},
            expected_metadata={"dont_run_on": ["cpr"]},
        ),
        MetadataTestCase(
            description="Add GPU requirement to empty metadata",
            clear_dont_run_on=False,
            add_dont_run_on=None,
            clear_require_gpu=False,
            add_require_gpu=True,
            initial_metadata={},
            expected_metadata={"compute_environment": {"gpu": True}},
        ),
        MetadataTestCase(
            description="Clear dont_run_on, add new entries and require GPU",
            clear_dont_run_on=True,
            add_dont_run_on=[DontRunOnEnum.cpr, DontRunOnEnum.af, DontRunOnEnum.cif],
            clear_require_gpu=False,
            add_require_gpu=True,
            initial_metadata={"dont_run_on": ["sabin", "cclw"]},
            expected_metadata={
                "dont_run_on": ["cif", "cpr", "af"],
                "compute_environment": {"gpu": True},
            },
        ),
    ],
    ids=lambda test_case: test_case.description,
)
def test_classifier_metadata__update(mock_wandb_context, test_case: MetadataTestCase):
    mock_run, mock_artifact = mock_wandb_context
    mock_artifact.metadata = test_case.initial_metadata

    # Run the main function
    update(
        wikibase_id=WikibaseID("Q123"),
        classifier_id=ClassifierID("abcd2345"),
        clear_dont_run_on=test_case.clear_dont_run_on,
        add_dont_run_on=test_case.add_dont_run_on,
        clear_require_gpu=test_case.clear_require_gpu,
        add_require_gpu=test_case.add_require_gpu,
        aws_env=AwsEnv.labs,
        update_specs=False,
    )

    # Check we updated the classifiers metadata
    mock_run.use_artifact.assert_called_once_with("Q123/abcd2345:labs")
    assert sorted(mock_artifact.metadata) == sorted(test_case.expected_metadata), (
        f"Test case '{test_case.description}' failed: "
        f"Expected metadata {test_case.expected_metadata}, "
        f"but got {mock_artifact.metadata}"
    )
    mock_artifact.save.assert_called_once()


def test_classifier_metadata__update_entire_env(mock_wandb_context):
    mock_run, mock_artifact = mock_wandb_context
    mock_artifact.metadata = {"dont_run_on": ["sabin"]}

    sample_data = textwrap.dedent("""
        ---
        - wikibase_id: Q368
          classifier_id: abcd2345
          wandb_registry_version: v3
        - wikibase_id: Q123
          classifier_id: efgh2345
          wandb_registry_version: v1
        - wikibase_id: Q999
          classifier_id: jkmn2345
          wandb_registry_version: v2
    """).lstrip()

    with TemporaryDirectory() as temp_dir:
        temp_spec_dir = Path(temp_dir)
        spec_file = temp_spec_dir / "sandbox.yaml"
        with open(spec_file, "w") as f:
            f.write(sample_data)

        with patch("flows.classifier_specs.spec_interface.SPEC_DIR", temp_spec_dir):
            update_entire_env(
                clear_dont_run_on=True,
                add_dont_run_on=[DontRunOnEnum.oep],
                aws_env=AwsEnv.sandbox,
                update_specs=False,
            )

            # Simple checks: verify the operations happened 3 times (once per classifier)
            artifacts_used = []
            for use_artifact_calls in mock_run.use_artifact.call_args_list:
                artifacts_used.extend(use_artifact_calls.args)

            assert set(artifacts_used) == set(
                [
                    "Q368/abcd2345:sandbox",
                    "Q123/efgh2345:sandbox",
                    "Q999/jkmn2345:sandbox",
                ]
            )
            assert mock_artifact.save.call_count == 3
