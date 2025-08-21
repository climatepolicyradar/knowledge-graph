from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional
from unittest.mock import Mock, patch

import pytest

from scripts.classifier_metadata import DontRunOnEnum, main
from scripts.cloud import AwsEnv
from src.identifiers import ClassifierID, WikibaseID


@dataclass
class MetadataTestCase:
    """Test case for metadata operations."""

    clear_dont_run_on: bool
    add_dont_run_on: Optional[list[DontRunOnEnum]]
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
            clear_dont_run_on=True,
            add_dont_run_on=None,
            initial_metadata={"dont_run_on": ["sabin", "cclw"]},
            expected_metadata={},
        ),
        MetadataTestCase(
            clear_dont_run_on=False,
            add_dont_run_on=[DontRunOnEnum.gef, DontRunOnEnum.unfccc],
            initial_metadata={"dont_run_on": ["sabin"]},
            expected_metadata={"dont_run_on": ["sabin", "gef", "unfccc"]},
        ),
        MetadataTestCase(
            clear_dont_run_on=True,
            add_dont_run_on=[DontRunOnEnum.cpr, DontRunOnEnum.af],
            initial_metadata={"dont_run_on": ["sabin", "cclw"]},
            expected_metadata={"dont_run_on": ["cpr", "af"]},
        ),
    ],
)
def test_classifier_metadata(mock_wandb_context, test_case: MetadataTestCase):
    mock_run, mock_artifact = mock_wandb_context
    mock_artifact.metadata = test_case.initial_metadata

    # Run the main function
    main(
        wikibase_id=WikibaseID("Q123"),
        classifier_id=ClassifierID("abcd2345"),
        clear_dont_run_on=test_case.clear_dont_run_on,
        add_dont_run_on=test_case.add_dont_run_on,
        aws_env=AwsEnv.labs,
        update_specs=False,
    )

    # Check we updated the classifiers metadata
    mock_run.use_artifact.assert_called_once_with("Q123/abcd2345:labs")
    assert sorted(mock_artifact.metadata) == sorted(test_case.expected_metadata)
    mock_artifact.save.assert_called_once()
