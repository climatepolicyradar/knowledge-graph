"""Some helper functions for IO with Weights & Biases, to ensure consistency."""

import logging
import tempfile
from pathlib import Path

import wandb
from wandb.sdk.wandb_run import Run as WandbRun

from flows.utils import (
    deserialise_pydantic_list_with_fallback,
    serialise_pydantic_list_as_jsonl,
)
from knowledge_graph.classifier import Classifier
from knowledge_graph.concept import Concept
from knowledge_graph.labelled_passage import LabelledPassage

logger = logging.getLogger(__name__)


class WandbArtifactNotFoundError(Exception):
    """Exception raised when an artifact of a type is not found in a W&B run."""

    def __init__(self, run_name: str, artifact_type):
        self.message = f"No artifacts of type {artifact_type} found in run {run_name}"
        super().__init__(self.message)


class WandbMultipleArtifactsFoundError(Exception):
    """Exception raised when multiple artifacts of the same type are found in a W&B run."""

    def __init__(self, run_name: str, artifact_type):
        self.message = (
            f"Multiple artifacts of type {artifact_type} found in run {run_name}"
        )
        super().__init__(self.message)


def log_labelled_passages_artifact_to_wandb_run(
    labelled_passages: list[LabelledPassage],
    run: WandbRun,
    concept: Concept,
    classifier: Classifier | None = None,
    artifact_name_prefix: str | None = None,
):
    """Upload a list of labelled passages from a classifier to Weights & Biases."""

    if classifier and artifact_name_prefix:
        raise ValueError(
            "artifact_name_prefix and classifier can not both be provided, as the classifier's ID will be used as prefix to the labelled passages artifact name"
        )

    if classifier:
        artifact_name = f"{classifier.id}-labelled-passages"
    elif artifact_name_prefix:
        artifact_name = f"{artifact_name_prefix}-labelled-passages"
    else:
        artifact_name = "labelled-passages"

    n_positives = len([p for p in labelled_passages if p.spans])
    n_negatives = len(labelled_passages) - n_positives

    artifact_metadata = {
        "concept_wikibase_revision": concept.wikibase_revision,
        "passage_count": len(labelled_passages),
        "n_positives": n_positives,
        "n_negatives": n_negatives,
    }

    if classifier:
        artifact_metadata |= {"classifier_id": classifier.id}

    labelled_passages_artifact = wandb.Artifact(
        name=artifact_name, type="labelled_passages", metadata=artifact_metadata
    )

    with labelled_passages_artifact.new_file(
        "labelled_passages.jsonl", mode="w", encoding="utf-8"
    ) as f:
        f.write(serialise_pydantic_list_as_jsonl(labelled_passages))

    run.log_artifact(labelled_passages_artifact)


def load_artifact_from_wandb_run(
    run: WandbRun,
    artifact_type: str,
    artifact_name: str | None = None,
) -> Path:
    """
    Loads a model artifact from a W&B run and returns the path to the downloaded artifact.

    :param run: The WandB run to load from
    :param artifact_type: The type of artifact to load (e.g., "model", "checkpoint")
    :param artifact_name: Optional name of the artifact to filter by (e.g., "training-data")
    :raises WandbArtifactNotFoundError: if no artifacts of the specified type are found
    :raises WandbMultipleArtifactsFoundError: if multiple artifacts of the specified
        type are found
    :returns: Path to the downloaded artifact directory
    """

    artifacts = [
        artifact
        for artifact in run.logged_artifacts()  # type: ignore[attr-defined]
        if artifact.type == artifact_type
    ]

    if artifact_name is not None:
        artifacts = [a for a in artifacts if a.name == artifact_name]

    if len(artifacts) == 0:
        raise WandbArtifactNotFoundError(str(run), artifact_type=artifact_type)

    if len(artifacts) > 1:
        raise WandbMultipleArtifactsFoundError(str(run), artifact_type=artifact_type)

    artifact = artifacts[0]

    temp_dir = tempfile.mkdtemp()
    artifact_dir = artifact.download(root=temp_dir)

    return Path(artifact_dir)


def load_labelled_passages_from_wandb_run(
    run: WandbRun,
    artifact_name: str | None = None,
) -> list[LabelledPassage]:
    """
    Loads labelled passages from a W&B run.

    These are any logged artifact with type 'labelled_passages' and suffix '.jsonl'.

    :param run: The WandB run to load from
    :param artifact_name: Optional name of the artifact to filter by (e.g., "training-data")
    :raises WandbArtifactNotFoundError: if no labelled passage artifacts are found for
        the given run
    """

    artifact_dir = load_artifact_from_wandb_run(
        run=run, artifact_type="labelled_passages", artifact_name=artifact_name
    )

    jsonl_files = list(Path(artifact_dir).glob("*.jsonl"))

    if not jsonl_files:
        logger.warning(
            f"⚠️ No JSON files found in labelled_passages artifact for run {run.name}"
        )

    labelled_passages = []
    for json_file in jsonl_files:
        with open(json_file, "r", encoding="utf-8") as f:
            file_labelled_passages = [
                LabelledPassage.model_validate_json(line) for line in f
            ]

        labelled_passages += file_labelled_passages

    return labelled_passages


def load_artifact_file_from_wandb(
    wandb_path: str,
    filename: str,
) -> Path:
    """
    Load an artifact file with a known filename from W&B.

    Returns the path to the downloaded file.
    """

    api = wandb.Api()
    artifact = api.artifact(wandb_path)
    artifact_dir = artifact.download()

    return Path(artifact_dir) / filename


def load_labelled_passages_from_wandb(wandb_path: str) -> list[LabelledPassage]:
    """
    Load labelled passages from a W&B path.

    :param str wandb_path: E.g. climatepolicyradar/Q913/rsgz5ygh:v0
    :return list[LabelledPassage]: List of labelled passages
    """

    file_path = load_artifact_file_from_wandb(
        wandb_path=wandb_path,
        filename="labelled_passages.jsonl",
    )

    labelled_passages = deserialise_pydantic_list_with_fallback(
        file_path.read_text(), LabelledPassage
    )

    return labelled_passages
