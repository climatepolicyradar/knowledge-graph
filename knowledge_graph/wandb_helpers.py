"""Some helper functions for IO with Weights & Biases, to ensure consistency."""

import logging
import tempfile
from pathlib import Path

import wandb
from wandb.sdk.wandb_run import Run as WandbRun

from knowledge_graph.classifier import Classifier
from knowledge_graph.concept import Concept
from knowledge_graph.labelled_passage import LabelledPassage

logger = logging.getLogger(__name__)


class WandbArtifactNotFoundError(Exception):
    """Exception raised when classifiers don't share the same concept."""

    def __init__(self, run_name: str, artifact_type):
        self.message = f"No artifacts of type {artifact_type} found in run {run_name}/"
        super().__init__(self.message)


def log_labelled_passages_artifact_to_wandb_run(
    labelled_passages: list[LabelledPassage],
    run: WandbRun,
    concept: Concept,
    classifier: Classifier | None = None,
):
    """Upload a list of labelled passages from a classifier to Weights & Biases."""

    artifact_name = (
        f"{classifier.id}-labelled-passages" if classifier else "labelled-passages"
    )
    artifact_metadata = {
        "concept_wikibase_revision": concept.wikibase_revision,
        "passage_count": len(labelled_passages),
    }

    if classifier:
        artifact_metadata |= {"classifier_id": classifier.id}

    labelled_passages_artifact = wandb.Artifact(
        name=artifact_name, type="labelled_passages", metadata=artifact_metadata
    )

    with labelled_passages_artifact.new_file(
        "labelled_passages.jsonl", mode="w", encoding="utf-8"
    ) as f:
        data = "\n".join([entry.model_dump_json() for entry in labelled_passages])
        f.write(data)

    run.log_artifact(labelled_passages_artifact)


def load_labelled_passages_from_wandb_run(
    run: WandbRun,
) -> list[LabelledPassage]:
    """
    Loads labelled passages from a W&B run.

    These are any logged artifact with type 'labelled_passages' and suffix '.jsonl'.

    :raises WandbArtifactNotFoundError: if no labelled passage artifacts are found for
        the given run
    """

    labelled_passage_artifacts = [
        artifact
        for artifact in run.logged_artifacts()  # type: ignore[attr-defined]
        if artifact.type == "labelled_passages"
    ]

    if len(labelled_passage_artifacts) == 0:
        raise WandbArtifactNotFoundError(str(run), artifact_type="labelled_passages")

    labelled_passages = []

    for artifact in labelled_passage_artifacts:
        with tempfile.TemporaryDirectory() as temp_dir:
            artifact_dir = artifact.download(root=temp_dir)
            jsonl_files = list(Path(artifact_dir).glob("*.jsonl"))

            if not jsonl_files:
                logger.warning(
                    f"⚠️ No JSON files found in labelled_passages artifact {artifact} for run {run.name}"
                )
                continue

            artifact_labelled_passages = []
            for json_file in jsonl_files:
                artifact_labelled_passages += LabelledPassage.from_jsonl(json_file)

            labelled_passages.extend(artifact_labelled_passages)

    return labelled_passages
