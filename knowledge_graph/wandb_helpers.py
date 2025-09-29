"""Some helper functions for IO with Weights & Biases, to ensure consistency."""

import wandb
from wandb.sdk.wandb_run import Run as WandbRun

from knowledge_graph.classifier import Classifier
from knowledge_graph.concept import Concept
from knowledge_graph.labelled_passage import LabelledPassage


def log_labelled_passages_artifact_to_wandb(
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
