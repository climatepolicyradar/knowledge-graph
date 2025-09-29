from contextlib import nullcontext
from pathlib import Path
from typing import Annotated, Optional

import typer
import wandb
from rich.console import Console

from knowledge_graph.classifier import Classifier
from knowledge_graph.config import WANDB_ENTITY
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import (
    LabelledPassage,
    save_labelled_passages_to_jsonl,
)
from knowledge_graph.wandb_helpers import (
    load_labelled_passages_from_wandb_run,
    log_labelled_passages_artifact_to_wandb_run,
)

console = Console()

app = typer.Typer()


def deduplicate_labelled_passages(
    labelled_passages: list[LabelledPassage],
) -> list[LabelledPassage]:
    """Remove duplicate labelled passages based on text content."""
    seen_texts = set()
    deduplicated_passages = []

    for passage in labelled_passages:
        if passage.text not in seen_texts:
            seen_texts.add(passage.text)
            deduplicated_passages.append(passage)

    return deduplicated_passages


@app.command()
def main(
    wikibase_id: Annotated[
        WikibaseID,
        typer.Option(
            ...,
            help="The Wikibase ID of the concept classifier to run",
            parser=WikibaseID,
        ),
    ],
    labelled_passages_path: Annotated[
        Optional[Path],
        typer.Option(
            help="Optional local path to labelled passages .jsonl file.",
            dir_okay=False,
            exists=True,
        ),
    ],
    labelled_passages_wandb_run_name: Annotated[
        Optional[str],
        typer.Option(
            help="""Optional W&B run name to look for a labelled passages artifact in.
            
            Will look for an artifact of type `labelled-passages` in the project 
            <wikibase_id>.
            """
        ),
    ],
    classifier_wandb_path: Annotated[
        str,
        typer.Option(
            help="Path of the classifier in W&B. E.g. 'climatepolicyradar/Q913/rsgz5ygh:v0'"
        ),
    ],
    track_and_upload: Annotated[
        bool,
        typer.Option(
            ...,
            help="Whether to track the training run with Weights & Biases. Includes uploading the model artifact to S3.",
        ),
    ] = True,
    batch_size: int = typer.Option(
        25,
        help="Number of passages to process in each batch",
    ),
    limit: Annotated[
        Optional[int],
        typer.Option(
            ...,
            help="Optionally limit the number of passages predicted on",
        ),
    ] = None,
    deduplicate_inputs: bool = typer.Option(
        True,
        help="Remove duplicate passages based on text content before prediction",
    ),
):
    """
    Load labelled passages from local dir or W&B, and run a classifier on them.

    Saves predicted passages to a local directory. Tracks the run and uploads results
    if `track_and_upload` is set.
    """

    wandb_config = {
        "batch_size": batch_size,
        "limit": limit,
        "classifier_path": classifier_wandb_path,
        "labelled_passages_path": labelled_passages_path,
        "labelled_passages_wandb_run_name": labelled_passages_wandb_run_name,
    }
    wandb_job_type = "predict_adhoc"

    with (
        wandb.init(
            entity=WANDB_ENTITY,
            project=wikibase_id,
            # TODO: is there a better name to separate document-level inference from
            # adhoc prediction?
            job_type=wandb_job_type,
            config=wandb_config,
        )
        if track_and_upload
        else nullcontext()
    ) as run:
        wandb_api = wandb.Api()

        # 1. load labelled passages
        match (labelled_passages_path, labelled_passages_wandb_run_name):
            case (local_path, wandb_run_name) if local_path and wandb_run_name:
                raise ValueError(
                    "Both `labelled_passages_path` and `labelled_passages_run_name` cannot be defined."
                )

            case (local_path, wandb_run_name) if local_path:
                labelled_passages = LabelledPassage.from_jsonl(local_path)

            case (local_path, wandb_run_name) if wandb_run_name:
                wandb_run = wandb_api.run(
                    f"{WANDB_ENTITY}/{wikibase_id}/{wandb_run_name}"
                )
                labelled_passages = load_labelled_passages_from_wandb_run(wandb_run)

            case _:
                raise ValueError(
                    "One of `labelled_passages_path` and `labelled_passages_run_name` must be defined."
                )

        if deduplicate_inputs:
            original_count = len(labelled_passages)
            labelled_passages = deduplicate_labelled_passages(labelled_passages)
            deduplicated_count = len(labelled_passages)
            console.print(
                f"Deduplicated {original_count} passages to {deduplicated_count} based on their text field"
                f"(removed {original_count - deduplicated_count} duplicates)"
            )

        if limit:
            labelled_passages = labelled_passages[:limit]

        # 2. load model
        classifier = Classifier.load_from_wandb(classifier_wandb_path)

        # 3. predict using model
        input_texts = [lp.text for lp in labelled_passages]
        model_predicted_spans = classifier.predict_many(
            input_texts,
            batch_size=batch_size,
            show_progress=True,
        )
        output_labelled_passages = [
            labelled_passage.model_copy(
                update={"spans": model_predicted_spans[idx]},
                deep=True,
            )
            for idx, labelled_passage in enumerate(labelled_passages)
        ]

        # 4. save to local (and wandb)
        save_labelled_passages_to_jsonl(
            labelled_passages=output_labelled_passages,
            wikibase_id=wikibase_id,
            classifier_id=classifier.id,
            include_wandb_run_name=run.name if run else None,
        )

        if track_and_upload and run:
            log_labelled_passages_artifact_to_wandb_run(
                output_labelled_passages, run=run, concept=classifier.concept
            )


if __name__ == "__main__":
    app()
