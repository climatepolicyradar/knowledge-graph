import os
from contextlib import nullcontext
from pathlib import Path
from typing import Annotated, Optional

import typer
import wandb
from dotenv import load_dotenv
from rich.console import Console

from flows.utils import (
    deserialise_pydantic_list_with_fallback,
    serialise_pydantic_list_as_jsonl,
)
from knowledge_graph.classifier import load_classifier_from_wandb
from knowledge_graph.cloud import AwsEnv, get_s3_client
from knowledge_graph.config import WANDB_ENTITY, predictions_dir
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import (
    LabelledPassage,
)
from knowledge_graph.labelling import label_passages_with_classifier
from knowledge_graph.wandb_helpers import (
    load_labelled_passages_from_wandb_run,
    log_labelled_passages_artifact_to_wandb_run,
)

console = Console()

app = typer.Typer()

load_dotenv()


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
    classifier_wandb_path: Annotated[
        str,
        typer.Option(
            help="Path of the classifier in W&B. E.g. 'climatepolicyradar/Q913/rsgz5ygh:v0'"
        ),
    ],
    labelled_passages_path: Annotated[
        Optional[Path],
        typer.Option(
            help="Optional local path to labelled passages .jsonl file.",
            dir_okay=False,
            exists=True,
        ),
    ] = None,
    labelled_passages_wandb_run_path: Annotated[
        Optional[str],
        typer.Option(
            help="""Optional W&B run name to look for a labelled passages artifact in.
            
            Will look for an artifact of type `labelled-passages` in the project 
            <wikibase_id>.
            """
        ),
    ] = None,
    track_and_upload: Annotated[
        bool,
        typer.Option(
            ...,
            help="Whether to track the training run with Weights & Biases. Includes uploading the model artifact to S3.",
        ),
    ] = True,
    batch_size: int = typer.Option(
        15,
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
    stop_after_n_positives: Annotated[
        Optional[int],
        typer.Option(
            help="Stop prediction after finding this many positive passages",
        ),
    ] = None,
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
        "labelled_passages_wandb_run_path": labelled_passages_wandb_run_path,
        "stop_after_n_positives": stop_after_n_positives,
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
        if labelled_passages_path and labelled_passages_wandb_run_path:
            raise ValueError(
                "Both `labelled_passages_path` and `labelled_passages_run_name` cannot be defined."
            )
        elif labelled_passages_path:
            labelled_passages: list[LabelledPassage] = (
                deserialise_pydantic_list_with_fallback(
                    content=labelled_passages_path.read_text(),
                    model_class=LabelledPassage,
                )
            )
        elif labelled_passages_wandb_run_path:
            wandb_run = wandb_api.run(labelled_passages_wandb_run_path)
            labelled_passages = load_labelled_passages_from_wandb_run(wandb_run)
        else:
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
        region_name = "eu-west-1"
        aws_env = AwsEnv.labs
        # When running in prefect the client is instantiated earlier
        # Set this, so W&B knows where to look for AWS credentials profile
        os.environ["AWS_PROFILE"] = aws_env
        get_s3_client(aws_env, region_name)

        classifier = load_classifier_from_wandb(classifier_wandb_path)

        # 3. predict using model
        if stop_after_n_positives is None:
            output_labelled_passages = label_passages_with_classifier(
                classifier=classifier,
                labelled_passages=labelled_passages,
                batch_size=batch_size,
                show_progress=True,
            )
        else:
            # Early stopping: process batch-by-batch until we have enough positives
            output_labelled_passages = []
            positives_found = 0
            passages_processed = 0

            console.print(
                f"[cyan]Early stopping enabled: will stop after finding {stop_after_n_positives} positive passages[/cyan]"
            )

            for i in range(0, len(labelled_passages), batch_size):
                batch = labelled_passages[i : i + batch_size]

                batch_output = label_passages_with_classifier(
                    classifier=classifier,
                    labelled_passages=batch,
                    batch_size=batch_size,
                    show_progress=True,
                )

                batch_positives = sum(1 for p in batch_output if len(p.spans) > 0)
                positives_found += batch_positives
                passages_processed += len(batch)

                output_labelled_passages.extend(batch_output)

                console.print(
                    f"[cyan]Processed {passages_processed}/{len(labelled_passages)} passages, "
                    f"found {positives_found} positives ({batch_positives} in batch)[/cyan]"
                )

                if positives_found >= stop_after_n_positives:
                    console.print(
                        f"[green]✓ Reached target of {stop_after_n_positives} positives. "
                        f"Stopping early (skipped {len(labelled_passages) - passages_processed} passages)[/green]"
                    )
                    break

        # 4. save to local (and wandb)
        labelled_passages_filename = (
            f"{classifier.id}_{run.name}.jsonl" if run else f"{classifier.id}.jsonl"
        )
        labelled_passages_path = (
            predictions_dir / wikibase_id / labelled_passages_filename
        )
        labelled_passages_jsonl = serialise_pydantic_list_as_jsonl(
            output_labelled_passages
        )
        Path(labelled_passages_filename).write_text(labelled_passages_jsonl)

        if track_and_upload and run:
            log_labelled_passages_artifact_to_wandb_run(
                output_labelled_passages, run=run, concept=classifier.concept
            )


if __name__ == "__main__":
    app()
