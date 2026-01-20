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
from knowledge_graph.cloud import AwsEnv, get_s3_client
from knowledge_graph.config import WANDB_ENTITY, predictions_dir
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import (
    LabelledPassage,
)
from knowledge_graph.labelling import label_passages_with_classifier
from knowledge_graph.wandb_helpers import (
    _load_labelled_passages_from_artifact_dir,
    load_classifier_from_wandb,
    load_labelled_passages_from_wandb,
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
    exclude_training_data: bool = typer.Option(
        True,
        help="Exclude passages that were in the model's training data from prediction",
    ),
    prediction_threshold: float | None = typer.Option(
        None, help="Optional prediction threshold for the classifier."
    ),
    stop_after_n_positives: Annotated[
        Optional[int],
        typer.Option(
            help="Stop prediction after finding this many positive passages",
        ),
    ] = None,
    restart_from_wandb_run: Annotated[
        Optional[str],
        typer.Option(
            help="Optional W&B run path to restart from. Loads already-predicted passages from this run and skips them.",
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
        "prediction_threshold": prediction_threshold,
        "stop_after_n_positives": stop_after_n_positives,
        "exclude_training_data": exclude_training_data,
        "restart_from_wandb_run": restart_from_wandb_run,
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
            labelled_passages = load_labelled_passages_from_wandb(run=wandb_run)
        else:
            raise ValueError(
                "One of `labelled_passages_path` and `labelled_passages_run_name` must be defined."
            )

        already_predicted_passages: list[LabelledPassage] = []
        if restart_from_wandb_run:
            console.print(
                f"Loading already-predicted passages from {restart_from_wandb_run} to skip..."
            )
            try:
                restart_run = wandb_api.run(restart_from_wandb_run)
                already_predicted_passages = load_labelled_passages_from_wandb(
                    run=restart_run
                )
                console.print(
                    f"[green]✓ Loaded {len(already_predicted_passages)} already-predicted passages[/green]"
                )

                # Filter out already-predicted passages based on ID
                already_predicted_ids = {p.id for p in already_predicted_passages}
                len_before = len(labelled_passages)
                labelled_passages = [
                    p for p in labelled_passages if p.id not in already_predicted_ids
                ]
                num_skipped = len_before - len(labelled_passages)
                console.print(
                    f"Skipped {num_skipped} already-predicted passages. {len(labelled_passages)} remaining to predict."
                )
            except Exception as e:
                console.print(
                    f"[red]⚠ Could not load already-predicted passages: {e}[/red]"
                )
                console.print("Continuing without skipping any passages")

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
            console.print(f"Limited number of passages to {len(labelled_passages)}")

        # 2. optionally exclude training data
        if exclude_training_data:
            console.print(
                "Fetching training data from classifier's W&B run to exclude from prediction..."
            )
            try:
                classifier_artifact = wandb_api.artifact(classifier_wandb_path)

                if classifier_run := classifier_artifact.logged_by():
                    if training_artifacts := [
                        a
                        for a in classifier_run.logged_artifacts()
                        if a.type == "labelled_passages" and "training-data" in a.name
                    ]:
                        artifact_dir = Path(training_artifacts[0].download())
                        training_data = _load_labelled_passages_from_artifact_dir(
                            artifact_dir
                        )

                        console.print(
                            f"✓ Loaded {len(training_data)} passages from training data artifact"
                        )

                        training_data_text = {p.text for p in training_data}

                        len_labelled_passages_before = len(labelled_passages)

                        labelled_passages = [
                            p
                            for p in labelled_passages
                            if p.text not in training_data_text
                        ]

                        num_labelled_passages_removed = (
                            len_labelled_passages_before - len(labelled_passages)
                        )
                        console.print(
                            f"Removed {num_labelled_passages_removed} passages from labelled passages dataset. {len(labelled_passages)} remaining."
                        )

                    else:
                        console.print(
                            "⚠ No training-data artifact found in classifier's run, skipping exclusion"
                        )
            except Exception as e:
                console.print(
                    f"⚠ Could not load training data: {e}\nContinuing with prediction without excluding training data"
                )

        # 3. load model
        region_name = "eu-west-1"
        aws_env = AwsEnv.labs
        # When running in prefect the client is instantiated earlier
        # Set this, so W&B knows where to look for AWS credentials profile
        os.environ["AWS_PROFILE"] = aws_env
        get_s3_client(aws_env, region_name)

        classifier = load_classifier_from_wandb(classifier_wandb_path)

        if prediction_threshold is not None:
            classifier.set_prediction_threshold(prediction_threshold)
            console.print(
                f"Classifier prediction threshold set to {prediction_threshold}"
            )

        # 4. predict using model
        output_labelled_passages: list[LabelledPassage] = []
        prediction_exception: Exception | None = None

        try:
            # Process batch-by-batch to save partial results on failure
            positives_found = 0
            passages_processed = 0

            if stop_after_n_positives is not None:
                console.print(
                    f"[cyan]Early stopping enabled: will stop after finding {stop_after_n_positives} positive passages[/cyan]"
                )

            console.print(
                "[cyan]You can end prediction early by pressing Ctrl+C. This will save passages predicted thus far.[/cyan]"
            )

            for i in range(0, len(labelled_passages), batch_size):
                batch = labelled_passages[i : i + batch_size]

                batch_output = label_passages_with_classifier(
                    classifier=classifier,
                    labelled_passages=batch,
                    batch_size=batch_size,
                    show_progress=True,
                )

                passages_processed += len(batch)
                output_labelled_passages.extend(batch_output)

                # Track positives if early stopping is enabled
                if stop_after_n_positives is not None:
                    batch_positives = sum(1 for p in batch_output if len(p.spans) > 0)
                    positives_found += batch_positives

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
                else:
                    console.print(
                        f"[cyan]Processed {passages_processed}/{len(labelled_passages)} passages[/cyan]"
                    )

        except Exception as e:
            prediction_exception = e
            console.print(
                f"[red]⚠ Prediction failed: {e}[/red]\n"
                f"[yellow]Saving {len(output_labelled_passages)} partial results...[/yellow]"
            )

        finally:
            if output_labelled_passages or already_predicted_passages:
                all_passages = already_predicted_passages + output_labelled_passages

                labelled_passages_filename = (
                    f"{classifier.id}_{run.name}.jsonl"
                    if run
                    else f"{classifier.id}.jsonl"
                )
                labelled_passages_path = (
                    predictions_dir / wikibase_id / labelled_passages_filename
                )
                labelled_passages_jsonl = serialise_pydantic_list_as_jsonl(all_passages)

                labelled_passages_path.parent.mkdir(parents=True, exist_ok=True)
                labelled_passages_path.write_text(labelled_passages_jsonl)
                console.print(
                    f"[green]✓ Saved {len(all_passages)} passages to {labelled_passages_path}[/green]"
                )

                if track_and_upload and run:
                    log_labelled_passages_artifact_to_wandb_run(
                        all_passages, run=run, concept=classifier.concept
                    )
                    console.print(
                        f"[green]✓ Uploaded passages to W&B run {run.name}[/green]"
                    )

        # Re-raise the exception after saving
        if prediction_exception:
            raise prediction_exception


if __name__ == "__main__":
    app()
