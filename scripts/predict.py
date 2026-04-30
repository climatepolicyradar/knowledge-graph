import asyncio
import json
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Annotated

import snowflake.connector
import typer
import wandb
from dotenv import load_dotenv

from knowledge_graph.cloud import AwsEnv, get_s3_client
from knowledge_graph.config import WANDB_ENTITY, predictions_dir
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.labelling import label_passages_with_classifier
from knowledge_graph.utils import (
    deserialise_pydantic_list_with_fallback,
    get_logger,
    serialise_pydantic_list_as_jsonl,
)
from knowledge_graph.wandb_helpers import (
    _load_labelled_passages_from_artifact_dir,
    load_classifier_from_wandb,
    load_labelled_passages_from_wandb,
    log_labelled_passages_artifact_to_wandb_run,
    log_labelled_passages_table_to_wandb_run,
)

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


def load_passages_from_snowflake(
    document_ids: list[str], minimum_text_chars: int = 0
) -> list[LabelledPassage]:
    """Load English passages from Snowflake for the given document IDs."""
    logger = get_logger()
    logger.info(
        f"Connecting to Snowflake to load passages for {len(document_ids)} document(s)"
    )

    con = snowflake.connector.connect(connection_name="local_dev")
    cur = con.cursor()

    placeholders = ", ".join(["%s"] * len(document_ids))
    query = f"""
    SELECT
        p.CONTENT AS text_block_text,
        p.content_type AS text_block_type,
        d.DOCUMENT_ID,
        d.content_type AS document_content_type,
        d.document_name AS document_name,
        d.document_slug AS document_slug,
        d.TRANSLATED AS document_metadata_translated,
        d.METADATA_CORPUS_TYPE_NAME AS document_metadata_corpus_type_name,
        d.METADATA_GEOGRAPHIES AS document_metadata_geographies
    FROM PRODUCTION.PUBLISHED.PIPELINE_DOCUMENTS_V1 d
    JOIN PRODUCTION.PUBLISHED.PIPELINE_PASSAGES_V2 p
        ON d.DOCUMENT_ID = p.DOCUMENT_ID
    WHERE p.LANGUAGE = 'en'
      AND p.CONTENT IS NOT NULL
      AND LENGTH(p.CONTENT) > {minimum_text_chars}
      AND d.DOCUMENT_ID IN ({placeholders})
    """

    cur.execute(query, document_ids)
    df = cur.fetch_pandas_all()
    con.close()

    logger.info(f"✓ Loaded {len(df)} passages from Snowflake")

    rename_cols = {
        "TEXT_BLOCK_TEXT": "text_block.text",
        "TEXT_BLOCK_TYPE": "text_block.type",
        "DOCUMENT_ID": "document_id",
        "DOCUMENT_CONTENT_TYPE": "document_content_type",
        "DOCUMENT_NAME": "document_name",
        "DOCUMENT_SLUG": "document_slug",
        "DOCUMENT_METADATA_TRANSLATED": "document_metadata.translated",
        "DOCUMENT_METADATA_CORPUS_TYPE_NAME": "document_metadata.corpus_type_name",
        "DOCUMENT_METADATA_GEOGRAPHIES": "document_metadata.geographies",
    }
    df = df.rename(columns=rename_cols)

    df["document_metadata.geographies"] = df["document_metadata.geographies"].apply(
        lambda x: json.loads(x) if isinstance(x, str) else x
    )

    labelled_passages = []
    for _, row in df.iterrows():
        metadata = row.to_dict()
        metadata.pop("text_block.text")
        for key, value in metadata.items():
            if hasattr(value, "tolist"):
                metadata[key] = value.tolist()
        labelled_passages.append(
            LabelledPassage(
                text=str(row["text_block.text"]),
                metadata=metadata,
                spans=[],
            )
        )

    return labelled_passages


async def run_prediction(
    wikibase_id: WikibaseID,
    classifier_wandb_path: str,
    labelled_passages_path: Path | None = None,
    labelled_passages_wandb_run_path: str | None = None,
    input_passages: list[LabelledPassage] | None = None,
    track_and_upload: bool = True,
    batch_size: int = 15,
    limit: int | None = None,
    deduplicate_inputs: bool = True,
    exclude_training_data: bool = True,
    prediction_threshold: float | None = None,
    stop_after_n_positives: int | None = None,
    restart_from_wandb_run: str | None = None,
    aws_env: AwsEnv = AwsEnv.production,
) -> None:
    """
    Load labelled passages from local dir or W&B, and run a classifier on them.

    Saves predicted passages to a local directory. Tracks the run and uploads results
    if `track_and_upload` is set.
    """

    logger = get_logger()

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
        elif input_passages is not None:
            labelled_passages: list[LabelledPassage] = input_passages
        elif labelled_passages_path:
            labelled_passages = deserialise_pydantic_list_with_fallback(
                content=labelled_passages_path.read_text(),
                model_class=LabelledPassage,
            )
        elif labelled_passages_wandb_run_path:
            wandb_run = wandb_api.run(labelled_passages_wandb_run_path)
            labelled_passages = load_labelled_passages_from_wandb(run=wandb_run)
        else:
            raise ValueError(
                "One of `labelled_passages_path`, `labelled_passages_wandb_run_path`, or `input_passages` must be provided."
            )

        already_predicted_passages: list[LabelledPassage] = []
        if restart_from_wandb_run:
            logger.info(
                f"Loading already-predicted passages from {restart_from_wandb_run} to skip..."
            )
            try:
                restart_run = wandb_api.run(restart_from_wandb_run)
                already_predicted_passages = load_labelled_passages_from_wandb(
                    run=restart_run
                )
                logger.info(
                    f"✓ Loaded {len(already_predicted_passages)} already-predicted passages"
                )

                # Filter out already-predicted passages based on ID
                already_predicted_ids = {p.id for p in already_predicted_passages}
                len_before = len(labelled_passages)
                labelled_passages = [
                    p for p in labelled_passages if p.id not in already_predicted_ids
                ]
                num_skipped = len_before - len(labelled_passages)
                logger.info(
                    f"Skipped {num_skipped} already-predicted passages. {len(labelled_passages)} remaining to predict."
                )
            except Exception as e:
                logger.warning(f"⚠ Could not load already-predicted passages: {e}")
                logger.info("Continuing without skipping any passages")

        if deduplicate_inputs:
            original_count = len(labelled_passages)
            labelled_passages = deduplicate_labelled_passages(labelled_passages)
            deduplicated_count = len(labelled_passages)
            logger.info(
                f"Deduplicated {original_count} passages to {deduplicated_count} based on their text field"
                f"(removed {original_count - deduplicated_count} duplicates)"
            )

        if limit:
            labelled_passages = labelled_passages[:limit]
            logger.info(f"Limited number of passages to {len(labelled_passages)}")

        # 2. optionally exclude training data
        if exclude_training_data:
            logger.info(
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

                        logger.info(
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
                        logger.info(
                            f"Removed {num_labelled_passages_removed} passages from labelled passages dataset. {len(labelled_passages)} remaining."
                        )

                    else:
                        logger.warning(
                            "⚠ No training-data artifact found in classifier's run, skipping exclusion"
                        )
            except Exception as e:
                logger.warning(
                    f"⚠ Could not load training data: {e}\nContinuing with prediction without excluding training data"
                )

        # 3. load model
        region_name = "eu-west-1"
        # When running in prefect the client is instantiated earlier
        # Set this, so W&B knows where to look for AWS credentials profile
        os.environ["AWS_PROFILE"] = aws_env
        get_s3_client(aws_env, region_name)

        classifier = load_classifier_from_wandb(classifier_wandb_path)

        if prediction_threshold is not None:
            classifier.set_prediction_threshold(prediction_threshold)
            logger.info(
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
                logger.info(
                    f"Early stopping enabled: will stop after finding {stop_after_n_positives} positive passages"
                )

            logger.info(
                "You can end prediction early by pressing Ctrl+C. This will save passages predicted thus far."
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

                    logger.info(
                        f"Processed {passages_processed}/{len(labelled_passages)} passages, "
                        f"found {positives_found} positives ({batch_positives} in batch)"
                    )

                    if positives_found >= stop_after_n_positives:
                        logger.info(
                            f"✓ Reached target of {stop_after_n_positives} positives. "
                            f"Stopping early (skipped {len(labelled_passages) - passages_processed} passages)"
                        )
                        break
                else:
                    logger.info(
                        f"Processed {passages_processed}/{len(labelled_passages)} passages"
                    )

        except Exception as e:
            prediction_exception = e
            logger.error(f"⚠ Prediction failed: {e}")
            logger.info(f"Saving {len(output_labelled_passages)} partial results...")

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
                logger.info(
                    f"✓ Saved {len(all_passages)} passages to {labelled_passages_path}"
                )

                if track_and_upload and run:
                    log_labelled_passages_artifact_to_wandb_run(
                        all_passages, run=run, concept=classifier.concept
                    )
                    log_labelled_passages_table_to_wandb_run(all_passages, run=run)
                    logger.info(f"✓ Uploaded passages to W&B run {run.name}")

        # Re-raise the exception after saving
        if prediction_exception:
            raise prediction_exception


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
        Path | None,
        typer.Option(
            help="Optional local path to labelled passages .jsonl file.",
            dir_okay=False,
            exists=True,
        ),
    ] = None,
    labelled_passages_wandb_run_path: Annotated[
        str | None,
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
        int | None,
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
        int | None,
        typer.Option(
            help="Stop prediction after finding this many positive passages",
        ),
    ] = None,
    restart_from_wandb_run: Annotated[
        str | None,
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
    asyncio.run(
        run_prediction(
            wikibase_id=wikibase_id,
            classifier_wandb_path=classifier_wandb_path,
            labelled_passages_path=labelled_passages_path,
            labelled_passages_wandb_run_path=labelled_passages_wandb_run_path,
            track_and_upload=track_and_upload,
            batch_size=batch_size,
            limit=limit,
            deduplicate_inputs=deduplicate_inputs,
            exclude_training_data=exclude_training_data,
            prediction_threshold=prediction_threshold,
            stop_after_n_positives=stop_after_n_positives,
            restart_from_wandb_run=restart_from_wandb_run,
        )
    )


@app.command()
def documents(
    document_ids: Annotated[
        list[str],
        typer.Argument(help="One or more document IDs to load passages from Snowflake"),
    ],
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
    track_and_upload: Annotated[
        bool,
        typer.Option(
            ...,
            help="Whether to track the run with Weights & Biases and upload results.",
        ),
    ] = True,
    batch_size: int = typer.Option(
        15,
        help="Number of passages to process in each batch",
    ),
    limit: Annotated[
        int | None,
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
        int | None,
        typer.Option(
            help="Stop prediction after finding this many positive passages",
        ),
    ] = None,
    restart_from_wandb_run: Annotated[
        str | None,
        typer.Option(
            help="Optional W&B run path to restart from. Loads already-predicted passages from this run and skips them.",
        ),
    ] = None,
):
    """
    Load passages for specific document IDs from Snowflake and run a classifier on them.

    Saves predicted passages to a local directory. Tracks the run and uploads results
    if `track_and_upload` is set.
    """
    passages = load_passages_from_snowflake(document_ids)
    asyncio.run(
        run_prediction(
            wikibase_id=wikibase_id,
            classifier_wandb_path=classifier_wandb_path,
            input_passages=passages,
            track_and_upload=track_and_upload,
            batch_size=batch_size,
            limit=limit,
            deduplicate_inputs=deduplicate_inputs,
            exclude_training_data=exclude_training_data,
            prediction_threshold=prediction_threshold,
            stop_after_n_positives=stop_after_n_positives,
            restart_from_wandb_run=restart_from_wandb_run,
        )
    )


if __name__ == "__main__":
    app()
