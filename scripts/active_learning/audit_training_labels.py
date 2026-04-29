from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
import wandb
from dotenv import load_dotenv
from rich.console import Console

from knowledge_graph.config import WANDB_ENTITY
from knowledge_graph.ensemble import create_ensemble
from knowledge_graph.ensemble.metrics import Disagreement, MajorityVote
from knowledge_graph.wandb_helpers import (
    load_classifier_from_wandb,
    load_classifier_training_data_from_wandb,
)

app = typer.Typer()
load_dotenv()

MODEL_CONFIDENT_THRESHOLD = 0.5

FLAG_LIKELY_MISLABEL = "likely_mislabel"
FLAG_AMBIGUOUS = "ambiguous"
FLAG_OK = "ok"


@app.command()
def main(
    classifier_wandb_path: Annotated[
        str,
        typer.Option(
            ...,
            help="Path to classifier in W&B (e.g., 'climatepolicyradar/Q913/rsgz5ygh:v0')",
        ),
    ],
    n_classifiers: Annotated[
        int,
        typer.Option(help="Number of variants in the ensemble (incl. the main model)"),
    ] = 5,
    batch_size: Annotated[
        int,
        typer.Option(help="Batch size for ensemble inference"),
    ] = 50,
    track_and_upload: Annotated[
        bool,
        typer.Option(
            help="Whether to track the run with W&B and upload the audit table/CSV",
        ),
    ] = True,
):
    """
    Audit a classifier's own training labels for likely mislabels.

    Loads a trained classifier and its `training-data` artifact from the classifier's
    W&B run, builds an n-variant ensemble (Monte-Carlo dropout for BERT classifiers),
    and predicts on every training passage. For each passage we compare the ensemble's
    majority vote with the given training label and compute the ensemble's internal
    disagreement.

    Each passage is flagged as one of:
    - `likely_mislabel`: ensemble is confident (low disagreement) and its majority vote
        contradicts the given label. Highest-priority review candidates.
    - `ambiguous`: ensemble is split (high disagreement). May indicate a
      noisy label or a passage that doesn't fit the labelling guidelines cleanly.
    - `ok`: ensemble is confident and agrees with the given label. No review needed.

    Outputs a CSV (sorted: `likely_mislabel` by ascending disagreement first, then
    `ambiguous` by descending disagreement) and logs a filterable W&B Table plus a
    downloadable CSV artifact.
    """
    console = Console()

    wandb_config = {
        "classifier_wandb_path": classifier_wandb_path,
        "n_classifiers": n_classifiers,
        "batch_size": batch_size,
        "model_confident_threshold": MODEL_CONFIDENT_THRESHOLD,
    }

    # Load classifier
    console.print(f"Loading classifier from W&B: {classifier_wandb_path}")
    classifier = load_classifier_from_wandb(classifier_wandb_path)
    if not classifier.is_fitted:
        raise ValueError("Classifier must be fitted to audit its training labels.")

    wikibase_id = classifier.concept.wikibase_id
    console.print(f"Loaded {classifier} for concept {wikibase_id}")

    with (
        wandb.init(
            entity=WANDB_ENTITY,
            project=wikibase_id,
            job_type="audit_training_labels",
            config=wandb_config,
        )
        if track_and_upload
        else nullcontext()
    ) as run:
        # Load the training-data artifact from the classifier's creating run
        console.print("Loading training data from classifier's creating run...")
        training_data, training_data_artifact = (
            load_classifier_training_data_from_wandb(classifier_wandb_path)
        )
        console.print(
            f"Loaded {len(training_data)} training passages from "
            f"{training_data_artifact.name}"
        )

        if track_and_upload and run:
            run.use_artifact(classifier_wandb_path)
            run.use_artifact(training_data_artifact)

        # Build ensemble and predict
        console.print(
            f"Creating ensemble with {n_classifiers} variants from {classifier}"
        )
        ensemble = create_ensemble(
            classifier=classifier,
            n_classifiers=n_classifiers,
        )

        console.print(f"Running ensemble inference on {len(training_data)} passages...")
        texts = [p.text for p in training_data]
        ensemble_predicted_spans = ensemble.predict(
            texts,
            batch_size=batch_size,
            show_progress=True,
        )

        # Compute per-passage metrics and flags
        disagreement_metric = Disagreement()
        majority_vote_metric = MajorityVote()

        rows: list[dict] = []
        for passage, spans_per_classifier in zip(
            training_data, ensemble_predicted_spans
        ):
            given_label = (
                1
                if any(span.concept_id == wikibase_id for span in passage.spans)
                else 0
            )
            disagreement = float(disagreement_metric(spans_per_classifier))
            majority_vote = float(majority_vote_metric(spans_per_classifier))
            majority_label = int(round(majority_vote))

            if disagreement >= MODEL_CONFIDENT_THRESHOLD:
                flag_type = FLAG_AMBIGUOUS
            elif majority_label != given_label:
                flag_type = FLAG_LIKELY_MISLABEL
            else:
                flag_type = FLAG_OK

            rows.append(
                {
                    "flag_type": flag_type,
                    "given_label": given_label,
                    "ensemble_majority_vote": majority_label,
                    "ensemble_disagreement": disagreement,
                    "text": passage.text,
                }
            )

        df = pd.DataFrame(rows)

        # Sort: likely_mislabel first, then ambiguous, then ok
        flag_priority = {FLAG_LIKELY_MISLABEL: 0, FLAG_AMBIGUOUS: 1, FLAG_OK: 2}
        df["_priority"] = df["flag_type"].map(flag_priority.get)
        df["_within_flag_sort"] = df.apply(
            lambda r: r["ensemble_disagreement"]
            if r["flag_type"] == FLAG_LIKELY_MISLABEL
            else -r["ensemble_disagreement"]
            if r["flag_type"] == FLAG_AMBIGUOUS
            else 0.0,
            axis=1,
        )
        df = df.sort_values(["_priority", "_within_flag_sort"]).drop(
            columns=["_priority", "_within_flag_sort"]
        )

        counts = df["flag_type"].value_counts().to_dict()
        n_likely_mislabel = int(counts.get(FLAG_LIKELY_MISLABEL, 0))
        n_ambiguous = int(counts.get(FLAG_AMBIGUOUS, 0))
        n_ok = int(counts.get(FLAG_OK, 0))
        console.print(
            f"Audit summary: {n_likely_mislabel} likely_mislabel, "
            f"{n_ambiguous} ambiguous, {n_ok} ok (total {len(df)})"
        )

        # Save CSV locally
        timestr = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(
            f"data/processed/audit_training_labels/{wikibase_id}_{timestr}"
        )
        output_dir.mkdir(exist_ok=True, parents=True)
        csv_path = output_dir / f"audit_{wikibase_id}.csv"
        df.to_csv(csv_path, index=False)
        console.print(f"✅ Saved audit CSV to {csv_path}")

        # Log to W&B: a Table for in-browser inspection + a CSV artifact for download
        if track_and_upload and run:
            flagged = df[df["flag_type"] != FLAG_OK]
            flagged_table = wandb.Table(dataframe=flagged.reset_index(drop=True))
            run.log({"flagged_passages": flagged_table})

            artifact = wandb.Artifact(
                name=f"audit-training-labels-{wikibase_id}",
                type="audit_results",
                description=(
                    "Per-passage audit of a classifier's training labels, comparing "
                    "ensemble majority vote vs given label."
                ),
            )
            artifact.add_file(str(csv_path))
            run.log_artifact(artifact)

            run.summary["num_likely_mislabel"] = n_likely_mislabel
            run.summary["num_ambiguous"] = n_ambiguous
            run.summary["num_ok"] = n_ok
            run.summary["num_total_passages"] = len(df)
            run.summary["pct_flagged"] = (
                round((n_likely_mislabel + n_ambiguous) / len(df) * 100, 2)
                if len(df)
                else 0.0
            )

            console.print(
                "✅ Logged flagged_passages table and CSV artifact to W&B run "
                f"({run.url})"
            )


if __name__ == "__main__":
    app()
