from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
import wandb
from dotenv import load_dotenv
from pydantic import BaseModel
from rich.console import Console

import knowledge_graph.ensemble.metrics as ensemble_metrics
from flows.utils import (
    deserialise_pydantic_list_with_fallback,
    serialise_pydantic_list_as_jsonl,
)
from knowledge_graph.classifier import Classifier, load_classifier_from_wandb
from knowledge_graph.config import WANDB_ENTITY
from knowledge_graph.ensemble import create_ensemble
from knowledge_graph.ensemble.metrics import EnsembleMetric, MajorityVote
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.span import Span, UnitInterval
from knowledge_graph.wandb_helpers import (
    load_labelled_passages_from_wandb,
    log_labelled_passages_artifact_to_wandb_run,
)

app = typer.Typer()

load_dotenv()


class EnsembleConfig(BaseModel):
    """An Ensemble, and a metric and threshold to use for escalating predictions."""

    model_config = {"arbitrary_types_allowed": True}

    n_classifiers: int
    ensemble_metric: EnsembleMetric
    ensemble_metric_threshold: UnitInterval


ensemble_config_bert = EnsembleConfig(
    n_classifiers=5,
    ensemble_metric=ensemble_metrics.Disagreement(),
    ensemble_metric_threshold=UnitInterval(0.0),
)

ensemble_config_llm = EnsembleConfig(
    n_classifiers=5,
    ensemble_metric=ensemble_metrics.Disagreement(),
    ensemble_metric_threshold=UnitInterval(0.0),
)


def annotate_passages_with_ensemble(
    passages: list[LabelledPassage],
    wikibase_id: WikibaseID,
    model: Classifier,
    ensemble_config: EnsembleConfig,
    batch_size: int = 50,
) -> tuple[list[LabelledPassage], list[LabelledPassage]]:
    """
    Annotate passages using an ensemble of classifiers.

    :param wikibase_id: The ID of the concept being classified
    :param passages: Passages to annotate
    :param model: The base classifier to create ensemble from
    :param ensemble_config: Configuration for ensemble and thresholding
    :param batch_size: Number of passages to process in each batch
    :return: Tuple of (annotated_passages, unannotated_passages) where annotated_passages
        contains passages with metric <= threshold (confident predictions) and
        uncertain_passages contains the rest of the passages
    """
    console = Console()

    console.print(
        f"Creating ensemble with {ensemble_config.n_classifiers} variants from base classifier {model}"
    )
    ensemble = create_ensemble(
        classifier=model,
        n_classifiers=ensemble_config.n_classifiers,
    )

    # predict all passages using ensemble
    console.print(f"Predicting on {len(passages)} passages...")
    text_to_predict = [passage.text for passage in passages]
    ensemble_predicted_spans = ensemble.predict(
        text_to_predict,
        batch_size=batch_size,
        show_progress=True,
    )

    # get uncertainties
    ensemble_metrics = [
        ensemble_config.ensemble_metric(spans) for spans in ensemble_predicted_spans
    ]
    majority_vote_metric = MajorityVote()
    ensemble_majority_votes = [
        majority_vote_metric(spans) for spans in ensemble_predicted_spans
    ]

    # for uncertainties <= threshold, add these passages to annotated_passages
    # and remove them from passages_to_annotate
    annotated_passages: list[LabelledPassage] = []
    unannotated_passages: list[LabelledPassage] = []

    for passage, metric_value, majority_vote_value in zip(
        passages, ensemble_metrics, ensemble_majority_votes
    ):
        if metric_value <= ensemble_config.ensemble_metric_threshold:
            # NOTE: here we use the prediction_probability field of Span to indicate
            # whether a model predicted the mention of a concept in text. This goes
            # against the convention in a lot of this code where a negative prediction
            # on a passage is indicated by empty spans. This is done here because
            # we want to store the model which made the negative prediction in the
            # span's `labellers` field.

            span = Span(
                text=passage.text,
                start_index=0,
                end_index=len(passage.text),
                prediction_probability=majority_vote_value,
                concept_id=wikibase_id,
                labellers=[str(model)],
                timestamps=[datetime.now()],
            )
            annotated_passages.append(passage.model_copy(update={"spans": [span]}))
        else:
            unannotated_passages.append(passage)

    console.print(
        f"Annotated {len(annotated_passages)} passages with metric {ensemble_config.ensemble_metric.name} <= {ensemble_config.ensemble_metric_threshold}. {len(unannotated_passages)} remaining."
    )

    return annotated_passages, unannotated_passages


def run_active_learning(
    wikibase_id: WikibaseID,
    labelled_passages: list[LabelledPassage],
    bert_classifier: Classifier,
    llm_classifier: Classifier,
    ensemble_config_bert: EnsembleConfig,
    ensemble_config_llm: EnsembleConfig,
    batch_size: int = 50,
) -> tuple[list[LabelledPassage], list[LabelledPassage], list[LabelledPassage]]:
    """
    Run prediction using a BERT then LLM ensemble, escalating uncertain predictions.

    :param wikibase_id: The concept being classified
    :param labelled_passages: Passages to annotate
    :param bert_classifier: BERT-based classifier for first ensemble
    :param llm_classifier: LLM-based classifier for second ensemble
    :param ensemble_config_bert: Configuration for BERT ensemble
    :param ensemble_config_llm: Configuration for LLM ensemble
    :param batch_size: Number of passages to process in each batch
    :return: Tuple of (bert_labelled_passages, llm_labelled_passages, unlabelled_passages)
        where bert_labelled_passages are passages labelled by BERT ensemble,
        llm_labelled_passages are passages labelled by LLM ensemble (that BERT was uncertain about),
        and unlabelled_passages are passages both ensembles were uncertain about
    """
    console = Console()

    # Run BERT ensemble annotation
    bert_labelled_passages, passages_after_bert = annotate_passages_with_ensemble(
        passages=labelled_passages,
        wikibase_id=wikibase_id,
        model=bert_classifier,
        ensemble_config=ensemble_config_bert,
        batch_size=batch_size,
    )

    # Run LLM ensemble annotation on passages that BERT was uncertain about
    llm_labelled_passages, unlabelled_passages = annotate_passages_with_ensemble(
        passages=passages_after_bert,
        wikibase_id=wikibase_id,
        model=llm_classifier,
        ensemble_config=ensemble_config_llm,
        batch_size=batch_size,
    )

    total_labelled = len(bert_labelled_passages) + len(llm_labelled_passages)
    percent_labelled = round(total_labelled / len(labelled_passages) * 100, 1)
    percent_unlabelled = round(100 - percent_labelled, 1)

    console.print(
        f"Complete: {total_labelled} ({percent_labelled}%) labelled "
        f"(BERT: {len(bert_labelled_passages)}, LLM: {len(llm_labelled_passages)}), "
        f"{len(unlabelled_passages)} ({percent_unlabelled}%) unlabelled."
    )

    return bert_labelled_passages, llm_labelled_passages, unlabelled_passages


@app.command()
def main(
    wikibase_id: Annotated[
        WikibaseID,
        typer.Option(
            ...,
            help="The Wikibase ID of the concept being classified",
            parser=WikibaseID,
        ),
    ],
    labelled_passages_wandb_path: Annotated[
        str,
        typer.Option(
            ...,
            help="W&B path to load labelled passages from (e.g., 'entity/project/artifact_id:version')",
        ),
    ],
    classifier_wandb_path_bert: Annotated[
        str,
        typer.Option(
            ...,
            help="Path to BERT classifier in W&B (e.g., 'entity/project/artifact_id:version')",
        ),
    ],
    classifier_wandb_path_llm: Annotated[
        str,
        typer.Option(
            ...,
            help="Path to LLM classifier in W&B (e.g., 'entity/project/artifact_id:version')",
        ),
    ],
    batch_size: Annotated[
        int,
        typer.Option(
            help="Number of passages to process in each batch",
        ),
    ] = 50,
    limit: Annotated[
        int | None,
        typer.Option(
            help="Optional limit on the number of passages to annotate",
        ),
    ] = None,
    track_and_upload: Annotated[
        bool,
        typer.Option(
            help="Whether to track the run with Weights & Biases and upload artifacts",
        ),
    ] = True,
):
    """
    Run active learning annotation using ensemble-based uncertainty estimation.

    Loads unlabeled passages and uses two classifier ensembles (BERT and LLM) to identify
    passages where both models are uncertain. These passages are the best candidates for
    human annotation.
    """
    console = Console()

    wandb_config = {
        "batch_size": batch_size,
        "limit": limit,
        "labelled_passages_wandb_path": labelled_passages_wandb_path,
        "classifier_wandb_path_bert": classifier_wandb_path_bert,
        "classifier_wandb_path_llm": classifier_wandb_path_llm,
        "ensemble_config_bert": {
            "n_classifiers": ensemble_config_bert.n_classifiers,
            "ensemble_metric": ensemble_config_bert.ensemble_metric.name,
            "ensemble_metric_threshold": float(
                ensemble_config_bert.ensemble_metric_threshold
            ),
        },
        "ensemble_config_llm": {
            "n_classifiers": ensemble_config_llm.n_classifiers,
            "ensemble_metric": ensemble_config_llm.ensemble_metric.name,
            "ensemble_metric_threshold": float(
                ensemble_config_llm.ensemble_metric_threshold
            ),
        },
    }
    wandb_job_type = "active_learning"

    with (
        wandb.init(
            entity=WANDB_ENTITY,
            project=wikibase_id,
            job_type=wandb_job_type,
            config=wandb_config,
        )
        if track_and_upload
        else nullcontext()
    ) as run:
        # load classifiers
        console.print("Loading classifiers from W&B...")
        bert_classifier = load_classifier_from_wandb(classifier_wandb_path_bert)
        if not bert_classifier._is_fitted:
            raise ValueError(
                "BERT classifier must have been fit before running active learning."
            )

        llm_classifier = load_classifier_from_wandb(classifier_wandb_path_llm)

        # load labelled passages
        console.print(
            f"Loading labelled passages from W&B: {labelled_passages_wandb_path}"
        )
        wandb_api = wandb.Api()
        labelled_passages = load_labelled_passages_from_wandb(
            labelled_passages_wandb_path
        )

        console.print(f"Loaded {len(labelled_passages)} passages")

        if track_and_upload and run:
            # declare models and labelled passages as inputs to this run
            run.use_artifact(classifier_wandb_path_bert)
            run.use_artifact(classifier_wandb_path_llm)
            run.use_artifact(labelled_passages_wandb_path)

        console.print("Loading BERT classifier's training data from W&B...")
        model_artifact = wandb_api.artifact(classifier_wandb_path_bert)
        artifact_creator_run = model_artifact.logged_by()

        try:
            assert artifact_creator_run is not None

            training_data_artifact = None
            for artifact in artifact_creator_run.logged_artifacts():
                if (
                    artifact.type == "labelled_passages"
                    and "training-data" in artifact.name
                ):
                    training_data_artifact = artifact
                    break

            if training_data_artifact is None:
                raise ValueError(
                    "No training data artifact found in the model's creating run"
                )

            console.print(
                f"Loading from artifact: {training_data_artifact.name}:{training_data_artifact.version}"
            )
            artifact_dir = training_data_artifact.download()
            labelled_passages_file = Path(artifact_dir) / "labelled_passages.jsonl"

            bert_training_data = deserialise_pydantic_list_with_fallback(
                labelled_passages_file.read_text(), LabelledPassage
            )
            console.print(
                f"Loaded {len(bert_training_data)} passages from BERT training data"
            )
        except Exception as e:
            console.print(
                f"[yellow]Warning: Could not load training data from classifier run: {e}[/yellow]"
            )
            console.print(
                "[yellow]This model may have been trained before training data logging was added.[/yellow]"
            )
            bert_training_data = []

        evaluation_data = bert_classifier.concept.labelled_passages
        console.print(
            f"Loaded {len(evaluation_data)} passages from concept evaluation data"
        )

        console.print(
            "Filtering out passages that were used in training or evaluation..."
        )
        passage_text_to_exclude = {
            passage.text for passage in bert_training_data + evaluation_data
        }
        original_count = len(labelled_passages)
        labelled_passages = [
            p for p in labelled_passages if p.text not in passage_text_to_exclude
        ]
        filtered_count = original_count - len(labelled_passages)
        console.print(
            f"Removed {filtered_count} passages (training + eval data), "
            f"{len(labelled_passages)} passages remaining for annotation"
        )

        if limit is not None:
            labelled_passages = labelled_passages[:limit]
            console.print(f"Limited labelled passages to {len(labelled_passages)}")

        # run classifier escalation
        bert_labelled_passages, llm_labelled_passages, unlabelled_passages = (
            run_active_learning(
                wikibase_id=wikibase_id,
                labelled_passages=labelled_passages,
                bert_classifier=bert_classifier,
                llm_classifier=llm_classifier,
                ensemble_config_bert=ensemble_config_bert,
                ensemble_config_llm=ensemble_config_llm,
                batch_size=batch_size,
            )
        )

        # Make updated BERT training data from its initial training data, plus all the
        # passages labelled by the LLM ensemble.
        updated_bert_training_data = bert_training_data + llm_labelled_passages

        timestr = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"data/processed/active_learning/{wikibase_id}_{timestr}")
        output_dir.mkdir(exist_ok=True, parents=True)

        bert_labelled_filename = f"bert_labelled_passages_{wikibase_id}.jsonl"
        bert_labelled_path = output_dir / bert_labelled_filename
        bert_labelled_jsonl = serialise_pydantic_list_as_jsonl(bert_labelled_passages)
        bert_labelled_path.write_text(bert_labelled_jsonl)
        console.print(
            f"✅ Saved {len(bert_labelled_passages)} BERT-labelled passages to {bert_labelled_path}"
        )

        llm_labelled_filename = f"llm_labelled_passages_{wikibase_id}.jsonl"
        llm_labelled_path = output_dir / llm_labelled_filename
        llm_labelled_jsonl = serialise_pydantic_list_as_jsonl(llm_labelled_passages)
        llm_labelled_path.write_text(llm_labelled_jsonl)
        console.print(
            f"✅ Saved {len(llm_labelled_passages)} LLM-labelled passages to {llm_labelled_path}"
        )

        unlabelled_filename = f"unlabelled_passages_{wikibase_id}.jsonl"
        unlabelled_path = output_dir / unlabelled_filename
        unlabelled_jsonl = serialise_pydantic_list_as_jsonl(unlabelled_passages)
        unlabelled_path.write_text(unlabelled_jsonl)
        console.print(
            f"✅ Saved {len(unlabelled_passages)} unlabelled passages to {unlabelled_path}"
        )

        updated_bert_training_filename = (
            f"updated_bert_training_data_{wikibase_id}.jsonl"
        )
        updated_bert_training_path = output_dir / updated_bert_training_filename
        updated_bert_training_jsonl = serialise_pydantic_list_as_jsonl(
            updated_bert_training_data
        )
        updated_bert_training_path.write_text(updated_bert_training_jsonl)
        console.print(
            f"✅ Saved updated BERT training data ({len(updated_bert_training_data)} passages) to {updated_bert_training_path}"
        )

        # Upload artifacts to W&B
        if track_and_upload and run:
            console.print("[cyan]Uploading artifacts to W&B...[/cyan]")

            # Upload BERT-labelled passages
            log_labelled_passages_artifact_to_wandb_run(
                bert_labelled_passages,
                run=run,
                concept=bert_classifier.concept,
                classifier=bert_classifier,
            )
            console.print(
                f"✅ Uploaded {len(bert_labelled_passages)} BERT-labelled passages to W&B"
            )

            # Upload LLM-labelled passages
            log_labelled_passages_artifact_to_wandb_run(
                llm_labelled_passages,
                run=run,
                concept=bert_classifier.concept,
                classifier=llm_classifier,
            )
            console.print(
                f"✅ Uploaded {len(llm_labelled_passages)} LLM-labelled passages to W&B"
            )

            # Upload unlabelled passages (need human annotation)
            log_labelled_passages_artifact_to_wandb_run(
                unlabelled_passages,
                run=run,
                concept=bert_classifier.concept,
                artifact_name="unlabelled",
            )
            console.print(
                f"✅ Uploaded {len(unlabelled_passages)} unlabelled passages to W&B"
            )

            # Upload updated BERT training data
            log_labelled_passages_artifact_to_wandb_run(
                updated_bert_training_data,
                run=run,
                concept=bert_classifier.concept,
                artifact_name="updated_bert_training_data",
            )
            console.print(
                f"✅ Uploaded {len(updated_bert_training_data)} updated BERT training passages to W&B"
            )

            run.summary["num_bert_labelled_passages"] = len(bert_labelled_passages)
            run.summary["num_llm_labelled_passages"] = len(llm_labelled_passages)
            run.summary["num_unlabelled_passages"] = len(unlabelled_passages)
            run.summary["num_updated_bert_training_data"] = len(
                updated_bert_training_data
            )


if __name__ == "__main__":
    app()
