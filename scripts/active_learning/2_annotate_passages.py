from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
import wandb
from dotenv import load_dotenv
from pydantic import BaseModel
from rich.console import Console

import knowledge_graph.ensemble.metrics as ensemble_metrics
from flows.utils import serialise_pydantic_list_as_jsonl
from knowledge_graph.classifier import Classifier, load_classifier_from_wandb
from knowledge_graph.ensemble import create_ensemble
from knowledge_graph.ensemble.metrics import EnsembleMetric, MajorityVote
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.span import Span, UnitInterval
from knowledge_graph.wandb_helpers import load_labelled_passages_from_wandb_run

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


def annotate_passages(
    labelled_passages: list[LabelledPassage],
    wikibase_id: WikibaseID,
    bert_classifier: Classifier,
    llm_classifier: Classifier,
    ensemble_config_bert: EnsembleConfig,
    ensemble_config_llm: EnsembleConfig,
    batch_size: int = 50,
) -> tuple[list[LabelledPassage], list[LabelledPassage]]:
    """Annotate passages using ensemble-based uncertainty estimation."""
    console = Console()

    models_and_ensemble_configs = [
        (bert_classifier, ensemble_config_bert),
        (llm_classifier, ensemble_config_llm),
    ]
    passages_to_annotate: list[LabelledPassage] = labelled_passages
    already_annotated_passages: list[LabelledPassage] = []

    for model, ensemble_config in models_and_ensemble_configs:
        console.print(
            f"Creating ensemble with {ensemble_config.n_classifiers} variants from base classifier {model}"
        )
        ensemble = create_ensemble(
            classifier=model,
            n_classifiers=ensemble_config.n_classifiers,
        )

        # predict all passages using ensemble
        console.print(f"Predicting on {len(passages_to_annotate)} passages...")
        text_to_predict = [passage.text for passage in passages_to_annotate]
        ensemble_predicted_spans = ensemble.predict(
            text_to_predict,
            batch_size=batch_size,
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
        new_passages_to_annotate: list[LabelledPassage] = []
        annotated_count = 0

        for passage, metric_value, majority_vote_value in zip(
            passages_to_annotate, ensemble_metrics, ensemble_majority_votes
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
                already_annotated_passages.append(
                    passage.model_copy(update={"spans": [span]})
                )
                annotated_count += 1
            else:
                new_passages_to_annotate.append(passage)

        passages_to_annotate = new_passages_to_annotate
        console.print(
            f"Annotated {annotated_count} passages with metric {ensemble_config.ensemble_metric.name} greater than {ensemble_config.ensemble_metric_threshold}. {len(passages_to_annotate)} remaining."
        )

    percent_annotated = round(
        len(already_annotated_passages) / len(labelled_passages) * 100, 1
    )
    percent_to_annotate = round(100 - percent_annotated, 1)

    console.print(
        f"🎉 Complete: {len(already_annotated_passages)} ({percent_annotated}%) annotated, {len(passages_to_annotate)} ({percent_to_annotate}%) remaining which both ensembles were uncertain about."
    )

    return already_annotated_passages, passages_to_annotate


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
    labelled_passages_wandb_run_path: Annotated[
        str,
        typer.Option(
            ...,
            help="W&B run path to load labelled passages from (e.g., 'entity/project/run_id')",
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
):
    """
    Run active learning annotation using ensemble-based uncertainty estimation.

    Loads unlabeled passages and uses two classifier ensembles (BERT and LLM) to identify
    passages where both models are uncertain. These passages are the best candidates for
    human annotation.
    """
    console = Console()

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
        f"Loading labelled passages from W&B run: {labelled_passages_wandb_run_path}"
    )
    wandb_api = wandb.Api()
    wandb_run = wandb_api.run(labelled_passages_wandb_run_path)
    labelled_passages = load_labelled_passages_from_wandb_run(wandb_run)

    console.print(f"Loaded {len(labelled_passages)} passages")

    console.print("Loading BERT classifier's training data from W&B...")
    model_artifact = wandb_api.artifact(classifier_wandb_path_bert)
    artifact_creator_run = model_artifact.logged_by()

    try:
        assert artifact_creator_run is not None
        bert_training_data = load_labelled_passages_from_wandb_run(
            artifact_creator_run,  # type: ignore
            artifact_name="training-data",
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

    console.print("Filtering out passages that were used in training or evaluation...")
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
    annotated_passages, remaining_passages = annotate_passages(
        labelled_passages=labelled_passages,
        wikibase_id=wikibase_id,
        bert_classifier=bert_classifier,
        llm_classifier=llm_classifier,
        ensemble_config_bert=ensemble_config_bert,
        ensemble_config_llm=ensemble_config_llm,
        batch_size=batch_size,
    )

    # Save passages to local files
    output_dir = Path("active_learning_output")
    output_dir.mkdir(exist_ok=True)

    annotated_filename = f"annotated_passages_{wikibase_id}.jsonl"
    annotated_path = output_dir / annotated_filename
    annotated_jsonl = serialise_pydantic_list_as_jsonl(annotated_passages)
    annotated_path.write_text(annotated_jsonl)
    console.print(
        f"✅ Saved {len(annotated_passages)} annotated passages to {annotated_path}"
    )

    remaining_filename = f"remaining_passages_{wikibase_id}.jsonl"
    remaining_path = output_dir / remaining_filename
    remaining_jsonl = serialise_pydantic_list_as_jsonl(remaining_passages)
    remaining_path.write_text(remaining_jsonl)
    console.print(
        f"✅ Saved {len(remaining_passages)} remaining passages to {remaining_path}"
    )

    # FIXME: upload LPs to W&B
    # FIXME: upload LPs to argilla


if __name__ == "__main__":
    app()
