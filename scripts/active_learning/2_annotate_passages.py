from pydantic import BaseModel
from rich.console import Console

import knowledge_graph.ensemble.metrics as ensemble_metrics
from knowledge_graph.cloud import AwsEnv
from knowledge_graph.ensemble import create_ensemble
from knowledge_graph.ensemble.metrics import EnsembleMetric
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelled_passage import LabelledPassage
from knowledge_graph.span import UnitInterval
from scripts.train import run_training


class EnsembleConfig(BaseModel):
    """An Ensemble, and a metric and threshold to use for escalating predictions."""

    n_classifiers: int
    classifier_type: str
    classifier_kwargs: dict[str, object] = {}
    ensemble_metric: EnsembleMetric
    ensemble_metric_threshold: UnitInterval


ensemble_config_bert = EnsembleConfig(
    n_classifiers=5,
    # TODO: it'd maybe be nice if we could pass in an initial classifier here rather than
    # a string for type checking? But then, this has advantages that it's easy to stick
    # in a text-based config system e.g. for prefect
    classifier_type="BertBasedClassifier",
    ensemble_metric=ensemble_metrics.Disagreement(),
    ensemble_metric_threshold=UnitInterval(0.0),
)

ensemble_config_llm = EnsembleConfig(
    n_classifiers=5,
    classifier_type="LLMClassifier",
    classifier_kwargs={"model_name": "gpt-5-mini"},
    ensemble_metric=ensemble_metrics.Disagreement(),
    ensemble_metric_threshold=UnitInterval(0.0),
)

ensemble_configs = [
    ensemble_config_bert,
    ensemble_config_llm,
]


async def annotate_passages(
    labelled_passages: list[LabelledPassage],
    wikibase_id: WikibaseID,
    training_data_wandb_run_path: str,
    ensemble_configs: list[EnsembleConfig],
):
    """Annotate passages using ensemble-based uncertainty estimation."""
    console = Console()

    console.print(
        f"Starting annotation of {len(labelled_passages)} passages with {len(ensemble_configs)} ensembles"
    )

    passages_to_annotate: list[LabelledPassage] = labelled_passages
    annotated_passages: list[tuple[LabelledPassage, UnitInterval]] = []

    for i, ensemble_config in enumerate(ensemble_configs, 1):
        console.print(
            f"⚙️ Processing ensemble {i}/{len(ensemble_configs)}: {ensemble_config.classifier_type}"
        )

        # Train classifier using run_training for full W&B tracking and evaluation
        console.print(
            f"Training {ensemble_config.classifier_type} classifier with W&B tracking"
        )
        initial_classifier = await run_training(
            wikibase_id=wikibase_id,
            track_and_upload=True,
            aws_env=AwsEnv.labs,
            classifier_type=ensemble_config.classifier_type,
            classifier_kwargs=ensemble_config.classifier_kwargs,
            training_data_wandb_run_path=training_data_wandb_run_path,
            evaluate=True,
        )

        # create ensemble from fitted classifier
        console.print(
            f"Creating ensemble with {ensemble_config.n_classifiers} variants from fitted classifier"
        )
        ensemble = create_ensemble(
            classifier=initial_classifier,
            n_classifiers=ensemble_config.n_classifiers,
        )

        # predict all passages on ensemble
        console.print(f"Predicting on {len(passages_to_annotate)} passages...")
        text_to_predict = [passage.text for passage in passages_to_annotate]
        predicted_spans = ensemble.predict(
            text_to_predict,
            batch_size=15,
        )

        # get uncertainties
        ensemble_metrics = [
            ensemble_config.ensemble_metric(spans) for spans in predicted_spans
        ]

        # for uncertainties <= threshold, add these passages to annotated_passages
        # and remove them from passages_to_annotate
        new_passages_to_annotate = []
        annotated_count = 0

        for passage, metric in zip(passages_to_annotate, ensemble_metrics):
            if metric <= ensemble_config.ensemble_metric_threshold:
                # FIXME: also need to store ensemble prediction here. can potentially
                # implement a MajorityVote metric and calculate it above, and then use
                # the result here. otherwise just pick the first classifier's results
                # from the majority class (which might be better, as then we get span-level
                # labels)
                annotated_passages.append((passage, metric))
                annotated_count += 1
            else:
                new_passages_to_annotate.append(passage)
        passages_to_annotate = new_passages_to_annotate
        console.print(
            f"Annotated {annotated_count} passages the ensemble is certain about. {len(passages_to_annotate)} remaining."
        )

    console.print(
        f"🎉 Complete: {len(annotated_passages)} annotated, {len(passages_to_annotate)} remaining for humans"
    )

    # FIXME: save passages to file and upload to W&B

    return annotated_passages
