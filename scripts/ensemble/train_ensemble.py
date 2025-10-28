import asyncio
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.progress import Progress

from knowledge_graph.cloud import AwsEnv
from knowledge_graph.ensemble import create_ensemble
from knowledge_graph.identifiers import WikibaseID
from scripts.get_concept import get_concept_async
from scripts.train import (
    load_training_data_from_wandb,
    parse_kwargs_from_strings,
    train_classifier,
)

app = typer.Typer()


async def train_ensemble(
    wikibase_id: WikibaseID,
    classifier_type: Annotated[
        str,
        typer.Option(
            help="Classifier type to use (e.g., LLMClassifier, KeywordClassifier)",
        ),
    ],
    n_classifiers: Annotated[
        int,
        typer.Option(help="Number of classifiers to include in the ensemble."),
    ],
    classifier_override: Annotated[
        Optional[list[str]],
        typer.Option(
            help="Classifier kwarg overrides in key=value format. Can be specified multiple times.",
        ),
    ] = None,
    training_data_wandb_run_path: Annotated[
        Optional[str],
        typer.Option(
            help="W&B run path (entity/project/run_id) to fetch training data from instead of using concept's labelled passages",
        ),
    ] = None,
):
    """
    Train an ensemble of classifiers of the same type on a concept.

    Logs the training runs (including inference on the concept's evaluation set) in
    weights and biases. Also prints the ID of the newly-created ensemble so it can
    be easily found in weights and biases.

    The classifiers are uploaded to the labs aws environment
    """

    console = Console()

    console.log(f"Getting concept {wikibase_id}")
    # TODO: could also implement WikibaseConfig here if we want to be able to use this
    # script in prefect
    concept = await get_concept_async(
        wikibase_id=wikibase_id,
        include_labels_from_subconcepts=True,
        include_recursive_has_subconcept=True,
    )

    labelled_passages = None
    if training_data_wandb_run_path:
        labelled_passages = load_training_data_from_wandb(
            training_data_wandb_run_path, console
        )

    classifier_kwargs = parse_kwargs_from_strings(classifier_override)
    ensemble = create_ensemble(
        concept=concept,
        classifier_type=classifier_type,
        classifier_kwargs=classifier_kwargs,
        n_classifiers=n_classifiers,
    )

    ensemble_name = str(ensemble)
    console.log(
        f"Ensemble created. To find relevant runs, filter by ensemble_name = {ensemble_name}."
    )

    extra_wandb_config = {
        "ensemble_name": ensemble_name,
        "experimental_model_type": True,
        "classifier_kwargs": classifier_kwargs,
    }
    if training_data_wandb_run_path:
        extra_wandb_config["training_data_wandb_run_path"] = (
            training_data_wandb_run_path
        )

    classifiers = ensemble.classifiers

    with Progress() as progress:
        task = progress.add_task(
            f"[cyan]Training {len(classifiers)} instances of {classifier_type}, with extra kwargs {classifier_kwargs}......",
            total=len(classifiers),
        )
        for clf in classifiers:
            await train_classifier(
                clf,
                wikibase_id,
                track_and_upload=True,
                evaluate=True,
                aws_env=AwsEnv.labs,
                extra_wandb_config=extra_wandb_config,
                train_validation_data=labelled_passages,
            )
            progress.update(task, advance=1)


@app.command()
def main(
    wikibase_id: Annotated[
        WikibaseID,
        typer.Option(
            ...,
            help="The Wikibase ID of the concept classifier to train",
            parser=WikibaseID,
        ),
    ],
    classifier_type: Annotated[
        str,
        typer.Option(
            help="Classifier type to use (e.g., LLMClassifier, KeywordClassifier)",
        ),
    ],
    n_classifiers: Annotated[
        int,
        typer.Option(help="Number of classifiers to include in the ensemble."),
    ] = 10,
    classifier_kwarg: Annotated[
        Optional[list[str]],
        typer.Option(
            help="Classifier kwargs in key=value format. Can be specified multiple times.",
        ),
    ] = None,
    training_data_wandb_run_path: Annotated[
        Optional[str],
        typer.Option(
            help="W&B run path (entity/project/run_id) to fetch training data from instead of using concept's labelled passages",
        ),
    ] = None,
):
    """
    Train an ensemble of classifiers of the same type on a concept.

    Logs the training runs (including inference on the concept's evaluation set) in
    weights and biases. Also prints the ID of the newly-created ensemble so it can
    be easily found in weights and biases.

    The classifiers are uploaded to the labs aws environment
    """
    return asyncio.run(
        train_ensemble(
            wikibase_id=wikibase_id,
            classifier_type=classifier_type,
            n_classifiers=n_classifiers,
            classifier_override=classifier_kwarg,
            training_data_wandb_run_path=training_data_wandb_run_path,
        )
    )


if __name__ == "__main__":
    app()
