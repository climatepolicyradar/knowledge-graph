import asyncio
from typing import Annotated, Optional, cast

import typer
from rich.console import Console
from rich.progress import Progress

from knowledge_graph.classifier.classifier import Classifier, VariantEnabledClassifier
from knowledge_graph.cloud import AwsEnv
from knowledge_graph.ensemble import Ensemble
from knowledge_graph.identifiers import WikibaseID
from scripts.get_concept import get_concept_async
from scripts.train import create_classifier, parse_classifier_kwargs, train_classifier

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
        typer.Option(
            default=10, help="Number of classifiers to include in the ensemble."
        ),
    ],
    classifier_kwarg: Annotated[
        Optional[list[str]],
        typer.Option(
            help="Classifier kwargs in key=value format. Can be specified multiple times.",
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

    # TODO: warn that random seed will be ignored for LLMClassifier
    classifier_kwargs = parse_classifier_kwargs(classifier_kwarg)
    initial_classifier = create_classifier(concept, classifier_type, classifier_kwargs)

    if not isinstance(initial_classifier, VariantEnabledClassifier):
        raise typer.BadParameter(
            f"Classifier type must be variant-enabled to be part of an ensemble.\nClassifier type {classifier_type} is not."
        )

    classifiers: list[Classifier] = [
        initial_classifier,
        *[
            cast(Classifier, initial_classifier.get_variant())
            for _ in range(n_classifiers - 1)
        ],
    ]

    ensemble = Ensemble(concept=concept, classifiers=classifiers)
    ensemble_name = str(ensemble)
    console.log(
        f"Ensemble assembled. To find relevant runs, filter by ensemble_name = {ensemble_name}."
    )

    extra_wandb_config = {
        "ensemble_name": ensemble_name,
    }

    with Progress() as progress:
        task = progress.add_task(
            f"[cyan]Training {len(classifiers)} instances of {classifier_type}, with extra kwargs {classifier_kwargs}......",
            total=len(classifiers),
        )
        for clf in classifiers:
            train_classifier(
                clf,
                wikibase_id,
                track_and_upload=True,
                evaluate=True,
                experimental_model_type=True,
                aws_env=AwsEnv.labs,
                extra_wandb_config=extra_wandb_config,
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
        typer.Option(
            default=10, help="Number of classifiers to include in the ensemble."
        ),
    ],
    classifier_kwarg: Annotated[
        Optional[list[str]],
        typer.Option(
            help="Classifier kwargs in key=value format. Can be specified multiple times.",
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
            classifier_kwarg=classifier_kwarg,
        )
    )


if __name__ == "__main__":
    app()
