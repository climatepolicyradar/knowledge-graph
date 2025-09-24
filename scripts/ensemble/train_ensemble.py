from typing import Annotated, Optional, cast

import typer

from knowledge_graph.classifier.classifier import Classifier, VariantEnabledClassifier
from knowledge_graph.cloud import AwsEnv
from knowledge_graph.identifiers import WikibaseID
from scripts.get_concept import get_concept_async
from scripts.train import create_classifier, parse_classifier_kwargs, train_classifier


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

    # TODO: could also implement WikibaseConfig here if we want to be able to use this
    # script in prefect
    concept = await get_concept_async(
        wikibase_id=wikibase_id,
        include_labels_from_subconcepts=True,
        include_recursive_has_subconcept=True,
    )

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

    # ensemble = Ensemble(concept=concept, classifiers=classifiers)
    # ensemble_id = str(ensemble)

    for clf in classifiers:
        train_classifier(
            clf,
            wikibase_id,
            track_and_upload=True,
            evaluate=True,
            experimental_model_type=True,
            aws_env=AwsEnv.labs,
        )
