"""
CLI wrapper for the train operation.

The reusable, Prefect-free logic (`run_training`, `train_classifier`, and the W&B/S3
helpers) lives in `knowledge_graph.operations.train` and is imported directly by the
training flows. This module only adds the Typer command used by `just train` and the
dispatch to the remote `train-on-cpu` / `train-on-gpu` Prefect deployments.
"""

import asyncio
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import typer
from prefect.client.schemas.objects import FlowRun
from prefect.deployments import run_deployment

from flows.utils import get_flow_run_ui_url
from knowledge_graph.classifier import Classifier
from knowledge_graph.cloud import AwsEnv, generate_deployment_name
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.operations.train import (
    resolve_config_inputs,
    run_training,
)
from knowledge_graph.utils import get_logger, parse_kwargs_from_strings
from knowledge_graph.wikibase import WikibaseSession

app = typer.Typer()


class ComputeTarget(str, Enum):
    """Where to run training."""

    local = "local"
    remote_cpu = "remote-cpu"
    remote_gpu = "remote-gpu"


# Maps remote compute targets to the Prefect flow that runs on that compute.
_REMOTE_FLOW_NAMES: dict["ComputeTarget", str] = {
    ComputeTarget.remote_cpu: "train-on-cpu",
    ComputeTarget.remote_gpu: "train-on-gpu",
}


@app.command()
def main(
    wikibase_id: Annotated[
        WikibaseID | None,
        typer.Option(
            help="The Wikibase ID to train. Required unless --from-yaml-config is given.",
            parser=WikibaseID,
        ),
    ] = None,
    from_yaml_config: Annotated[
        Path | None,
        typer.Option(
            help="Whether to use custom-classifier YAML config.",
        ),
    ] = None,
    track_and_upload: Annotated[
        bool,
        typer.Option(
            ...,
            help="Whether to track the training run with Weights & Biases. Includes uploading the model artifact to S3.",
        ),
    ] = True,
    aws_env: Annotated[
        AwsEnv,
        typer.Option(
            ...,
            help="AWS environment to use for S3 uploads",
        ),
    ] = AwsEnv.production,
    compute: Annotated[
        ComputeTarget,
        typer.Option(
            help=(
                "Where to run training. 'local' runs in-process. 'remote-cpu' and "
                "'remote-gpu' dispatch the corresponding Prefect deployment (the "
                "classifier won't be available locally after training). Use "
                "'remote-gpu' for BERT classifiers and 'remote-cpu' for non-BERT "
                "classifiers."
            ),
        ),
    ] = ComputeTarget.local,
    evaluate: Annotated[
        bool,
        typer.Option(
            ...,
            help="Whether to evaluate the model after training",
        ),
    ] = True,
    classifier_type: Annotated[
        Optional[str],
        typer.Option(
            help="Classifier type to use (e.g., LLMClassifier, KeywordClassifier). If not specified, uses ClassifierFactory default.",
        ),
    ] = None,
    classifier_override: Annotated[
        Optional[list[str]],
        typer.Option(
            help="Classifier kwarg overrides in key=value format. Can be specified multiple times.",
        ),
    ] = None,
    concept_override: Annotated[
        Optional[list[str]],
        typer.Option(
            help="Concept property overrides in key=value format. Can be specified multiple times.",
        ),
    ] = None,
    training_data_wandb_path: Annotated[
        Optional[str],
        typer.Option(
            help="W&B artifact path (e.g., 'entity/project/artifact:version') to fetch training data from.",
        ),
    ] = None,
    limit_training_samples: Annotated[
        Optional[int],
        typer.Option(
            help="Maximum number of training samples to use. Samples are selected in a way that achieves the best possible class balance. If not specified, all samples are used.",
        ),
    ] = None,
) -> Classifier | None:
    """
    Main function to train the model and optionally upload the artifact.

    :param wikibase_id: The Wikibase ID of the concept classifier to train.
    :type wikibase_id: WikibaseID
    :param from_yaml_config: Whether to use custom-classifier YAML config
    :type from_yaml_config: path
    :param track_and_upload: Whether to track the training run with Weights & Biases. Includes uploading the model artifact to S3.
    :type track_and_upload: bool
    :param aws_env: The AWS environment to use for S3 uploads.
    :type aws_env: AwsEnv
    :param compute: Where to run training: locally, or by dispatching the remote
        CPU or GPU Prefect deployment
    :type compute: ComputeTarget
    :param evaluate: Whether to evaluate the model after training
    :type evaluate: bool
    :param classifier_type: The classifier type to use, optional. Defaults to the
    classifier chosen by ClassifierFactory otherwise
    :type classifier_type: Optional[str]
    :param classifier_override: List of classifier kwargs in key=value format
    :type classifier_override: Optional[list[str]]
    :param concept_override: List of concept property overrides in key=value format (e.g., description, labels)
    :type concept_override: Optional[list[str]]
    :param limit_training_samples: Maximum number of training samples to use
    :type limit_training_samples: Optional[int]

    """

    classifier_kwargs = parse_kwargs_from_strings(classifier_override)
    concept_overrides = parse_kwargs_from_strings(concept_override)
    wikibase_id, cfg = resolve_config_inputs(wikibase_id, from_yaml_config)
    if cfg is not None:
        concept_overrides = cfg.concept_overrides.as_overrides()
        if classifier_type == "LLMClassifier":
            related = cfg.llm.related_definitions
            session = WikibaseSession() if related else None
            defs = (
                {
                    wid: (session.get_concept(wid).definition or "")
                    for wid in set(related)
                }
                if session
                else {}
            )
            classifier_kwargs = cfg.llm.to_classifier_kwargs(definitions=defs)
        elif classifier_type == "BertBasedClassifier":
            classifier_kwargs = cfg.bert.to_classifier_kwargs()
            training_data_wandb_path = cfg.bert.training_data_wandb_path
            limit_training_samples = cfg.bert.limit_training_samples
        else:
            raise typer.BadParameter(
                "--classifier-type must be LLMClassifier or BertBasedClassifier"
            )

    if compute is not ComputeTarget.local:
        flow_name = _REMOTE_FLOW_NAMES[compute]
        deployment_name = generate_deployment_name(flow_name=flow_name, aws_env=aws_env)
        qualified_name = f"{flow_name}/{deployment_name}"

        flow_run: FlowRun = run_deployment(  # type: ignore[misc]
            name=qualified_name,
            parameters={
                "wikibase_id": wikibase_id,
                "track_and_upload": track_and_upload,
                "aws_env": aws_env,
                "evaluate": evaluate,
                "classifier_type": classifier_type,
                "classifier_kwargs": classifier_kwargs,
                "concept_overrides": concept_overrides,
                "training_data_wandb_path": training_data_wandb_path,
                "limit_training_samples": limit_training_samples,
            },
            timeout=0,  # Don't wait for the flow to finish before continuing
        )
        get_logger().info(
            f"Deployment started. Flow run URL: {get_flow_run_ui_url(flow_run)}"
        )

        return None  # Can't return the classifier when running remotely
    else:
        return asyncio.run(
            run_training(
                wikibase_id=wikibase_id,
                track_and_upload=track_and_upload,
                aws_env=aws_env,
                evaluate=evaluate,
                classifier_type=classifier_type,
                classifier_kwargs=classifier_kwargs,
                concept_overrides=concept_overrides,
                training_data_wandb_path=training_data_wandb_path,
                limit_training_samples=limit_training_samples,
            )
        )


if __name__ == "__main__":
    app()
