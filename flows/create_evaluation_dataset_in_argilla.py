from typing import Annotated, Optional

from prefect import flow
from pydantic import Field

from flows.config import Config
from flows.push_new_dataset import push_new_dataset
from flows.sample import sample
from knowledge_graph.cloud import AwsEnv
from knowledge_graph.identifiers import WikibaseID


@flow
async def create_evaluation_dataset_in_argilla(
    wikibase_id: Annotated[
        WikibaseID,
        Field(
            description="The Wikibase ID of the concept to sample and push to Argilla"
        ),
    ],
    workspace_name: Annotated[
        str,
        Field(description="The name of the existing workspace in Argilla"),
    ] = "knowledge-graph",
    dataset_name: Annotated[
        str,
        Field(
            description="Dataset to use",
            json_schema_extra={"enum": ["balanced", "combined"]},
        ),
    ] = "balanced",
    sample_size: Annotated[
        int,
        Field(description="The number of passages to sample"),
    ] = 130,
    min_negative_proportion: Annotated[
        float,
        Field(description="The minimum proportion of negative samples to take"),
    ] = 0.1,
    corpus_types_include: Annotated[
        Optional[list[str]],
        Field(
            description="Corpus types to include. If not set, all types are included.",
        ),
    ] = None,
    corpus_types_exclude: Annotated[
        Optional[list[str]],
        Field(description="Corpus types to exclude."),
    ] = None,
    max_size_to_sample_from: Annotated[
        int,
        Field(
            description="Maximum number of passages to load from the dataset before sampling"
        ),
    ] = 500_000,
    max_negative_proportion: Annotated[
        Optional[float],
        Field(
            description="Maximum proportion of the sample that can be negative. If not set, fills remaining sample_size after positives."
        ),
    ] = None,
    concept_override: Annotated[
        Optional[list[str]],
        Field(
            description="Concept property overrides in key=value format.",
        ),
    ] = None,
    limit: Annotated[
        Optional[int],
        Field(description="Limit the number of passages loaded to Argilla"),
    ] = 130,
    aws_env: AwsEnv = AwsEnv.production,
    config: Optional[Config] = None,
) -> None:
    """
    Sample passages for a concept and push them to Argilla as a new dataset.

    Runs the sample flow (uploading the artifact to W&B) then immediately calls
    push_new_dataset using the latest artifact for the concept.
    """
    if not config:
        config = await Config.create()

    wandb_artifact_path = await sample(
        wikibase_id=wikibase_id,
        dataset_name=dataset_name,
        sample_size=sample_size,
        min_negative_proportion=min_negative_proportion,
        corpus_types_include=corpus_types_include,
        corpus_types_exclude=corpus_types_exclude,
        max_size_to_sample_from=max_size_to_sample_from,
        max_negative_proportion=max_negative_proportion,
        track_and_upload=True,
        concept_override=concept_override,
        aws_env=aws_env,
        config=config,
    )

    if wandb_artifact_path is None:
        raise RuntimeError("sample flow did not return an artifact path")

    await push_new_dataset(
        wikibase_id=wikibase_id,
        wandb_artifact_path=wandb_artifact_path,
        workspace_name=workspace_name,
        limit=limit,
        config=config,
    )
