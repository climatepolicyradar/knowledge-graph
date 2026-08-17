import os
from typing import Annotated, Optional

import wandb
from prefect import flow
from pydantic import Field

from flows.config import Config
from flows.sample import sample
from knowledge_graph.cloud import AwsEnv
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelling import (
    create_llm_ensemble_for_prelabelling_validation_dataset,
)
from knowledge_graph.operations.extend_dataset import (
    get_passage_ids_in_argilla,
    run_extend_dataset,
)
from knowledge_graph.utils import get_logger
from knowledge_graph.wandb_helpers import load_labelled_passages_from_wandb
from knowledge_graph.wikibase import WikibaseSession


@flow
async def extend_existing_dataset(
    wikibase_id: Annotated[
        WikibaseID,
        Field(description="The Wikibase ID of the concept whose dataset to extend"),
    ],
    wandb_artifact_path: Annotated[
        Optional[str],
        Field(description="W&B artifact to take candidate passages from."),
    ] = None,
    n_new_passages: Annotated[
        Optional[int],
        Field(
            description="Number of new passages to add, applied after deduplicating "
            "against the passages already in Argilla"
        ),
    ] = 50,
    sample_size: Annotated[
        Optional[int],
        Field(
            description="Number of passages to sample. Defaults to n_new_passages; "
            "passages already in Argilla are excluded from the sampling pool"
        ),
    ] = None,
    require_full_limit: Annotated[
        bool,
        Field(
            description="Fail, rather than warn and add fewer, when fewer than "
            "n_new_passages of the input passages are new"
        ),
    ] = True,
    workspace_name: Annotated[
        str,
        Field(description="The name of the existing workspace in Argilla"),
    ] = "knowledge-graph",
    dataset_name: Annotated[
        str,
        Field(
            description="Dataset to sample from",
            json_schema_extra={"enum": ["balanced", "combined"]},
        ),
    ] = "balanced",
    min_negative_proportion: Annotated[
        float,
        Field(description="The minimum proportion of negative samples to take"),
    ] = 0.1,
    max_negative_proportion: Annotated[
        Optional[float],
        Field(
            description="Maximum proportion of the sample that can be negative. If not "
            "set, fills remaining sample_size after positives."
        ),
    ] = None,
    corpus_types_include: Annotated[
        Optional[list[str]],
        Field(
            description="Corpus types to include. If not set, all types are included."
        ),
    ] = None,
    corpus_types_exclude: Annotated[
        Optional[list[str]],
        Field(description="Corpus types to exclude."),
    ] = None,
    max_size_to_sample_from: Annotated[
        int,
        Field(
            description="Maximum number of passages to load from the dataset before "
            "sampling."
        ),
    ] = 500_000,
    concept_override: Annotated[
        Optional[list[str]],
        Field(description="Concept property overrides in key=value format."),
    ] = None,
    prelabel_with_llm_ensemble: Annotated[
        bool,
        Field(
            description="Whether to attach LLM ensemble predictions as annotator "
            "suggestions. Defaults to False."
        ),
    ] = False,
    aws_env: AwsEnv = AwsEnv.production,
    config: Optional[Config] = None,
) -> int:
    """Add more labelling passages to a concept's existing Argilla dataset."""
    logger = get_logger()

    if not config:
        config = await Config.create()

    argilla_api_url = config.argilla_api_url or os.getenv("ARGILLA_API_URL")
    argilla_api_key = (
        config.argilla_api_key.get_secret_value() if config.argilla_api_key else None
    ) or os.getenv("ARGILLA_API_KEY")
    if not argilla_api_url or not argilla_api_key:
        raise ValueError(
            "Missing Argilla credentials. Set /Argilla/APIURL and "
            "/Argilla/Owner/APIKey in SSM for this environment, or ARGILLA_API_URL and "
            "ARGILLA_API_KEY in the environment."
        )

    if config.wandb_api_key:
        wandb.login(key=config.wandb_api_key.get_secret_value())

    if prelabel_with_llm_ensemble and config.openrouter_api_key:
        os.environ["OPENROUTER_API_KEY"] = config.openrouter_api_key.get_secret_value()

    existing_passage_ids = get_passage_ids_in_argilla(
        wikibase_id=wikibase_id,
        workspace=workspace_name,
        argilla_api_url=argilla_api_url,
        argilla_api_key=argilla_api_key,
    )

    if not wandb_artifact_path:
        resolved_sample_size = (
            sample_size if sample_size is not None else n_new_passages
        )
        if resolved_sample_size is None:
            raise ValueError(
                f"No sample size for {wikibase_id}: set n_new_passages, or sample_size "
                f"if n_new_passages is deliberately unlimited."
            )
        logger.info(
            f"No wandb_artifact_path given, sampling {resolved_sample_size} passages for "
            f"{wikibase_id}"
        )
        wandb_artifact_path = await sample(
            wikibase_id=wikibase_id,
            dataset_name=dataset_name,
            sample_size=resolved_sample_size,
            min_negative_proportion=min_negative_proportion,
            max_negative_proportion=max_negative_proportion,
            corpus_types_include=corpus_types_include,
            corpus_types_exclude=corpus_types_exclude,
            max_size_to_sample_from=max_size_to_sample_from,
            exclude_passage_ids=sorted(existing_passage_ids),
            track_and_upload=True,
            concept_override=concept_override,
            aws_env=aws_env,
            config=config,
        )
        if wandb_artifact_path is None:
            raise RuntimeError("sample flow did not return an artifact path")

    labelled_passages = load_labelled_passages_from_wandb(
        wandb_path=wandb_artifact_path
    )
    logger.info(
        f"Loaded {len(labelled_passages)} candidate passages from {wandb_artifact_path}"
    )

    suggestion_model = None
    if prelabel_with_llm_ensemble:
        wikibase = WikibaseSession(
            username=config.wikibase_username,
            password=config.wikibase_password.get_secret_value()
            if config.wikibase_password
            else None,
            url=config.wikibase_url,
        )
        concept = await wikibase.get_concept_async(wikibase_id)
        suggestion_model = create_llm_ensemble_for_prelabelling_validation_dataset(
            concept
        )

    added = run_extend_dataset(
        wikibase_id=wikibase_id,
        labelled_passages=labelled_passages,
        workspace=workspace_name,
        limit=n_new_passages,
        suggestion_model=suggestion_model,
        require_full_limit=require_full_limit,
        existing_passage_ids=existing_passage_ids,
        argilla_api_url=argilla_api_url,
        argilla_api_key=argilla_api_key,
    )
    return len(added)
