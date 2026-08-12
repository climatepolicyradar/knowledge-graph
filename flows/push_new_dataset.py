import os
from typing import Annotated, Optional

import wandb
from prefect import flow
from pydantic import Field

from flows.config import Config
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.labelling import (
    ArgillaSession,
    create_llm_ensemble_for_prelabelling_validation_dataset,
)
from knowledge_graph.utils import get_logger
from knowledge_graph.wandb_helpers import load_labelled_passages_from_wandb
from knowledge_graph.wikibase import WikibaseSession


@flow
async def push_new_dataset(
    wikibase_id: Annotated[
        WikibaseID,
        Field(description="The Wikibase ID of the concept to create a dataset for"),
    ],
    wandb_artifact_path: Annotated[
        str,
        Field(
            description="W&B artifact path to load labelled passages from, e.g. 'climatepolicyradar/Q913/labelled-passages:v0'"
        ),
    ],
    workspace_name: Annotated[
        str,
        Field(description="The name of the existing workspace in Argilla"),
    ] = "knowledge-graph",
    limit: Annotated[
        Optional[int],
        Field(description="Limit the number of passages loaded to Argilla"),
    ] = 130,
    prelabel_with_llm_ensemble: Annotated[
        bool,
        Field(
            description="Whether to pre-label the passages with an LLM ensemble, whose "
            "predictions are shown to annotators as suggestions"
        ),
    ] = True,
    config: Optional[Config] = None,
) -> None:
    """
    Push labelled passages from a W&B artifact to Argilla as a new dataset.

    Reads the output of a prior sample flow run from W&B.
    """
    logger = get_logger()

    if not config:
        config = await Config.create()

    if config.wandb_api_key:
        wandb.login(key=config.wandb_api_key.get_secret_value())

    if prelabel_with_llm_ensemble and config.openrouter_api_key:
        os.environ["OPENROUTER_API_KEY"] = config.openrouter_api_key.get_secret_value()

    labelled_passages = load_labelled_passages_from_wandb(
        wandb_path=wandb_artifact_path
    )

    if limit is not None:
        logger.info(f"Limiting number of labelled passages to {limit}")
        labelled_passages = labelled_passages[:limit]

    argilla = ArgillaSession(
        api_url=config.argilla_api_url,
        api_key=config.argilla_api_key.get_secret_value()
        if config.argilla_api_key
        else None,
    )
    logger.info("✅ Connected to Argilla")

    wikibase = WikibaseSession(
        username=config.wikibase_username,
        password=config.wikibase_password.get_secret_value()
        if config.wikibase_password
        else None,
        url=config.wikibase_url,
    )
    concept = await wikibase.get_concept_async(wikibase_id)
    logger.info(f"✅ Loaded metadata for {concept}")

    dataset = argilla.create_dataset(concept, workspace=workspace_name)
    logger.info(f'✅ Created dataset "{dataset.name}" for {concept}')

    suggestion_model = None
    if prelabel_with_llm_ensemble:
        suggestion_model = create_llm_ensemble_for_prelabelling_validation_dataset(
            concept
        )
        logger.info(
            f"Pre-labelling {len(labelled_passages)} passages with an ensemble of "
            f"{len(suggestion_model.classifiers)} LLM classifiers"
        )

    argilla.add_labelled_passages(
        labelled_passages=labelled_passages,
        wikibase_id=wikibase_id,
        workspace=workspace_name,
        suggestion_model=suggestion_model,
    )
    logger.info(f"✅ Pushed {len(labelled_passages)} passages to dataset")
