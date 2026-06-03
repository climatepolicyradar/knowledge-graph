from typing import Annotated, Optional

import wandb
from prefect import flow
from pydantic import Field

from flows.config import Config
from knowledge_graph.cloud import AwsEnv
from knowledge_graph.identifiers import WikibaseID
from knowledge_graph.wandb_helpers import load_labelled_passages_from_wandb
from scripts.argilla.push_new_dataset import push_passages_to_argilla


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
    aws_env: AwsEnv = AwsEnv.production,
    config: Optional[Config] = None,
) -> None:
    """Push labelled passages from a W&B artifact to Argilla as a new dataset."""
    if not config:
        config = await Config.create()

    if config.wandb_api_key:
        wandb.login(key=config.wandb_api_key.get_secret_value())

    labelled_passages = load_labelled_passages_from_wandb(
        wandb_path=wandb_artifact_path
    )

    push_passages_to_argilla(
        labelled_passages=labelled_passages,
        wikibase_id=wikibase_id,
        workspace_name=workspace_name,
        limit=limit,
        argilla_api_url=config.argilla_api_url,
        argilla_api_key=config.argilla_api_key.get_secret_value()
        if config.argilla_api_key
        else None,
        wikibase_username=config.wikibase_username,
        wikibase_password=config.wikibase_password.get_secret_value()
        if config.wikibase_password
        else None,
        wikibase_url=config.wikibase_url,
    )
