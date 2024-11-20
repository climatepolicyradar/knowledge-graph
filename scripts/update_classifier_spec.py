import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import typer
import wandb
import yaml
from rich.console import Console
from wandb.apis.public import ArtifactType
from wandb.apis.public.artifacts import ArtifactCollection

from scripts.cloud import AwsEnv
from src.identifiers import WikibaseID

console = Console()

app = typer.Typer()


WANDB_MODEL_ORG = "climatepolicyradar_UZODYJSN66HCQ"
WANDB_MODEL_REGISTRY = "wandb-registry-model"
SPEC_DIR = Path("flows") / "classifier_specs"


def build_spec_file_path(aws_env: AwsEnv) -> str:
    file_path = SPEC_DIR / f"{aws_env}.yaml"
    return file_path


def read_spec_file(aws_env: AwsEnv) -> list[str]:
    file_path = build_spec_file_path(aws_env)
    with open(file_path, "r") as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def write_spec_file(file_path: str, data: list[str]):
    """Save a classifier spec YAML"""
    with open(file_path, "w") as file:
        yaml.dump(data, file, explicit_start=True)


def is_concept_model(model: ArtifactCollection) -> bool:
    """
    Check if a model is a concept classifier

    This check is based on whether the model name can be instantiated as a WikibaseID.
    For example: `Q123`, `Q972`
    """
    try:
        WikibaseID(model.name)

        return True
    except ValueError as e:
        if "is not a valid Wikibase ID" in str(e):
            return False

        raise


def get_relevant_model_version(
    model: ArtifactCollection, aws_env: AwsEnv
) -> Optional[list[str]]:
    """Returns the model name and version if a valid model is found"""

    if not is_concept_model(model):
        return None

    for model_artifacts in model.artifacts():
        if ("aws_env", aws_env.value) in model_artifacts.metadata.items():
            return model_artifacts.name


@app.command()
def get_all_available_classifiers(
    aws_envs: list[AwsEnv] = [e.value for e in AwsEnv],
) -> list[str]:
    """
    Return all available models for the given environment

    Current implementation relies on the wandb sdk abstraction over
    the graphql endpoint, which queries each item as an individual
    request.
    """

    api_key = os.environ["WANDB_API_KEY"]
    api = wandb.Api(api_key=api_key)
    model_type = ArtifactType(
        api.client,
        entity=WANDB_MODEL_ORG,
        project=WANDB_MODEL_REGISTRY,
        type_name="model",
    )
    model_collections = model_type.collections()

    classifier_specs = defaultdict(list)
    for model in model_collections:
        if not is_concept_model(model):
            continue

        console.log(f"Checking for matching environments for model: {model.name}")
        for model_artifacts in model.artifacts():
            model_env = model_artifacts.metadata.get("aws_env")
            if model_env not in aws_envs:
                continue

            # Ignore older models for the same env
            env_models = [m.split(":")[0] for m in classifier_specs[model_env]]
            if model.name not in env_models:
                classifier_specs[model_env].append(model_artifacts.name)

    for aws_env, spec in classifier_specs.items():
        spec_path = build_spec_file_path(aws_env)
        write_spec_file(spec_path, data=spec)


if __name__ == "__main__":
    app()
