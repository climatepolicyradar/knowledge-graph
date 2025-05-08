import os
from collections import defaultdict
from pathlib import Path

import typer
import yaml  # type: ignore
from dotenv import load_dotenv
from rich.console import Console

import wandb
from scripts.cloud import AwsEnv, ClassifierSpec
from src.identifiers import WikibaseID
from wandb.apis.public.artifacts import ArtifactCollection, ArtifactType

load_dotenv()

console = Console()

app = typer.Typer()


WANDB_MODEL_ORG = "climatepolicyradar_UZODYJSN66HCQ"
WANDB_MODEL_REGISTRY = "wandb-registry-model"
SPEC_DIR = Path("flows") / "classifier_specs"


def build_spec_file_path(aws_env: AwsEnv) -> Path:
    file_path = SPEC_DIR / f"{aws_env}.yaml"
    return file_path


def read_spec_file(aws_env: AwsEnv) -> list[str]:
    file_path = build_spec_file_path(aws_env)
    with open(file_path, "r") as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def parse_spec_file(aws_env: AwsEnv) -> list[ClassifierSpec]:
    contents = read_spec_file(aws_env)
    classifier_specs: list[ClassifierSpec] = []
    for item in contents:
        try:
            name, alias = item.split(":")
            classifier_specs.append(ClassifierSpec(name=name, alias=alias))
        except ValueError:
            raise ValueError(f"Invalid format in spec file: {item}")

    return classifier_specs


def write_spec_file(file_path: Path, data: list[ClassifierSpec]):
    """Save a classifier spec YAML"""
    serialised_data = list(map(lambda spec: f"{spec.name}:{spec.alias}", data))
    with open(file_path, "w") as file:
        yaml.dump(serialised_data, file, explicit_start=True)


def is_concept_model(model_artifact) -> bool:
    """
    Check if a model is a concept classifier

    This check is based on whether the model name can be instantiated as a WikibaseID.
    For example: `Q123`, `Q972`
    """
    try:
        WikibaseID(model_artifact.name)  # type: ignore

        return True
    except ValueError as e:
        if "is not a valid Wikibase ID" in str(e):
            return False

        raise


def get_relevant_model_version(
    model: ArtifactCollection, aws_env: AwsEnv
) -> list[str] | None:
    """Returns the model name and version if a valid model is found"""
    if not is_concept_model(model):
        return None

    for model_artifacts in model.artifacts():
        if ("aws_env", aws_env.value) in model_artifacts.metadata.items():
            return model_artifacts.name


def is_latest_model_in_env(
    classifier_specs: list[ClassifierSpec], model_name: str
) -> bool:
    """Check to see if this model already has a version found for the env"""
    env_models = [m.name for m in classifier_specs]
    return model_name not in env_models


def sort_specs(specs: list[ClassifierSpec]) -> list[ClassifierSpec]:
    # Have stable ordering. First, the name is gotten from
    # `Q000:v0`, and then the number part is gotten from `Q000`.
    return sorted(specs, key=lambda x: x.name)


@app.command()
def get_all_available_classifiers(
    aws_envs: list[AwsEnv] | None = None,
    api_key: str | None = None,
) -> None:
    """
    Return all available models for the given environment

    Current implementation relies on the wandb sdk abstraction over
    the graphql endpoint, which queries each item as an individual
    request.
    """
    if not aws_envs:
        aws_envs = [e for e in AwsEnv]

    console.log(
        f"Running for AWS environments: {[aws_env.value for aws_env in aws_envs]}"
    )

    if not api_key:
        api_key = os.environ["WANDB_API_KEY"]
    api = wandb.Api(api_key=api_key)  # type: ignore
    model_type = ArtifactType(
        api.client,
        entity=WANDB_MODEL_ORG,
        project=WANDB_MODEL_REGISTRY,
        type_name="model",
    )
    model_collections = model_type.collections()

    console.log("Checking for matching environments for model")
    classifier_specs: dict[AwsEnv, list[ClassifierSpec]] = defaultdict(list)
    for model in model_collections:
        if not is_concept_model(model):
            continue

        console.log(model.name)
        for model_artifacts in model.artifacts():
            model_env = AwsEnv(model_artifacts.metadata.get("aws_env"))
            if model_env not in aws_envs:
                continue

            if is_latest_model_in_env(classifier_specs[model_env], model.name):
                model_parts: list[str] = model_artifacts.name.split(":")
                if len(model_parts) != 2:
                    raise ValueError(
                        f"Model name had unexpected format: {model_artifacts.name}"
                    )
                classifier_specs[model_env].append(
                    ClassifierSpec(
                        name=model_parts[0],
                        alias=model_parts[1],
                    )
                )

    for aws_env, specs in classifier_specs.items():
        aws_env = AwsEnv(aws_env)
        spec_path = build_spec_file_path(aws_env)
        sorted_specs = sort_specs(specs)
        write_spec_file(spec_path, data=sorted_specs)

    console.log("Finished!")


if __name__ == "__main__":
    app()
