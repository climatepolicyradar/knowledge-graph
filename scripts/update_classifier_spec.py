from collections import defaultdict
from pathlib import Path

import typer
import wandb
import yaml  # type: ignore
from dotenv import load_dotenv
from rich.console import Console

from flows.classifier_specs.spec_interface import (
    ClassifierSpec,
    DontRunOnEnum,
    determine_spec_file_path,
)
from knowledge_graph.cloud import AwsEnv
from knowledge_graph.identifiers import WikibaseID

load_dotenv()

console = Console()

app = typer.Typer()


WANDB_MODEL_ORG = "climatepolicyradar_UZODYJSN66HCQ"
WANDB_MODEL_REGISTRY = "wandb-registry-model"


def write_spec_file(file_path: Path, data: list[ClassifierSpec]):
    """Save a classifier spec YAML"""
    serialised_data = [d.model_dump(exclude_none=True, mode="json") for d in data]

    # Reorder to put the Wikibase ID first in each spec
    ordered_data = []
    for spec in serialised_data:
        ordered_spec = {"wikibase_id": spec["wikibase_id"]}
        ordered_spec.update({k: v for k, v in spec.items() if k != "wikibase_id"})
        ordered_data.append(ordered_spec)

    with open(file_path, "w") as file:
        yaml.dump(ordered_data, file, explicit_start=True, sort_keys=False)


def sort_specs(specs: list[ClassifierSpec]) -> list[ClassifierSpec]:
    """Order the classifier spec items consistently"""
    return sorted(
        specs,
        key=lambda spec: (WikibaseID(spec.wikibase_id).numeric, spec.classifier_id),
    )


@app.command()
def refresh_all_available_classifiers(aws_envs: list[AwsEnv] | None = None) -> None:
    """Refreshes the classifier specs with the latest state of wandb."""
    if not aws_envs:
        aws_envs = [e for e in AwsEnv]

    console.log(
        f"Running for AWS environments: {[aws_env.value for aws_env in aws_envs]}"
    )
    api = wandb.Api()

    registry_filters = {"name": {"$regex": "model"}}

    collection_filters = {"name": {"$regex": WikibaseID.regex}}

    version_filters = {"$or": [{"alias": env} for env in aws_envs]}

    artifacts = (
        api.registries(filter=registry_filters)
        .collections(collection_filters)
        .versions(filter=version_filters)
    )
    specs = defaultdict(list)
    for art in artifacts:
        # Skipping old data model
        if not art.metadata.get("classifier_name"):
            continue

        env = art.metadata["aws_env"]
        wikibase_id, wandb_registry_version = art.name.split(":")
        classifier_id, wandb_project_version = art.source_name.split(":")  # noqa: F841
        WikibaseID(wikibase_id)

        spec_data = {
            "wikibase_id": wikibase_id,
            "classifier_id": classifier_id,
            "wandb_registry_version": wandb_registry_version,
        }

        if dont_run_on := art.metadata.get("dont_run_on"):
            spec_data["dont_run_on"] = [DontRunOnEnum(item) for item in dont_run_on]

        if compute_environment := art.metadata.get("compute_environment"):
            spec_data["compute_environment"] = compute_environment

        if concept_id := art.metadata.get("concept_id"):
            spec_data["concept_id"] = concept_id

        if classifiers_profiles := art.metadata.get("classifiers_profiles"):
            spec_data["classifiers_profiles"] = classifiers_profiles

        spec = ClassifierSpec(**spec_data)
        specs[env].append(spec)

    for env in aws_envs:
        spec_path = determine_spec_file_path(env)
        sorted_specs = sort_specs(specs[env])
        write_spec_file(spec_path, data=sorted_specs)

    console.log("Finished!")


if __name__ == "__main__":
    app()
