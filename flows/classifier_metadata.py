"""
Reusable logic for editing classifier metadata in W&B.

These functions depend on `flows.classifier_specs` types, so they live in `flows/`
rather than `knowledge_graph/operations/` (operations must not import from flows). The
`scripts/classifier_metadata.py` CLI is a thin Typer wrapper around `update` and
`update_entire_env`.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

import wandb

from flows.classifier_specs.spec_interface import DontRunOnEnum, load_classifier_specs
from flows.update_classifier_spec import refresh_all_available_classifiers
from knowledge_graph.classifier import ModelPath
from knowledge_graph.cloud import AwsEnv, ComputeEnvironment
from knowledge_graph.config import WANDB_ENTITY
from knowledge_graph.identifiers import ClassifierID, WikibaseID
from knowledge_graph.utils import get_logger
from knowledge_graph.version import get_latest_model_version

log = get_logger(__name__)

REGISTRY_NAME = "model"
JOB_TYPE = "configure_model"


def update_entire_env(
    clear_dont_run_on: bool = False,
    add_dont_run_on: list[DontRunOnEnum] | None = None,
    clear_require_gpu: bool = False,
    add_require_gpu: bool = False,
    aws_env: AwsEnv = AwsEnv.labs,
    update_specs: bool = True,
    max_workers: int = 8,
):
    """Update classifier metadata for every classifier in an envs spec."""
    specs = load_classifier_specs(aws_env)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for spec in specs:
            future = executor.submit(
                update,
                wikibase_id=spec.wikibase_id,
                classifier_id=spec.classifier_id,
                clear_dont_run_on=clear_dont_run_on,
                add_dont_run_on=add_dont_run_on,
                clear_require_gpu=clear_require_gpu,
                add_require_gpu=add_require_gpu,
                aws_env=aws_env,
                update_specs=False,  # since we'll only update once all are done
            )
            futures.append(future)

        for future in as_completed(futures):
            future.result()

    if update_specs:
        refresh_all_available_classifiers([aws_env])


def update(
    wikibase_id: WikibaseID,
    classifier_id: ClassifierID,
    clear_dont_run_on: bool = False,
    add_dont_run_on: list[DontRunOnEnum] | None = None,
    clear_require_gpu: bool = False,
    add_require_gpu: bool = False,
    aws_env: AwsEnv = AwsEnv.labs,
    add_classifiers_profiles: list[str] | None = None,
    remove_classifiers_profiles: list[str] | None = None,
    update_specs: bool = True,
):
    """
    Updates the metadata for a classifier to determine behaviour at inference time.

    Set `dont_run_on` key to prevent a classifier from running on documents from a
    specific source. You can either add to existing items by passing
    `--add-dont-run-on <source>` or clear all items with `--clear-dont-run-on`
    use both to reset the list down to a new selection, eg:
    ```
    ... --clear-dont-run-on --add-dont-run-on gef --add-dont-run-on sabin
    ```

    Raises ``ValueError`` on invalid input; callers facing a CLI should convert this
    into the appropriate user-facing error.
    """
    model_path = ModelPath(wikibase_id=wikibase_id, classifier_id=classifier_id)

    api = wandb.Api()
    artifacts = api.artifacts(type_name="model", name=f"{model_path}")
    classifier_version = get_latest_model_version(artifacts, aws_env)

    artifact_id = f"{model_path}:{classifier_version}"

    with wandb.init(entity=WANDB_ENTITY, project=wikibase_id, job_type=JOB_TYPE) as run:
        artifact: wandb.Artifact = run.use_artifact(artifact_id)

        if clear_dont_run_on:
            log.info(f"Clearing existing `dont_run_on` metadata from {artifact_id}")
            if add_dont_run_on:
                log.info("`add_dont_run_on` is set, so values will be fully refreshed")

            artifact.metadata.pop("dont_run_on", None)

        if add_dont_run_on:
            additions: list[str] = [a.value for a in add_dont_run_on]
            log.info(f"Applying {additions=} to {artifact_id}")
            current: list[str] = artifact.metadata.get("dont_run_on", [])
            update: list[str] = list(set(current + additions))
            artifact.metadata["dont_run_on"] = update

        if clear_require_gpu:
            if add_require_gpu:
                raise ValueError(
                    "`clear-require-gpu` and `add-require-gpu` can't both be set"
                )

            if artifact.metadata.get("compute_environment"):
                artifact.metadata["compute_environment"].pop("gpu", None)

                # Remove if now empty
                if not artifact.metadata.get("compute_environment"):
                    artifact.metadata.pop("compute_environment", None)

        elif add_require_gpu:
            compute_environment: ComputeEnvironment = artifact.metadata.get(
                "compute_environment", {}
            )
            compute_environment: ComputeEnvironment = compute_environment | {
                "gpu": True
            }
            artifact.metadata["compute_environment"] = compute_environment

        if add_classifiers_profiles or remove_classifiers_profiles:
            add_class_prof: set[str] = (
                set(add_classifiers_profiles) if add_classifiers_profiles else set()
            )
            remove_class_prof: set[str] = (
                set(remove_classifiers_profiles)
                if remove_classifiers_profiles
                else set()
            )

            if dupes := add_class_prof & remove_class_prof:
                raise ValueError(
                    f"duplicate values found for adding and removing classifiers profiles: `{','.join(dupes)}`"
                )

            current_class_prof: set[str] = set(
                artifact.metadata.get("classifiers_profiles", [])
            )

            if (
                classifiers_profiles := (current_class_prof | add_class_prof)
                - remove_class_prof
            ):
                if len(classifiers_profiles) > 1:
                    raise ValueError(
                        f"Artifact must have maximum of one classifiers profile in metadata, or you must specify 1 to remove. Current classifiers profiles `{current_class_prof}`"
                    )
                else:
                    artifact.metadata["classifiers_profiles"] = classifiers_profiles
            else:
                artifact.metadata.pop("classifiers_profiles", None)

        artifact.save()

    if update_specs:
        refresh_all_available_classifiers([aws_env])
