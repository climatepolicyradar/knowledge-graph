import os
import pickle

import argilla as rg
from rich.console import Console

from scripts.config import data_dir
from src.identifiers import WikibaseID

console = Console()

source_workspace_name = "impacted-groups-relabel"
target_workspace_name = "knowledge-graph"

rg.init(api_key=os.getenv("ARGILLA_API_KEY"), api_url=os.getenv("ARGILLA_API_URL"))

with console.status("📚 Getting datasets from source workspace..."):
    source_datasets = rg.list_datasets(workspace=source_workspace_name)
console.print(
    f'✅ Found {len(source_datasets)} datasets in the "{source_workspace_name}" workspace:'
)
console.print([dataset.name for dataset in source_datasets])


with console.status("🔍 Checking for source workspace..."):
    try:
        source_workspace = rg.Workspace.from_name(name=source_workspace_name)
        console.print(f"✅ Found source workspace: {source_workspace.name}")
    except ValueError:
        raise ValueError(f"Source workspace {source_workspace_name} not found.")

with console.status("🔍 Checking for target workspace..."):
    try:
        target_workspace = rg.Workspace.from_name(name=target_workspace_name)
        console.print(f"✅ Found target workspace: {target_workspace.name}")
    except ValueError:
        target_workspace = rg.Workspace.create(name=target_workspace_name)
        console.print(f"⚡ Created target workspace: {target_workspace.name}")

# copy each dataset from the source workspace to the target workspace
for source_dataset in source_datasets:
    try:
        wikibase_id = WikibaseID(source_dataset.name.split("-")[-1])
    except ValueError:
        raise ValueError(
            f'Dataset name should end with a Wikibase ID. Got "{source_dataset.name}"'
        )

    local_dataset = source_dataset.pull()
    # save a local copy of the dataset in the /data/raw/argilla directory
    argilla_dir = data_dir / "raw" / "argilla"
    argilla_dir.mkdir(parents=True, exist_ok=True)
    with open(argilla_dir / f"{source_dataset.name}.pkl", "wb") as f:
        pickle.dump(local_dataset, f)

    # check whether we've lost anything in the process of pickling and unpickling
    with open(argilla_dir / f"{source_dataset.name}.pkl", "rb") as f:
        dataset_loaded_from_pickle = pickle.load(f)
        assert isinstance(dataset_loaded_from_pickle, rg.FeedbackDataset)
        assert len(dataset_loaded_from_pickle.records) == len(source_dataset.records)

    # check whether the dataset already exists in the target workspace. If it does,
    # we'll append the records from the local copy of the dataset to the local copy of
    # the target dataset before pushing it back to argilla. Otherwise, we'll just use
    # the local copy of the source dataset.
    existing_target_dataset = None
    local_existing_target_dataset = None
    try:
        existing_target_dataset = rg.FeedbackDataset.from_argilla(
            name=wikibase_id, workspace=target_workspace.name
        )
        local_existing_target_dataset = existing_target_dataset.pull()
        console.print(
            f'✅ Found existing target dataset: "{existing_target_dataset.name}"'
            f'in the "{target_workspace.name}" workspace'
        )
    except ValueError:
        pass

    if local_existing_target_dataset:
        with open(argilla_dir / f"{source_dataset.name}.pkl", "wb") as f:
            pickle.dump(local_existing_target_dataset, f)

        # append the records from the local copy of the dataset to the local copy of the
        # target dataset
        local_dataset.records.extend(local_existing_target_dataset.records)

        assert len(local_dataset.records) == len(source_dataset.records) + len(
            local_existing_target_dataset.records
        )

    # push a renamed version of the dataset to the target workspace with only the
    # wikibase_id (no preferred label prefix), so that the name will remain
    # consistent even if the preferred label of the concept is updated in wikibase
    if existing_target_dataset:
        # delete the existing target dataset
        existing_target_dataset.delete()

    local_dataset.push_to_argilla(
        name=wikibase_id, workspace=target_workspace.name, show_progress=False
    )
    console.print(
        f'✅ Copied "{source_dataset.name}" to the "{target_workspace.name}" workspace '
        f'with the new name "{wikibase_id}".'
    )

    # make sure that all of the users from the source dataset are are also added to
    # the target workspace
    unique_user_ids_in_the_remote_dataset = set(
        [
            response.user_id
            for record in source_dataset.records
            for response in record.responses
        ]
    )

    for user_id in unique_user_ids_in_the_remote_dataset:
        user = rg.User.from_id(user_id)

        try:
            target_workspace.add_user(user.id)
            console.print(
                f'✅ Added {user.username} to the "{target_workspace.name}" workspace.'
            )
        except ValueError:
            console.print(
                f'👌 {user.username} already has access to the "{target_workspace.name}" workspace.'
            )

    # with console.status(f"🚮 Deleting dataset {source_dataset.name} from source workspace..."):
    #     rg.FeedbackDataset.from_argilla(name=source_dataset.name, workspace=source_workspace_name).delete()
    # console.print(f"🚮 Deleted {source_dataset.name} from source workspace.")
