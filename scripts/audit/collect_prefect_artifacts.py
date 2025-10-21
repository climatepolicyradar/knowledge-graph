import asyncio
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Annotated, Any, Callable

import typer
from prefect.client.orchestration import PrefectClient, get_client
from prefect.client.schemas.filters import (
    ArtifactFilter,
    ArtifactFilterFlowRunId,
    FlowRunFilter,
    FlowRunFilterName,
    FlowRunFilterParentFlowRunId,
)
from prefect.client.schemas.objects import Artifact, FlowRun
from rich import box
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()


def _create_key(artifact: Artifact, row_of_data: dict[str, str]) -> str:
    """
    Accepts an artifact and creates a unique key

    First attempt uses the description, failing this will fall back to the column keys.
    """
    if desc := artifact.description:
        # Return first line if its there
        for part in desc.split("\n"):
            return "_".join(part.lstrip("# ").split()).lower()

    # fallback to building from data headers
    headers = row_of_data.keys()
    joined_headers = "_".join(headers)
    cleaned_headers = "".join(joined_headers.split()).lower()
    return cleaned_headers


async def _paginate_prefect_read(fn: Callable, kwargs: dict[str, Any]) -> list[Any]:
    """
    Dynamic wrapper for prefect client interactions that adds unlimited paginations.

    Can be used to get all results for a given client get request function & kwargs
    for the request.
    """
    page_size = 200
    offset = 0
    results = []
    while True:
        page_of_results = await fn(**kwargs, limit=page_size, offset=offset)
        results.extend(page_of_results)
        console.log(f"Collected '{len(page_of_results)}' results for {fn.__name__}")

        if len(page_of_results) < page_size:
            break
        else:
            offset += page_size
            continue

    return results


async def flow_run_from_name(client: PrefectClient, flow_run_name: str) -> FlowRun:
    """
    Looks up the flow run using a flow run name

    The run names are more user friendly and easier to find than ids, but ids are used in
    other client requests.
    """
    flow_runs = await client.read_flow_runs(
        flow_run_filter=FlowRunFilter(name=FlowRunFilterName(any_=[flow_run_name]))
    )
    if not flow_runs:
        raise ValueError(f"No flow run found with name: {flow_run_name}")
    else:
        return flow_runs[0]


async def collect_subflows(
    client: PrefectClient, flow_runs: list[FlowRun]
) -> list[FlowRun]:
    """
    Given a parent flow id return all the sub flow ids across all children.

    This is recursive so will include subflows of subflows, etc.
    """
    all_subflows = []
    current_level = flow_runs

    while current_level:
        kwargs = {
            "flow_run_filter": FlowRunFilter(
                parent_flow_run_id=FlowRunFilterParentFlowRunId(
                    any_=[fr.id for fr in current_level]
                )
            )
        }
        subflows = await _paginate_prefect_read(fn=client.read_flow_runs, kwargs=kwargs)

        if not subflows:
            break
        else:
            all_subflows.extend(subflows)
            current_level = subflows
            continue

    return all_subflows


async def artifacts_from_run_ids(
    client: PrefectClient, flow_runs: list[FlowRun]
) -> list[Artifact]:
    """Returns all artifacts associated with a list of flow run ids"""
    kwargs = {
        "artifact_filter": ArtifactFilter(
            flow_run_id=ArtifactFilterFlowRunId(any_=[fr.id for fr in flow_runs])
        )
    }
    artifacts = await _paginate_prefect_read(fn=client.read_artifacts, kwargs=kwargs)
    return artifacts


def display_run_results(flow_runs):
    results = {
        k: str(v) for k, v in Counter([fr.state_name for fr in flow_runs]).items()
    }
    table = Table(box=box.ROUNDED)

    for state in results.keys():
        table.add_column(state)
    table.add_row(*list(results.values()))

    console.print(table)


async def run(
    flow_run_name: str,
    download_dir: Path,
    include_sub_flows: bool,
    artifact_types_to_print: list[str],
    success_summary: bool,
):
    async with get_client() as client:
        flow_run = await flow_run_from_name(client, flow_run_name)

        flow_runs = [flow_run]
        if include_sub_flows:
            flow_runs.extend(await collect_subflows(client, flow_runs))

        if success_summary:
            display_run_results(flow_runs)

        artifacts = await artifacts_from_run_ids(client, flow_runs)
        console.log(
            f"Found {len(artifacts)} artifacts across {len(flow_runs)} flows/subflows"
        )

        artifact_type_grouping = defaultdict(list)
        for a in artifacts:
            if a.type in artifact_types_to_print:
                console.log(a.description)

            match a.type:
                case "table":
                    data = json.loads(a.data)  # pyright: ignore[reportArgumentType]
                    if data:
                        key = _create_key(a, data)
                        artifact_type_grouping[key].extend(data)
                case "progress" | "markdown":
                    pass
                case _:
                    raise ValueError(f"Unsupported artifact type {a.type}")

        download_dir.mkdir(parents=True, exist_ok=True)
        for key, table in artifact_type_grouping.items():
            path = download_dir / f"{key}.json"
            with open(path, "w") as f:
                f.write(json.dumps(table, indent=2))
            console.print(f"Written table to `{path}`")


@app.command()
def main(
    flow_run_name: Annotated[
        str, typer.Argument(..., help="The key for a given artifact to download")
    ],
    download_dir: Annotated[
        Path, typer.Option(..., help="The path to store artifact table data in")
    ] = Path("data") / "audit" / "prefect_artifacts",
    include_sub_flows: Annotated[
        bool,
        typer.Option(
            ...,
            help="Treat the run as a parent flow and collect all the artifacts of any associated subflows",
        ),
    ] = True,
    success_summary: Annotated[
        bool,
        typer.Option(
            ...,
            help="Print out success/failure count for flows",
        ),
    ] = True,
    artifact_types_to_print: Annotated[
        str,
        typer.Option(
            ...,
            help="The types of artifact to print, defaults to all, pass empty to print nothing",
        ),
    ] = "progress table markdown links images",
):
    """
    Inspect prefect artifacts.

    Requires cli auth, and a run name for a flow. This will write out descriptions to
    the cli and will group table data before writing it into a subfolder for further
    investigation.

    This can be used to get artifacts for individual runs, as well as runs that have
    sub-flows (and subflows of subflows).
    """
    asyncio.run(
        run(
            flow_run_name=flow_run_name,
            download_dir=(download_dir / flow_run_name),
            success_summary=success_summary,
            include_sub_flows=include_sub_flows,
            artifact_types_to_print=artifact_types_to_print.split(),
        )
    )


if __name__ == "__main__":
    app()
