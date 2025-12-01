"""
Fetch performance tables from the most recent W&B runs for specified wikibase IDs.

This script downloads the 'performance' table logged during model evaluation
from the most recent training run for each wikibase ID, and exports them to CSV files.

A CSV file for all Wikibase IDs is output too.
"""

import json
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
import wandb
from rich.console import Console

from knowledge_graph.config import WANDB_ENTITY, metrics_dir

app = typer.Typer()
console = Console()


def get_most_recent_run(entity: str, wikibase_id: str):
    """
    Get the most recent W&B run for a wikibase ID.

    :param entity: W&B entity name
    :type entity: str
    :param wikibase_id: Wikibase ID (used as W&B project name)
    :type wikibase_id: str
    :return: The most recent run object, or None if no runs found
    """
    api = wandb.Api()

    try:
        # Get runs sorted by creation time (most recent first)
        runs = api.runs(
            f"{entity}/{wikibase_id}",
            filters={"jobType": "train_model"},
            order="-created_at",
        )

        if not runs:
            console.log(f"[yellow]⚠️  No runs found for {wikibase_id}[/yellow]")
            return None

        return runs[0]
    except Exception as e:
        console.log(f"[red]❌ Error fetching runs for {wikibase_id}: {e}[/red]")
        return None


def fetch_performance_table(run) -> pd.DataFrame | None:
    """
    Fetch the 'performance' table from a W&B run.

    :param run: W&B run object
    :return: DataFrame with performance metrics, or None if table not found
    """
    try:
        # Get the performance table artifact
        for artifact in run.logged_artifacts():
            if artifact.type == "run_table":
                artifact_dir = artifact.download()
                table_path = Path(artifact_dir) / "performance.table.json"

                if table_path.exists():
                    with open(table_path, "r") as f:
                        table_data = json.load(f)

                    columns = table_data.get("columns", [])
                    data = table_data.get("data", [])

                    if columns and data:
                        df = pd.DataFrame(data, columns=columns)
                        return df

        # Alternative: try to get it from run history
        history = run.scan_history(keys=["performance"])
        for row in history:
            if "performance" in row:
                table = row["performance"]
                if hasattr(table, "get_dataframe"):
                    return table.get_dataframe()

        console.log(f"[yellow]⚠️  No performance table found in run {run.name}[/yellow]")
        return None

    except Exception as e:
        console.log(
            f"[red]❌ Error fetching performance table from {run.name}: {e}[/red]"
        )
        return None


@app.command()
def main(
    wikibase_ids: Annotated[
        list[str],
        typer.Argument(help="List of wikibase IDs to fetch performance tables from"),
    ],
    combined_name: Annotated[
        str | None,
        typer.Option(
            help="Name for the combined CSV file (without .csv extension). If not provided, uses concatenated wikibase IDs."
        ),
    ] = None,
):
    """
    Fetch performance tables from the most recent W&B training runs.

    Downloads the 'performance' table from the most recent training run
    for each specified wikibase ID and saves them as CSV files to data/processed/classifiers_performance.

    Example usage:
        uv run python scripts/fetch_performance_tables.py Q123 Q456 Q789

        uv run python scripts/fetch_performance_tables.py Q123 Q456 --combined-name adaptation
    """
    output_dir = metrics_dir
    console.log(f"🔍 Fetching performance tables for {len(wikibase_ids)} wikibase IDs")

    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    all_dfs = []

    for wikibase_id in wikibase_ids:
        console.log(f"\n📊 Processing wikibase ID: {wikibase_id}")

        run = get_most_recent_run(entity=WANDB_ENTITY, wikibase_id=wikibase_id)

        if not run:
            results.append(
                {"wikibase_id": wikibase_id, "status": "no_runs_found", "path": None}
            )
            continue

        console.log(f"  ✓ Found run: {run.name} (created: {run.created_at})")
        console.log(f"  🔗 Run URL: {run.url}")

        df = fetch_performance_table(run)

        if df is None:
            results.append(
                {"wikibase_id": wikibase_id, "status": "no_table_found", "path": None}
            )
            continue

        # Add wikibase_id column to identify source
        df.insert(0, "wikibase_id", wikibase_id)

        output_path = output_dir / f"{wikibase_id}_performance.csv"
        df.to_csv(output_path, index=False)
        console.log(f"  ✅ Saved to: {output_path}")

        results.append(
            {"wikibase_id": wikibase_id, "status": "success", "path": output_path}
        )
        all_dfs.append(df)

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)

        if combined_name:
            filename = f"{combined_name}_performance.csv"
        else:
            concatenated_ids = "-".join(wikibase_ids)
            filename = f"{concatenated_ids}_performance.csv"

        combined_path = output_dir / filename
        combined_df.to_csv(combined_path, index=False)
        console.log(f"\n📋 Combined CSV saved to: {combined_path}")

    # Print summary
    console.log("\n" + "=" * 60)
    console.log("📊 Summary:")
    success_count = sum(1 for r in results if r["status"] == "success")
    console.log(f"  ✅ Successfully fetched: {success_count}/{len(wikibase_ids)}")

    if success_count < len(wikibase_ids):
        console.log("\n  Failed wikibase IDs:")
        for r in results:
            if r["status"] != "success":
                console.log(f"    • {r['wikibase_id']}: {r['status']}")


if __name__ == "__main__":
    app()
