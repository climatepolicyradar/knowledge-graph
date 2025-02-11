import os
import shutil
import argilla as rg
from pathlib import Path
from typing import Set

import typer
from rich.console import Console
from rich.progress import track

from static_sites.labelling_librarian.checks import find_long_spans
from static_sites.labelling_librarian.template import (
    create_index_page,
)
from dotenv import load_dotenv

load_dotenv()

app = typer.Typer()
console = Console()

# Get the directory where this file lives
current_dir = Path(__file__).parent.resolve()


@app.command()
def main():
    rg.init(
        api_url=os.getenv("ARGILLA_API_URL"),
        api_key=os.getenv("ARGILLA_API_KEY"),
    )
    with console.status("Fetching datasets from argilla"):
        datasets: list[rg.FeedbackDataset] = rg.list_datasets(workspace="knowledge-graph")  # type: ignore
    console.log(f"Fetched {len(datasets)} datasets")

    issues = []
    for dataset in track(datasets[:3], "Checking datasets for issues..."):
        issues.extend(find_long_spans(dataset))

    console.log(f"Found {len(issues)} issues in {len(datasets)} datasets")

    # Delete and recreate the output directory
    output_dir = current_dir / "dist"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    # Generate and save the index page
    html_content = create_index_page(issues)
    output_path = output_dir / "index.html"
    output_path.write_text(html_content)
    console.log("Generated index page")


if __name__ == "__main__":
    typer.run(main)
