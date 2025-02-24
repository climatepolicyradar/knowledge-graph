import os
import shutil
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import track

import argilla as rg
from static_sites.labelling_librarian.checks import (
    all_dataset_level_checks,
    check_whether_span_border_is_in_word,
    check_whether_spans_are_long,
    check_whether_spans_have_high_non_alphabetical_ratio,
)
from static_sites.labelling_librarian.template import (
    create_dataset_page,
    create_index_page,
)

load_dotenv()

app = typer.Typer()
console = Console()

# Get the directory where this file lives
current_dir = Path(__file__).parent.resolve()


@app.command()
def main():
    client = rg.Argilla(
        api_url=os.getenv("ARGILLA_API_URL"),
        api_key=os.getenv("ARGILLA_API_KEY"),
    )
    workspace = client.workspaces("knowledge-graph")
    assert workspace is not None
    with console.status("Fetching datasets from argilla"):
        datasets: list[rg.Dataset] = workspace.datasets
    console.log(f"Fetched {len(datasets)} datasets")

    issues = []

    for check in [
        check_whether_spans_are_long,
        all_dataset_level_checks,
        check_whether_span_border_is_in_word,
        check_whether_spans_have_high_non_alphabetical_ratio,
    ]:
        for dataset in track(datasets[:4], "Checking datasets for issues..."):
            issues.extend(check(dataset))

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

    # Generate and save individual dataset pages
    for dataset in track(
        datasets, description="Generating dataset pages", transient=True
    ):
        dataset_name = dataset.name  # type: ignore
        relevant_issues = [
            issue for issue in issues if issue.dataset_name == dataset_name
        ]
        html_content = create_dataset_page(
            dataset_name=dataset_name, issues=relevant_issues
        )
        output_path = output_dir / f"{dataset_name}.html"
        output_path.write_text(html_content)
    console.log(f"Generated {len(datasets)} dataset pages")


if __name__ == "__main__":
    typer.run(main)
