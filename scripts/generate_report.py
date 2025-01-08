"""
Generate a markdown report of classifier performance

The script generates a markdown report of classifier performance, beginning with a
table of performance metrics for each concept, and a check on the consistency of those
metrics across strata within the dataset. It then pulls out the outlier groups and
displays example human and machine-labelled passages from each, to help the user
understand where/why the classifier is performing inconsistently.

NB inconsistent performance on a group does not necessarily mean that the classifier is
performing poorly! The classifier might perform _better_ on that group, or the group
may be too small to generate reliable metrics. These numbers and symbols should be
interpreted with caution!
"""

from pathlib import Path

import pandas as pd
import typer
from rich import box
from rich.console import Console
from rich.table import Table
from typing_extensions import Annotated

from scripts.config import classifier_dir, concept_dir, metrics_dir
from src.classifier import Classifier
from src.concept import Concept
from src.labelled_passage import LabelledPassage

console = Console(highlight=False)

app = typer.Typer()


@app.command()
def main(
    wikibase_ids: Annotated[
        str,
        typer.Option(
            ...,
            help="Comma-separated list of Wikibase IDs to evaluate",
        ),
    ],
    output_file: Path = typer.Option(
        Path("data/report.md"),
        help="Path to the output file",
    ),
):
    # Convert comma-separated string to list
    wikibase_ids_list = [id.strip() for id in wikibase_ids.split(",")]

    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        console = Console(file=f, width=150)

        data = []
        for wikibase_id in wikibase_ids_list:
            concept = Concept.load(concept_dir / f"{wikibase_id}.json")
            performance_data_path = metrics_dir / f"{wikibase_id}.json"
            df = pd.read_json(performance_data_path)
            passage_level_precision = df.loc[
                (df["Group"] == "all") & (df["Agreement at"] == "Passage level"),
                "Precision",
            ].values[0]
            span_level_precision = df.loc[
                (df["Group"] == "all") & (df["Agreement at"] == "Span level (0)"),
                "Precision",
            ].values[0]

            span_level_precision_std = df.loc[
                (df["Agreement at"] == "Span level (0)"), "Precision"
            ].values.std()

            data.append(
                {
                    "Wikibase ID": wikibase_id,
                    "Preferred label": concept.preferred_label,
                    "Passage-level precision": f"{passage_level_precision:.2f}",
                    "Span-level precision": f"{span_level_precision:.2f}",
                    "Span-level precision standard deviation": f"{span_level_precision_std:.2f}",
                    "Equity strata consistency": "✅"
                    if span_level_precision_std < 0.1
                    else "❌",
                }
            )

        # Create and print the performance metrics table
        output_data = pd.DataFrame(data)
        table = Table(box=box.MARKDOWN)
        for column in output_data.columns:
            table.add_column(column)
        for _, row in output_data.iterrows():
            table.add_row(*[str(x) for x in row])
        console.print(table)

        # Find concepts that failed the consistency check
        inconsistent_concepts = output_data.loc[
            output_data["Equity strata consistency"] == "❌"
        ]["Wikibase ID"].values

        # For each inconsistent concept, find and display outlier groups
        for wikibase_id in inconsistent_concepts:
            concept = Concept.load(concept_dir / f"{wikibase_id}.json")
            classifier = Classifier.load(classifier_dir / wikibase_id)
            performance_data_path = metrics_dir / f"{wikibase_id}.json"
            df = pd.read_json(performance_data_path)
            df = df[df["Agreement at"] == "Span level (0)"]

            span_level_precision_values = df["Precision"].values

            # Find the groups which are outliers
            outlier_groups = df.loc[
                (
                    df["Precision"]
                    > (
                        span_level_precision_values.mean()
                        + span_level_precision_values.std()
                    )
                )
                | (
                    df["Precision"]
                    < (
                        span_level_precision_values.mean()
                        - span_level_precision_values.std()
                    )
                )
            ]

            console.print(f"\n## Outlier groups for {concept}\n")
            console.print(
                f"Overall performance: {df.loc[df['Group'] == 'all', 'Precision'].values[0]:.2f}\n"
            )

            # Create and print the outlier groups table
            table = Table(box=box.MARKDOWN)
            table.add_column("Group")
            table.add_column("Value")
            table.add_column("Number of passages")
            table.add_column("Performance")

            for _, row in outlier_groups.iterrows():
                group, value = row["Group"].split(": ")
                n_passages = len(
                    [
                        passage
                        for passage in concept.labelled_passages
                        if group in passage.metadata
                        and passage.metadata[group] == value
                    ]
                )
                table.add_row(
                    group,
                    value,
                    str(n_passages),
                    f"{row['Precision']:.2f}",
                )
            console.print(table)

            # Display example passages from each outlier group
            for _, row in outlier_groups.iterrows():
                group, value = row["Group"].split(": ")
                console.print(f"\n### Examples from {group}: {value}\n")

                passages = []
                for passage in concept.labelled_passages:
                    metadata = passage.metadata
                    if group in metadata and metadata[group] == value:
                        passages.append(passage)

                console.print(f"Found {len(passages)} passages in this group:\n")

                for i, passage in enumerate(passages[:3]):  # Show first 3 examples
                    console.print(f"{i+1}. Ground truth:")
                    # Format ground truth with markdown bold
                    text = passage.text
                    for span in passage.spans:
                        text = (
                            text[: span.start_index]
                            + f"**{text[span.start_index:span.end_index]}**"
                            + text[span.end_index :]
                        )
                    console.print(f"   {text}\n")

                    # Format predictions with markdown bold
                    prediction = LabelledPassage(
                        text=passage.text, spans=classifier.predict(passage.text)
                    )
                    console.print("   Prediction:")
                    text = prediction.text
                    for span in prediction.spans:
                        text = (
                            text[: span.start_index]
                            + f"**{text[span.start_index:span.end_index]}**"
                            + text[span.end_index :]
                        )
                    console.print(f"   {text}\n")


if __name__ == "__main__":
    app()
