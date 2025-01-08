"""
Generate a markdown report of classifier performance

The script will output a table of performance metrics for each concept, followed by a
table of outlier groups for each concept that failed the consistency check.
"""

import pandas as pd
from rich import box
from rich.console import Console
from rich.table import Table

from scripts.config import classifier_dir, concept_dir, metrics_dir
from src.classifier import Classifier
from src.concept import Concept
from src.labelled_passage import LabelledPassage

console = Console(highlight=False)

wikibase_ids = [
    "Q676",
    "Q684",
    "Q690",
    "Q695",
    "Q701",
    "Q704",
    "Q708",
    "Q1016",
    "Q1160",
    "Q1167",
]
data = []
for wikibase_id in wikibase_ids:
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

    span_level_recall = df.loc[
        (df["Group"] == "all") & (df["Agreement at"] == "Span level (0)"),
        "Recall",
    ].values[0]

    span_level_f1 = df.loc[
        (df["Group"] == "all") & (df["Agreement at"] == "Span level (0)"),
        "F1 score",
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
            > (span_level_precision_values.mean() + span_level_precision_values.std())
        )
        | (
            df["Precision"]
            < (span_level_precision_values.mean() - span_level_precision_values.std())
        )
    ]

    print(f"\n## Outlier groups for {concept}\n")
    print(
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
                if group in passage.metadata and passage.metadata[group] == value
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
        print(f"\n### Examples from {group}: {value}\n")

        passages = []
        for passage in concept.labelled_passages:
            metadata = passage.metadata
            if group in metadata and metadata[group] == value:
                passages.append(passage)

        print(f"Found {len(passages)} passages in this group:\n")

        for i, passage in enumerate(passages[:3]):  # Show first 3 examples
            print(f"{i+1}. Ground truth:")
            # Format ground truth with markdown bold
            text = passage.text
            for span in passage.spans:
                text = (
                    text[: span.start_index]
                    + f"**{text[span.start_index:span.end_index]}**"
                    + text[span.end_index :]
                )
            print(f"   {text}\n")

            # Format predictions with markdown bold
            prediction = LabelledPassage(
                text=passage.text, spans=classifier.predict(passage.text)
            )
            highlighted_prediction = (
                prediction.get_highlighted_text()
                .replace("[cyan]", "**")
                .replace("[/cyan]", "**")
            )
            print("   Prediction:")
            print(f"   {highlighted_prediction}\n")
