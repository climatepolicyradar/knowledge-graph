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

Run the script as follows:
    COLUMNS=1000 poetry run python scripts/generate_report.py > report.md
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

# replace with the list of wikibase IDs that you want to evaluate
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

    # Display the passages from each outlier group
    for _, row in outlier_groups.iterrows():
        group, value = row["Group"].split(": ")
        print(f"\n### Passages from {group}: {value}\n")

        passages = []
        for passage in concept.labelled_passages:
            metadata = passage.metadata
            if group in metadata and metadata[group] == value:
                passages.append(passage)

        print(f"Found {len(passages)} passages in this group:\n")

        for i, passage in enumerate(passages):
            print(f"{i+1}.\tGround truth:")
            # Format labelled passages with markdown bolding instead of the rich colour codes
            highlighted_ground_truth = (
                passage.get_highlighted_text()
                .replace("[cyan]", "**")
                .replace("[/cyan]", "**")
            )
            print(f"\t{highlighted_ground_truth}\n")

            prediction = LabelledPassage(
                text=passage.text, spans=classifier.predict(passage.text)
            )
            highlighted_prediction = (
                prediction.get_highlighted_text()
                .replace("[cyan]", "**")
                .replace("[/cyan]", "**")
            )
            print("\tPrediction:")
            print(f"\t{highlighted_prediction}\n")
