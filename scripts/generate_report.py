"""
Generate an HTML report of classifier performance

The script generates an HTML report of classifier performance, beginning with a
table of performance metrics for each concept, and a check on the consistency of those
metrics across a few important metadata strata within the dataset. It then pulls out
the outlier groups and displays example human and machine-labelled passages from each,
to help the user understand where/why the classifier is performing inconsistently.

NB inconsistent performance on a group does not necessarily mean that the classifier is
performing poorly! The classifier might perform _better_ on that group, or the group
may be too small to generate reliable metrics. These numbers and symbols should be
interpreted with caution!
"""

from pathlib import Path

import pandas as pd
import typer
from typing_extensions import Annotated

from scripts.config import concept_dir, metrics_dir
from scripts.evaluate import load_classifier_local
from src.concept import Concept
from src.labelled_passage import LabelledPassage

app = typer.Typer()


def dataframe_to_html_table(df: pd.DataFrame, classes: str = "") -> str:
    """Convert a pandas DataFrame to an HTML table with styling."""
    html = f'<table class="w-full text-left table-auto {classes}">'
    # Header
    html += "<thead><tr>"
    for col in df.columns:
        html += '<th class="p-4 border-b border-slate-300 dark:border-slate-600 bg-slate-50 dark:bg-gray-700">'
        html += f'<p class="block text-sm font-normal leading-none text-slate-500 dark:text-slate-400">{col}</p>'
        html += "</th>"
    html += "</tr></thead>"
    # Body
    html += '<tbody class="text-sm">'
    for _, row in df.iterrows():
        html += '<tr class="hover:bg-slate-50 dark:hover:bg-gray-700">'
        for val in row:
            html += (
                '<td class="p-4 border-b border-slate-200 dark:border-slate-600 py-5">'
            )
            html += (
                f'<p class="text-slate-800 dark:text-slate-200 break-words">{val}</p>'
            )
            html += "</td>"
        html += "</tr>"
    html += "</tbody></table>"
    return html


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
        Path("data/report.html"),
        help="Path to the output file",
    ),
):
    # Convert comma-separated string to list
    wikibase_ids_list = [id.strip() for id in wikibase_ids.split(",")]

    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Start building HTML content
    html_content = """<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>Classifier Performance Report</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <script>
            tailwind.config = {
                darkMode: "media",
            };
        </script>
    </head>
    <body class="bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 p-6">
    <div class="max-w-7xl mx-auto">
        <div class="bg-white dark:bg-gray-800 rounded-lg shadow-md p-6 mb-6">
            <h1 class="text-xl font-semibold text-slate-800 dark:text-slate-100 mb-4">
                Classifier Performance Report
            </h1>
    """

    data = []
    for wikibase_id in wikibase_ids_list:
        concept = Concept.load(concept_dir / f"{wikibase_id}.json")
        performance_data_path = metrics_dir / f"{wikibase_id}.json"
        df = pd.read_json(performance_data_path)

        # Get passage-level metrics for all data
        passage_metrics = df.loc[
            (df["Group"] == "all") & (df["Agreement at"] == "Passage level")
        ].iloc[0]

        # Calculate mean F1 across groups with enough passages
        groups_with_passages = [
            group
            for group in df["Group"].unique()
            if group != "all"
            and len(
                [
                    p
                    for p in concept.labelled_passages
                    if any(
                        group.startswith(k) and str(v) in group
                        for k, v in p.metadata.items()
                    )
                ]
            )
            >= 5
        ]

        if groups_with_passages:
            f1_scores = df.loc[
                (df["Agreement at"] == "Passage level")
                & (df["Group"].isin(groups_with_passages)),
                "F1 score",
            ].values

            f1_mean = f1_scores.mean()
            f1_std = f1_scores.std()

            # Only flag if there are groups performing significantly worse than the mean
            # AND they have enough passages
            has_poor_performing_groups = any(
                f1 < (f1_mean - f1_std)
                for group, f1 in zip(
                    df.loc[df["Agreement at"] == "Passage level", "Group"],
                    df.loc[df["Agreement at"] == "Passage level", "F1 score"],
                )
                if group in groups_with_passages
            )
        else:
            # If no groups have enough passages, we can't make a fair assessment
            has_poor_performing_groups = False

        data.append(
            {
                "Wikibase ID": wikibase_id,
                "Preferred label": concept.preferred_label,
                "Passage-level precision": f"{passage_metrics['Precision']:.2f}",
                "Passage-level recall": f"{passage_metrics['Recall']:.2f}",
                "Passage-level F1": f"{passage_metrics['F1 score']:.2f}",
                "Equity strata consistency": "❌"
                if has_poor_performing_groups
                else "✅",
            }
        )

    # Add performance metrics table
    output_data = pd.DataFrame(data)
    html_content += dataframe_to_html_table(output_data)

    # Find concepts that failed the consistency check
    inconsistent_concepts = output_data.loc[
        output_data["Equity strata consistency"] == "❌"
    ]["Wikibase ID"].values

    # For each inconsistent concept, find and display outlier groups
    for wikibase_id in inconsistent_concepts:
        concept = Concept.load(concept_dir / f"{wikibase_id}.json")
        classifier = load_classifier_local(wikibase_id)
        performance_data_path = metrics_dir / f"{wikibase_id}.json"
        df = pd.read_json(performance_data_path)
        df = df[df["Agreement at"] == "Passage level"]

        # Get passage-level metrics for all data
        passage_metrics = df.loc[df["Group"] == "all"].iloc[0]

        # Calculate mean F1 across groups with enough passages
        groups_with_passages = [
            group
            for group in df["Group"].unique()
            if group != "all"
            and len(
                [
                    p
                    for p in concept.labelled_passages
                    if any(
                        group.startswith(k) and str(v) in group
                        for k, v in p.metadata.items()
                    )
                ]
            )
            >= 5
        ]

        # Calculate mean and std using all groups with enough passages
        f1_scores = df.loc[
            (df["Agreement at"] == "Passage level")
            & (df["Group"].isin(groups_with_passages)),
            "F1 score",
        ].values
        f1_mean = f1_scores.mean()
        f1_std = f1_scores.std()

        # Find only the groups which are performing significantly worse than the mean
        outlier_groups = df.loc[
            (df["F1 score"] < (f1_mean - f1_std))
            & (df["Group"].isin(groups_with_passages))
        ]

        html_content += f"""
            <div class="mt-8 bg-white dark:bg-gray-800 rounded-lg shadow-md p-6">
                <h2 class="text-lg font-semibold text-slate-800 dark:text-slate-100 mb-4">
                    Poorly performing groups for {concept}
                </h2>
                <p class="text-slate-600 dark:text-slate-300 mb-4">
                    Overall performance: Precision={passage_metrics["Precision"]:.2f}, 
                    Recall={passage_metrics["Recall"]:.2f}, 
                    F1={passage_metrics["F1 score"]:.2f}
                </p>
        """

        # Create outlier groups table
        outlier_data = []
        for _, row in outlier_groups.iterrows():
            group_parts = row["Group"].split(": ")
            if len(group_parts) == 2:
                group, value = group_parts
            else:
                group = group_parts[0]
                value = group_parts[0]
            n_passages = len(
                [
                    passage
                    for passage in concept.labelled_passages
                    if group in passage.metadata and passage.metadata[group] == value
                ]
            )
            outlier_data.append(
                {
                    "Group": group,
                    "Value": value,
                    "Number of passages": str(n_passages),
                    "Precision": f"{row['Precision']:.2f}",
                    "Recall": f"{row['Recall']:.2f}",
                    "F1": f"{row['F1 score']:.2f}",
                }
            )

        if outlier_data:
            outlier_df = pd.DataFrame(outlier_data)
            html_content += dataframe_to_html_table(outlier_df, "mb-6")

            # Display example passages from each outlier group
            for _, row in outlier_groups.iterrows():
                group_parts = row["Group"].split(": ")
                if len(group_parts) == 2:
                    group, value = group_parts
                else:
                    group = group_parts[0]
                    value = group_parts[0]
                html_content += f"""
                    <div class="mt-6">
                        <h3 class="text-md font-semibold text-slate-700 dark:text-slate-200 mb-2">
                            Examples from {group}: {value}
                        </h3>
                """

                passages = []
                for passage in concept.labelled_passages:
                    metadata = passage.metadata
                    if group in metadata and metadata[group] == value:
                        passages.append(passage)

                html_content += f"""
                    <p class="text-slate-600 dark:text-slate-300 mb-4">
                        Found {len(passages)} passages in this group:
                    </p>
                """

                for i, passage in enumerate(passages):
                    html_content += f"""
                        <div class="mb-6 bg-slate-50 dark:bg-gray-700 p-4 rounded-lg">
                            <p class="font-medium text-slate-700 dark:text-slate-200 mb-2">{
                        i + 1
                    }. Ground truth:</p>
                            <p class="text-slate-600 dark:text-slate-300 mb-4 font-mono">
                                {
                        passage.get_highlighted_text(
                            start_pattern='<span class="bg-yellow-200 dark:bg-yellow-800">',
                            end_pattern="</span>",
                        )
                    }
                            </p>
                            
                            <p class="font-medium text-slate-700 dark:text-slate-200 mb-2">Prediction:</p>
                    """

                    prediction = LabelledPassage(
                        text=passage.text, spans=classifier.predict(passage.text)
                    )
                    highlighted = prediction.get_highlighted_text(
                        start_pattern='<span class="bg-yellow-200 dark:bg-yellow-800">',
                        end_pattern="</span>",
                    )
                    html_content += f"""
                            <p class="text-slate-600 dark:text-slate-300 font-mono">
                                {highlighted}
                            </p>
                        </div>
                    """

                html_content += "</div>"

        html_content += "</div>"

    # Close HTML tags
    html_content += """
        </div>
    </div>
    </body>
    </html>
    """

    # Write the HTML file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html_content)


if __name__ == "__main__":
    app()
