from enum import Enum
from pathlib import Path
from typing import Annotated

import pandas as pd
from rich.console import Console
from typer import Option, Typer

from scripts.config import processed_data_dir
from src.labelled_passage import LabelledPassage
from src.sampling import SamplingConfig
from src.span import merge_overlapping_spans

console = Console(highlight=False)


app = Typer()


class OutputFormat(Enum):
    """Allowed output formats"""

    JSON = "json"
    CSV = "csv"


@app.command()
def main(
    config_path: Annotated[Path, Option(..., help="Path to the sampling config")],
    threshold: Annotated[
        float,
        Option(
            ...,
            help=(
                "Jaccard similarity threshold for filtering. A value of 0.5 means that "
                "two spans are considered similar if one span shares at least half of "
                "its tokens with the other span. 1 corresponds to an exact match, "
                "while 0 corresponds to no overlap. Use a very small value to allow "
                "for matches with very little overlap."
            ),
            min=0,
            max=1,
        ),
    ] = 0.5,
    format: Annotated[
        OutputFormat,
        Option(
            ...,
            help="The format of the output file. Either 'json' or 'csv'",
            case_sensitive=False,
        ),
    ] = "json",
):
    """
    Filter labelled passages for spans where annotators agree

    The script uses a supplied configuration file to determine which datasets to process.

    Before running this script, you should have run the `save_labelled_passages_from_argilla.py`
    """

    console.log(f"⚙️ Loading config from {config_path}")
    config = SamplingConfig.load(config_path)
    console.log("✅ Config loaded")
    console.log(f"👀 Filtering for Jaccard similarity >= {threshold}")

    for wikibase_id in config.wikibase_ids:
        console.log(f"🤔 Processing {wikibase_id}")
        data_dir = processed_data_dir / "labelled_passages" / wikibase_id
        labelled_passages_path = data_dir / "labelled_passages.jsonl"

        if not labelled_passages_path.exists():
            console.log(
                f"❌ {labelled_passages_path} does not exist. To fetch the data, "
                "run the `save_labelled_passages_from_argilla.py` script using the "
                "same config."
            )
            break

        with open(labelled_passages_path, "r", encoding="utf-8") as f:
            labelled_passages = [
                LabelledPassage.model_validate_json(line) for line in f
            ]
        n_annotations = sum([len(entry.spans) for entry in labelled_passages])
        console.log(
            f"🚚 Loaded {len(labelled_passages)} labelled passages "
            f"with {n_annotations} individual annotations"
        )

        # for any spans with a jaccard similarity >= threshold, add their union to the
        # labelled_passages_with_agreements list. Any spans which do not have a match
        # are added to the labelled_passages_with_disagreements list.
        labelled_passages_with_agreements: list[LabelledPassage] = []
        labelled_passages_with_disagreements: list[LabelledPassage] = []

        for labelled_passage in labelled_passages:
            agreements = []
            disagreements = []

            merged_spans = merge_overlapping_spans(labelled_passage.spans, threshold)
            for span in merged_spans:
                if len(span.labellers) > 1:
                    agreements.append(span)
                else:
                    disagreements.append(span)

            labelled_passages_with_agreements.append(
                labelled_passage.model_copy(update={"spans": agreements}, deep=True)
            )
            labelled_passages_with_disagreements.append(
                labelled_passage.model_copy(update={"spans": disagreements}, deep=True)
            )

        console.log("🤝 Filtered for agreement between annotators.")

        if format == OutputFormat.JSON:
            # dump the agreements and disagreements to separate jsonl files
            agreements_path = data_dir / "agreements.json"
            with open(agreements_path, "w", encoding="utf-8") as f:
                f.writelines(
                    [
                        entry.model_dump_json() + "\n"
                        for entry in labelled_passages_with_agreements
                    ]
                )

            console.log(
                f"📝 Found {len([span for entry in labelled_passages_with_agreements for span in entry.spans])} spans which agree. "
                f"Wrote passages to {agreements_path}"
            )

            disagreements_path = data_dir / "disagreements.json"
            with open(disagreements_path, "w", encoding="utf-8") as f:
                f.writelines(
                    [
                        entry.model_dump_json() + "\n"
                        for entry in labelled_passages_with_disagreements
                    ]
                )

            console.log(
                f"📝 Found {len([span for entry in labelled_passages_with_disagreements for span in entry.spans])} spans which disagree. "
                f"Wrote passages to {disagreements_path}",
                end="\n\n",
            )
        elif format == OutputFormat.CSV:
            agreements_path = data_dir / "agreements.csv"
            disagreements_path = data_dir / "disagreements.csv"
            pd.DataFrame(
                [
                    span.model_dump()
                    for labelled_passage in labelled_passages_with_agreements
                    for span in labelled_passage.spans
                ],
            ).to_csv(agreements_path, index=False)
            console.log(
                f"📝 Found {len([span for entry in labelled_passages_with_agreements for span in entry.spans])} spans which agree. "
                f"Wrote passages to {agreements_path}"
            )
            pd.DataFrame(
                [
                    span.model_dump()
                    for labelled_passage in labelled_passages_with_disagreements
                    for span in labelled_passage.spans
                ],
            ).to_csv(disagreements_path, index=False)
            console.log(
                f"📝 Found {len([span for entry in labelled_passages_with_disagreements for span in entry.spans])} spans which disagree. "
                f"Wrote passages to {disagreements_path}"
            )


if __name__ == "__main__":
    app()
