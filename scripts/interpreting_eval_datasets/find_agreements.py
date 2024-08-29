from enum import Enum
from pathlib import Path
from typing import Annotated

from rich.console import Console
from typer import Option, Typer

from scripts.config import processed_data_dir
from src.labelled_passage import LabelledPassage
from src.sampling import SamplingConfig
from src.span import Span, spans_are_similar, spans_match_exactly, spans_overlap

console = Console()


app = Typer()


class IAA_TYPES(Enum):
    """Types of inter-annotator agreement"""

    EXACT = "exact"
    OVERLAP = "overlap"
    JACCARD = "jaccard"


@app.command()
def main(
    config_path: Annotated[Path, Option(..., help="Path to the sampling config")],
    similarity_type: Annotated[
        IAA_TYPES,
        Option(
            ...,
            help=(
                "Type of similarity to filter for. 'exact' for spans whose start and "
                "end indices match exactly between annotators, 'overlap' for spans that"
                "that overlap with at least one other annotator, 'jaccard' for spans "
                "between annotators which have a Jaccard similarity greater than a "
                "supplied threshold (see jaccard_threshold)."
            ),
        ),
    ] = IAA_TYPES.JACCARD,
    jaccard_threshold: Annotated[
        float, Option(..., help="Jaccard threshold for filtering")
    ] = 0.5,
):
    """
    Filter labelled passages for spans where annotators agree

    The script uses a supplied configuration file to determine which datasets to process.

    Before running this script, you should have run the `save_labelled_passages_from_argilla.py`
    """

    console.log(f"⚙️ Loading config from {config_path}")
    config = SamplingConfig.load(config_path)
    console.log("✅ Config loaded")
    if similarity_type == IAA_TYPES.JACCARD:
        console.log(f"👀 Filtering for Jaccard similarity >= {jaccard_threshold}")

    for wikibase_id in config.wikibase_ids:
        console.log(f"🔍 Processing {wikibase_id}")
        data_dir = processed_data_dir / "labelled_passages" / wikibase_id
        labelled_passages_path = data_dir / "labelled_passages.jsonl"

        if not labelled_passages_path.exists():
            console.log(
                f"❌ {labelled_passages_path} does not exist. To fetch the data, "
                "run the `save_labelled_passages_from_argilla.py` script using the "
                "same config."
            )
            break

        with open(labelled_passages_path, "r") as f:
            labelled_passages = [
                LabelledPassage.model_validate_json(line) for line in f
            ]
        n_annotations = sum([len(entry.spans) for entry in labelled_passages])
        console.log(
            f"✅ Loaded {len(labelled_passages)} labelled passages "
            f"with {n_annotations} annotations"
        )

        agreed_upon_annotations = []
        for labelled_passage in labelled_passages:
            agreed_upon_spans = set()
            for span in labelled_passage.spans:
                for other_span in labelled_passage.spans:
                    agreements = {
                        IAA_TYPES.EXACT: spans_match_exactly(span, other_span),
                        IAA_TYPES.OVERLAP: spans_overlap(span, other_span),
                        IAA_TYPES.JACCARD: spans_are_similar(
                            span, other_span, jaccard_threshold
                        ),
                    }
                    if (
                        agreements[similarity_type]
                        and span.identifier == other_span.identifier
                        and span.labeller != other_span.labeller
                    ):
                        agreed_upon_spans.add(
                            Span(
                                start_index=min(
                                    span.start_index, other_span.start_index
                                ),
                                end_index=max(span.end_index, other_span.end_index),
                                identifier=span.identifier,
                                labeller="all",
                            )
                        )

            agreed_upon_annotations.append(
                labelled_passage.model_copy(update={"spans": list(agreed_upon_spans)})
            )

        n_agreed_upon_annotations = sum(
            [len(entry.spans) for entry in agreed_upon_annotations]
        )
        console.log(
            "🤝 Filtered annotations for agreement between annotators. "
            f"Found {n_agreed_upon_annotations}"
        )

        # dump the agreed-upon annotations to a jsonl file
        agreed_upon_annotations_path = data_dir / "agreements.json"
        with open(agreed_upon_annotations_path, "w") as f:
            f.writelines(
                [entry.model_dump_json() + "\n" for entry in labelled_passages]
            )

        console.log(
            f"📝 Wrote {len(agreed_upon_annotations)} passages to "
            f"{agreed_upon_annotations_path}"
        )


if __name__ == "__main__":
    app()
