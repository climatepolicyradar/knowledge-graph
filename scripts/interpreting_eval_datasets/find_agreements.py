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


class AGREEMENT_TYPES(Enum):
    """Types of inter-annotator agreement"""

    EXACT = "exact"
    OVERLAP = "overlap"
    JACCARD = "jaccard"


@app.command()
def main(
    config_path: Annotated[Path, Option(..., help="Path to the sampling config")],
    agreement_type: Annotated[
        AGREEMENT_TYPES,
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
    ] = AGREEMENT_TYPES.JACCARD,
    jaccard_threshold: Annotated[
        float, Option(..., help="Jaccard threshold for filtering")
    ] = 0.5,
    find_disagreements: Annotated[
        bool, Option(..., help="Find disagreements as well as agreements")
    ] = False,
):
    """
    Filter labelled passages for spans where annotators agree

    The script uses a supplied configuration file to determine which datasets to process.

    Before running this script, you should have run the `save_labelled_passages_from_argilla.py`
    """

    console.log(f"⚙️ Loading config from {config_path}")
    config = SamplingConfig.load(config_path)
    console.log("✅ Config loaded")
    if agreement_type == AGREEMENT_TYPES.JACCARD:
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

        agreements = []
        disagreements = []
        for labelled_passage in labelled_passages:
            agreeing_spans = set()
            disagreeing_spans = set()
            for span in labelled_passage.spans:
                agreement_with_at_least_one_other_labeller = False
                for other_span in labelled_passage.spans:
                    if agreement_type == AGREEMENT_TYPES.EXACT:
                        labellers_agree = spans_match_exactly(span, other_span)
                    elif agreement_type == AGREEMENT_TYPES.OVERLAP:
                        labellers_agree = spans_overlap(span, other_span)
                    elif agreement_type == AGREEMENT_TYPES.JACCARD:
                        labellers_agree = spans_are_similar(
                            span, other_span, jaccard_threshold
                        )

                    if (
                        labellers_agree
                        and span.identifier == other_span.identifier
                        and span.labeller != other_span.labeller
                    ):
                        agreement_with_at_least_one_other_labeller = True
                        agreeing_spans.add(
                            Span(
                                # Take the union of the two spans as the new span that
                                # they agree on
                                start_index=min(
                                    span.start_index, other_span.start_index
                                ),
                                end_index=max(span.end_index, other_span.end_index),
                                identifier=span.identifier,
                                labeller="all",
                            )
                        )

                if not agreement_with_at_least_one_other_labeller:
                    disagreeing_spans.add(span)

            agreements.append(
                labelled_passage.model_copy(update={"spans": list(agreeing_spans)})
            )

            disagreements.append(
                labelled_passage.model_copy(update={"spans": list(disagreeing_spans)})
            )

        n_agreements = sum([len(entry.spans) for entry in agreements])
        console.log(
            "🤝 Filtered annotations for agreement between annotators. "
            f"Found {n_agreements}"
        )

        # dump the agreed-upon annotations to a jsonl file
        agreements_path = data_dir / "agreements.json"
        with open(agreements_path, "w") as f:
            f.writelines([entry.model_dump_json() + "\n" for entry in agreements])

        console.log(f"📝 Wrote {len(agreements)} passages to {agreements_path}")

        if find_disagreements:
            n_disagreements = sum([len(entry.spans) for entry in disagreements])
            console.log(
                "⚔️ Filtered annotations for disagreement between annotators. "
                f"Found {n_disagreements}"
            )

            # dump the disagreeing annotations to a jsonl file
            disagreements_path = data_dir / "disagreements.json"
            with open(disagreements_path, "w") as f:
                f.writelines(
                    [entry.model_dump_json() + "\n" for entry in disagreements]
                )

            console.log(
                f"📝 Wrote {len(disagreements)} passages to {disagreements_path}"
            )


if __name__ == "__main__":
    app()
