from pathlib import Path
from typing import Annotated

from rich.console import Console
from typer import Option, Typer

from scripts.config import processed_data_dir
from src.labelled_passage import LabelledPassage
from src.sampling import SamplingConfig
from src.span import Span, spans_are_similar

console = Console()


app = Typer()


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
            f"with {n_annotations} individual annotations"
        )

        agreements = []
        disagreements = []
        for labelled_passage in labelled_passages:
            processed_spans = set()
            agreeing_spans = []
            disagreeing_spans = []

            for span in labelled_passage.spans:
                if span in processed_spans:
                    continue

                agreement_found = False
                for other_span in labelled_passage.spans:
                    if (
                        span != other_span
                        and spans_are_similar(span, other_span, threshold)
                        and span.identifier == other_span.identifier
                        and span.labeller != other_span.labeller
                    ):
                        agreement_found = True
                        agreeing_span = Span(
                            # We take the union of the two as the new, agreed-upon span
                            start_index=min(span.start_index, other_span.start_index),
                            end_index=max(span.end_index, other_span.end_index),
                            identifier=span.identifier,
                            labeller=span.labeller,
                        )
                        agreeing_spans.append(agreeing_span)
                        processed_spans.add(span)
                        processed_spans.add(other_span)
                        break

                if not agreement_found and span not in processed_spans:
                    disagreeing_spans.append(span)
                    processed_spans.add(span)

            agreements.append(
                labelled_passage.model_copy(update={"spans": agreeing_spans})
            )
            disagreements.append(
                labelled_passage.model_copy(update={"spans": disagreeing_spans})
            )

        n_agreements = sum(len(entry.spans) for entry in agreements)
        n_disagreements = sum(len(entry.spans) for entry in disagreements)

        # dump the agreements and disagreements to separate jsonl files
        agreements_path = data_dir / "agreements.json"
        with open(agreements_path, "w") as f:
            f.writelines([entry.model_dump_json() + "\n" for entry in agreements])

        console.log(
            f"🤝 Filtered for agreement between annotators. Found {n_agreements}. "
            f"Wrote passages to {agreements_path}"
        )

        disagreements_path = data_dir / "disagreements.json"
        with open(disagreements_path, "w") as f:
            f.writelines([entry.model_dump_json() + "\n" for entry in disagreements])

        console.log(
            f"⚔️ Filtered for disagreement between annotators. Found {n_disagreements}. "
            f"Wrote passages to {disagreements_path}"
        )


if __name__ == "__main__":
    app()
