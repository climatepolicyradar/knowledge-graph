"""Filter for spans which have an IOU of at least 0.5 with at least one other annotator"""

from pathlib import Path

import typer
from rich.console import Console

from scripts.config import processed_data_dir
from src.labelled_passage import LabelledPassage
from src.sampling import SamplingConfig
from src.span import Span, jaccard_similarity

console = Console()

app = typer.Typer()


@app.command()
def main(config_path: Path):
    console.log(f"⚙️ Loading config from {config_path}")
    config = SamplingConfig.load(config_path)
    console.log("✅ Config loaded")

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

        # produce a golden dataset by filtering the labelled passages to only include
        # the spans on which annotators agree (that is, the spans that overlap with at
        # least one other annotator, with an IOU of at least 0.5).
        similar_annotations = []
        for labelled_passage in labelled_passages:
            similar_spans = set()
            for span in labelled_passage.spans:
                for other_span in labelled_passage.spans:
                    if (
                        jaccard_similarity(span, other_span) >= 0.5
                        and span.identifier == other_span.identifier
                        and span.labeller != other_span.labeller
                    ):
                        # create a new span which is the union of the two similar spans
                        similar_spans.add(
                            Span(
                                start_index=min(
                                    span.start_index, other_span.start_index
                                ),
                                end_index=max(span.end_index, other_span.end_index),
                                identifier=span.identifier,
                                labeller="all",
                            )
                        )
            similar_annotations.append(
                labelled_passage.model_copy(update={"spans": list(similar_spans)})
            )

        n_similar_annotations = sum([len(entry.spans) for entry in similar_annotations])
        console.log(
            "🤝 Filtered annotations for strong agreement between annotators. "
            f"Found {n_similar_annotations}"
        )

        # dump the annotations to a jsonl file
        similar_annotations_path = data_dir / "similar_annotations.json"
        with open(similar_annotations_path, "w") as f:
            f.writelines([entry.model_dump_json() for entry in labelled_passages])

        console.log(
            f"📝 Wrote {len(similar_annotations)} passages to "
            f"{similar_annotations_path}"
        )


if __name__ == "__main__":
    app()
